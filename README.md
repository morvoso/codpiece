# codpiece

A local LLM server for one specific setup: **two RTX 3090s** running
**Qwen3.8-27B**. It does the same job as llama.cpp or vLLM — you send it
prompts over HTTP, it streams back answers — but it is tuned for exactly this
hardware and this model, and on that ground it is faster than both at the
same accuracy. It speaks the OpenAI API, so anything that works with OpenAI,
llama.cpp, or Ollama endpoints works with it unchanged: chat clients, coding
assistants, scripts. It also reads images.

What "tuned" buys you, in plain terms:

- **Faster replies.** 67–88 tokens/s where llama.cpp gets 53–65 on the same
  GPUs, because it drafts several tokens ahead and verifies them in one pass.
- **Same answers.** With randomness off it produces byte-for-byte the same
  text llama.cpp does. Speed never comes from cutting corners on the model.
- **Instant return to old conversations.** Revisiting a long chat costs ~1
  second instead of ~25, because conversation state stays parked in GPU
  memory.
- **Images.** Screenshots and photos through the standard `image_url` chat
  field, answered by the same model.

## How to use

### Build

Requires Rust, CMake, and CUDA. First fetch the pinned ggml sources (the
compute kernels are vendored from llama.cpp, not reimplemented):

```sh
scripts/fetch-deps.sh
cargo build --release --features cuda
```

Omit `--features cuda` for a CPU-only build (useful for tests; far too slow
to serve the 27B).

### Try it from the command line

```sh
# one-shot generation
codpiece gen model.gguf -p "Hello" -n 64 -c 8192 --tp 0,1

# inspect any GGUF without loading it
codpiece inspect model.gguf
```

`--tp 0,1` splits the model across both GPUs (tensor parallel). The 27B does
not fit on one card.

### Start the server

```sh
codpiece serve model.gguf --host 0.0.0.0 --port 8020 -c 65536 --tp 0,1 \
    --mmproj mmproj-BF16.gguf --mmproj-gpu 0
```

`-c` is the context size in tokens. `--mmproj` enables image input; leave it
off for text only.

### Talk to it

Any OpenAI-compatible client works. Raw curl:

```sh
# plain completion
curl http://localhost:8020/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'

# chat (the model's own template is applied server-side)
curl http://localhost:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Explain BPE in one paragraph"}],
       "max_tokens": 200, "stream": true}'
```

Images go in the standard content-parts form. Only `data:` URLs are
accepted — the server never fetches remote URLs:

```json
{"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}},
    {"type": "text", "text": "What does this screenshot show?"}
]}]}
```

Tool calling works as it does against OpenAI. Pass `tools`, get
`tool_calls` back with `finish_reason: "tool_calls"`, send the assistant
turn and the tool result back for the next turn:

```json
{"messages": [{"role": "user", "content": "Weather in Berlin?"}],
 "tools": [{"type": "function", "function": {
     "name": "get_weather",
     "parameters": {"type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]}}}]}
```

Qwen3.8 emits calls in an XML-ish framing of its own
(`<tool_call><function=name><parameter=key>`), not the JSON blob earlier
Qwen generations used; the server parses that back into the OpenAI shape.
Argument types follow the schema you sent — a parameter declared `string`
stays a string even when it looks numeric, everything else is parsed as
JSON.

For evaluation harnesses, `/v1/completions` implements `echo` +
`logprobs`: the prompt is scored token by token and returned with
per-token logprobs, top-k alternatives and text offsets, which is what
loglikelihood benchmarks (MMLU, HellaSwag, ARC) run on. The prompt may
also be sent as an array of token ids so a harness keeps its own
tokenization authoritative.

```sh
curl http://localhost:8020/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is Paris.",
       "max_tokens": 0, "echo": true, "logprobs": 3}'
```

Endpoints: `/v1/completions`, `/v1/chat/completions` (both stream with
`"stream": true`), `/tokenize`, `/detokenize`, `/v1/models`, `/health`,
`/slots`. Responses include a `timings` object with prefill/decode speed and
speculation acceptance.

Thinking models are handled the way the Qwen API does it: the `<think>`
block comes back in `reasoning_content`, the answer in `content`, in both
streaming and non-streaming responses. Qwen warns that greedy decoding
with thinking on degenerates into repetition, so a thinking request that
sets *no* sampling fields adopts the sampling the model itself recommends
(`general.sampling.*` in the GGUF: temp 1, top_k 20, top_p 0.95). Set any
sampling field — including `"temperature": 0` — and your choice stands.

### Settings that matter

| knob | default | what it does |
|---|---|---|
| `--depth N` | 3 | speculation depth; `0` disables, `auto` self-tunes |
| `--think-budget N` | 4096 | max tokens in a `<think>` block before it is force-closed |
| `--alias NAME` | file stem | model id reported to clients |
| `CODPIECE_CHAIN_PMIN` | 0 (off) | confidence gate on carried drafts; raise to trade speed for acceptance ratio |
| `CODPIECE_REDRAFT_PMIN` | 0.75 | confidence gate on re-drafts, which cost real time; 0.75 is the measured optimum |
| `CODPIECE_SESSIONS` | 2 (≤70K ctx) | conversations kept resident in VRAM |
| `CODPIECE_BATCH` | 4 | slots for concurrent requests |
| `CODPIECE_MMPROJ` | unset | vision tower path (alternative to `--mmproj`) |
| `--dflash PATH` / `CODPIECE_DFLASH` | unset | DFlash2 block drafter: +22-37% greedy decode (needs ~1.9 GB VRAM for the Q8 draft) |
| `CODPIECE_IMAGE_MIN_TOKENS` | 1024 | image detail floor; Qwen-VL reads text poorly below this |
| `CODPIECE_IMAGE_MAX_TOKENS` | from free VRAM | image detail ceiling; see below |
| `CODPIECE_GRAPH_CACHE_MIB` | 1024 | VRAM the per-session cache of compiled graph shapes may hold |
| `CODPIECE_BATCH_GREEDY` | on | all-greedy batch rounds argmax in the graph instead of reading back full logits |

The image ceiling is derived from measured headroom rather than fixed,
because the vision tower runs on its own backend with its own CUDA pool —
and a pool never returns memory to the driver, so whatever the encoder
peaks at is taken from the trunk for the life of the process. One
oversized image can leave a card with nothing left and every later request
failing to allocate. The server prints the range it chose (`serve: image
tokens 1024..1024`) and says so when the ceiling lands below Qwen-VL's
1024-token grounding floor.

Diagnostics, all off by default: `CODPIECE_BATCH_TRACE=1` (per-round
fill/compute/readback at width N), `CODPIECE_DECODE_TRACE=1` (single-stream
verify vs draft time and tokens committed per round), `CODPIECE_TRACE_VRAM=1`
(a VRAM line per request), `CODPIECE_TRACE_MEM=1` (compute buffer size per
compiled graph shape), `CODPIECE_TRACE_REUSE=1` (why a prefix-cache lookup
missed). The server also reports per-GPU VRAM and its chosen prefill chunk
at startup — on a 48 GiB box, every context decision is a memory decision.

The defaults maximize throughput. Acceptance ratio is a conversion
statistic, not a quality metric — the output is identical at any setting —
so the gates only filter drafts whose *cost* matters: carried chain links
are free to verify (gate off), re-drafts cost a ~6 ms pass each (gate 0.75).
Setting both to 0.9 makes the ratio read >= 0.90 at ~15% decode cost.

### Deploying

`deploy/docker-compose.codpiece.yml` runs the server in the CUDA container
this project builds in; `deploy/switch.sh {codpiece|llamacpp|vllm}` swaps
engines on port 8020 so rollback is one command. All tunables are
`CODPIECE_`-prefixed environment variables so a shared `.env` with other
engines cannot bleed settings in.

## Measured performance

Same box, same model (Qwen3.8-27B UD-Q8_K_XL), same prompts, locked bench
windows. llama.cpp is b10423 in its production configuration.

| measurement | codpiece | llama.cpp | vLLM (FP8) |
|---|---|---|---|
| greedy decode, prose | **71.2 tok/s** | 64.5 | 76–95 |
| greedy decode, code | **82.2** | 53.3 | — |
| greedy decode, arithmetic | **117.4** | — | — |
| sampled (temp 1.0), short | **51.2** | 32.4 | — |
| sampled, 32K context | **53.7** | 52.7 | — |
| prefill, 32K prompt | 1,337 | 1,353 | — |
| aggregate throughput (32 streams) | **285.7** | ~101 (8-way) | 251–260 |
| return to a 32K conversation | **0.9 s** | 1.3 s | — |
| needle retrieval at 97K tokens | **9/9 across depths** | — | — |
| speculation acceptance | 0.64–0.73 (0.91–0.93 opt-in) | ~0.78 | — |
| vision (same page, same question) | same answer, 847 tok/s prefill | same, 837 | n/a |
| greedy output parity | byte-identical | reference | not comparable (FP8) |

The greedy rows are measured with the DFlash2 block drafter
(`--dflash`), which drafts up to 7 tokens per call through a 1.9B
block-diffusion model and wins every greedy prompt class by 22-37%;
without it the MTP-head numbers are 58-86 by class. Sampled decode
routes to the gumbel-coupled chain automatically — measured better
there.

Honest caveats: sampled decode at 32K now runs 53.7 vs llama.cpp's 52.7
(gumbel-coupled chains closed its last speed win); its one remaining edge
is context reach — llama.cpp will serve 196K by quantizing KV to fp8,
which is an accuracy trade this engine declines, so codpiece ships 98K
with f16 KV and the full stack resident. Its higher acceptance ratio is
denominator filtering (`--spec-draft-p-min 0.75` declines low-confidence
drafts), not better text. The aggregate row uses 32 batch slots
(`CODPIECE_BATCH=32 CODPIECE_BATCH_CTX=1536`); vLLM needs FP8 weight
precision for its 251-260, which codpiece passes at Q8 with the
sensitive tensors in BF16.

At 32-way the round is compute bound — 64.3 ms against a 15.6 ms
weight-read floor — so wider batching is near its knee on this hardware,
and speculation (which pays only while a round is memory bound) does not
help there. The remaining aggregate headroom is quantized-matmul kernel
efficiency, not scheduling: per-round host work measures 0.02 ms.

## Technical details

### What it is

codpiece replaces the engine layer — scheduling, batching, KV and
recurrent-state management, speculative decoding, session caching, the HTTP
server — and keeps ggml's CUDA kernels underneath, vendored at the exact
llama.cpp build production ran (b10423) plus a small patch set
(`patches/ggml-codpiece.patch`). Accuracy against llama.cpp is therefore a
bit-parity property of shared kernels, checked by gates, not an aspiration.

The model is Qwen3.8-27B: 64 trunk layers, 48 of them Gated-DeltaNet
recurrences and 16 attention, plus an embedded MTP draft head. The recurrent
majority is why decode barely slows with context (fixed-size state) and why
rejected speculative tokens need explicit state rollback: the fused GDN
kernel emits K trailing state snapshots, and rollback promotes one.

### The bandwidth argument

Single-stream decode is memory-bound: ~14.7 GiB of weights per GPU per token
against 936 GB/s is a 15.7 ms floor, ~64 tok/s. Speculation is the only way
past it — one weight read verifies several drafted tokens, so the floor
applies per round, not per token. The whole optimization problem is accepted
tokens per round. Details and measurements: `ENGINE.md`,
`docs/ARCHITECTURE.md`.

### Speculation

The fused round runs verification and the next draft chain in one graph,
including in-graph argmax→embedding chaining so the chain conditions on the
trunk's own prediction without a round trip. At temperature, drafts are
accepted with probability p(draft) and rejections re-drawn from the residual,
which provably emits the target distribution unchanged (verified on the 27B:
total-variation test/null ratio 1.039 over 24K sampled tokens).

Two confidence gates filter drafts by cost, not by ratio. The fused graph
emits each chain link's softmax peak (a batched `get_rows` over the softmax
viewed as `[1, vocab, n_out]`); `CODPIECE_CHAIN_PMIN` gates carried chain
links and defaults off, because a link computed last round is nearly free to
verify and rejecting a 60%-likely free draft loses tokens (gating the chain
at 0.9 measured -17% at 32K). `CODPIECE_REDRAFT_PMIN` gates post-divergence
re-drafts, which cost a real ~6 ms MTP pass each; 0.75 is the measured
optimum. Since verification accepts with probability p(x0), setting both to
0.9 pushes the measured ratio to 0.91–0.93 for deployments that want that
number — the output is identical either way.

Speculative output is not always byte-identical to one-token-at-a-time
greedy decoding: batched verification reduces GEMMs in a different order, and
a near-tied argmax can flip. llama.cpp's own MTP speculation diverges from
its own greedy output at the same token on the same prompt
(`scripts/specparity-payload.sh`). Non-speculative greedy output is gated
byte-identical against llama.cpp, and that gate holds.

### Serving

One engine thread owns the model (ggml pointers are not Send). Single
requests take a speculative fast path; overlapping requests move to a batch
scheduler with fixed-width rounds and per-slot samplers, where the
recurrent-state slot dimension doubles as the sequence dimension (batch mode
does not speculate, so snapshots and sequences never coexist). Prefill runs
bulk chunks through the trunk only and the last 64 tokens through the fused
round to warm the draft head; chunk size follows VRAM headroom, not context.

Conversations stay resident: a pool of whole sessions in VRAM with
longest-prefix matching (tensor-parallel GDN state cannot be copied
off-device, so the pool switches by pointer), plus a host-RAM snapshot store
on single-device builds. Multi-turn thinking chats round-trip exactly — the
server splits generations at `</think>` into `reasoning_content` the way the
template re-renders them, so a returning conversation is a token-exact prefix
and costs ~1 s instead of a full re-prefill.

### Vision

`crates/codpiece-vision` is an op-for-op port of llama.cpp's qwen3vl_merger
clip graph: dual-conv patch embedding, 2×2 block reorder, bilinearly resized
position embeddings, 27 transformer layers with vision M-RoPE, and the
merger MLP into the trunk width. Encoder parity vs `llama-mtmd-debug` on the
same build: the ported front end is bit-exact, full-depth drift stays under
BF16 weight precision. Preprocessing (smart-resize, PAD_CEIL bilinear,
normalization) mirrors mtmd's rounding exactly.

Image embeddings enter the trunk through an embedding-input graph with the
Qwen-VL position rule — the RoPE clock advances by max(grid_w, grid_h) while
the KV cache advances by the token count; the gap is tracked per session —
and image spans appear in the prompt cache as content-hash pseudo-ids, so
prefix reuse matches an image only when its bytes match. The ViT runs flash
attention on CUDA (the non-FA path materializes a >1 GiB attention matrix at
1024 image tokens); the CPU keeps the exact reference path for parity runs.
End-to-end gate: `scripts/vision-payload.sh` asks llama.cpp and codpiece the
same question about the same newspaper page and requires the same reading.

### Verification

Every capability has a gate script under `scripts/`, run inside
`scripts/bench-window.sh` (locks the GPUs, stops production, restores and
health-checks it afterwards — see `docs/SAFETY.md`):

- `final-verify.sh` — fused speculation vs plain greedy, three prompts, all
  depths
- `serve-payload.sh` — greedy over HTTP must equal the CLI byte-for-byte
- `specparity2-payload.sh` — sampling distribution preservation on the 27B
- `vision-payload.sh` — end-to-end image answer vs llama.cpp
- tokenizer: 297,193/297,193 tokens identical on wikitext-2; perplexity
  20.4453 vs 20.4429

## Layout

- `crates/codpiece-gguf` — dependency-free GGUF reader
- `crates/codpiece-tok` — BPE tokenizer (token-identical to llama.cpp)
- `crates/codpiece-model` — weight loading, graphs, sessions, speculation,
  tensor-parallel split rules
- `crates/codpiece-vision` — image encoder and preprocessing
- `crates/codpiece-sample` — samplers, llama.cpp-compatible semantics
- `crates/codpiece-server` — HTTP server, OpenAI API, engine thread, batch
  scheduler
- `crates/codpiece-cli` — `codpiece {serve,gen,inspect,vision,...}`
- `docs/` — architecture, roadmap (M0–M7, each with an accuracy gate),
  safety protocol
- `ENGINE.md`, `notes/` — the measured knowledge base under the design

## License

MIT. Vendored ggml/llama.cpp sources are MIT (ggml-org).
