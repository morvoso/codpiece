# codpiece roadmap

Every milestone has a **gate**: a measurable pass/fail check on llm-host.
A milestone without a passed gate is not done. Speed numbers only count from
benches run via the locked bench window (SAFETY.md) with ≥2 reps (MTP
acceptance noise is ±10% rep-to-rep).

Baselines to beat (ENGINE.md, measured 2026-08-19 unless noted):

| metric | llama.cpp b10423 (prod) | vLLM 0.27.1 |
|---|---|---|
| decode prose/code, 1 stream | 51 / 67 tok/s | 76 / 90 |
| decode @ 27K depth | 52–57 | — |
| prefill 27K | 1,387 tok/s | 1,591 |
| 4-way aggregate | ~101 (np 4) / 73 | 251 |
| MTP acceptance @ n-max 3 | 0.78–0.92 | — |
| KV fidelity @ 196K | f16 | fp8 forced |
| session revisit @ 27K | 1.3 s | 26.9 s (no cache) |

## The goal, as stated by the owner (2026-08-20)

> A fully operational replacement for llama.cpp and vLLM for Qwen3.8-27B, the most
> accurate quant possible, on 2x RTX 3090. **Accuracy first, speed second, context
> size third.**

Note the order: speed now ranks above context. Earlier notes in this repo say
`ACCURACY > CONTEXT > SPEED`; that is superseded.

**On "the most accurate quant possible" — already met, and it is arithmetic.**
Usable VRAM across the two cards is ~47 GiB after the CUDA contexts.

| | weights | + 196K KV | fits |
|---|---|---|---|
| BF16 (27B x 2 B) | 50.3 GiB | 62.3 | **no** |
| **UD-Q8_K_XL (in use)** | **29.3 GiB** | 41.3 | yes |
| plain Q8_0 | 26.6 GiB | 38.6 | yes |
| UD-Q6_K_XL | 23.6 GiB | 35.6 | yes |

Full BF16 does not fit and never will on 48 GiB. The file in use is *not* plain
Q8_0: 16.4% of it is BF16 — the tensors the quantiser judged sensitive, including
`output.weight` — which is why it is 29.3 GiB rather than 26.6. That is the most
accurate weight set that fits, so the accuracy axis is closed at the quant level
and the remaining accuracy work is parity of the engine, which is what every gate
in this document measures.

## M0 — Ground truth + skeleton  ✅ 2026-08-19

- [x] SSH + recon of llm-host (notes/recon-2026-08-19.md)
- [x] Repo, workspace, safety protocol
- [x] `codpiece-gguf` parser + `codpiece inspect`, validated against all four
      production GGUFs (27B Q8_K_XL, Q6_K_XL-v3, DFlash2 Q8, mmproj)
- [x] qwen35 architecture fully mapped from GGUF metadata + llama.cpp source
- [x] Vendored ggml pinned @ b10423; `codpiece-ggml-sys` links and runs CPU
      graphs on both machines (portable-AVX2 build after a -march=native
      illegal-instruction lesson); CUDA sm_86 compile proven via the oracle
      container build on llm-host

**Gate:** inspector census matches llama.cpp's loader (866 tensors, 48 GDN /
16 attn / 1 MTP schedule) ✅; FFI smoke (matmul + fused-GDN op present) ✅.

## M1 — CPU correctness on a small same-arch sibling  ◕ mostly done 2026-08-19

Target upgraded from "dense qwen3" to **Qwen3.5-0.8B** (same qwen35 arch as
prod's 27B: 18 GDN + 6 attn layers): tokenizer → weights → stateless forward
graph (fused GDN, IMROPE, packed Q+gate attention) → greedy sampler.

Status 2026-08-19, all vs oracles built at b10423, CPU, GPU-blind:
- [x] Tokenizer parity: **token-identical** with `llama-tokenize` in both
      parse-special modes on a 236 KB adversarial corpus (68,347 tokens:
      CJK, Arabic diacritics, Devanagari, Zalgo, emoji ZWJ, code). Includes
      the USER_DEFINED-always-parsed subtlety.
- [x] Generation parity: greedy continuation **token-identical to EOS**
      (43 tokens) vs `llama-completion --temp 0` on the wrapped prompt,
      BF16 weights.
- [x] Formal gates CLOSED (session 2): tokenizer identical on wikitext-2
      test (297,193/297,193 tokens, 1.29 MB); **PPL 20.4453 vs oracle
      20.4429 (0.012%)** over 12×512 chunks — the numeric whole-model gate,
      stronger than a logit maxdiff; scripts/parity-gen.sh 3 prompts × 64
      tokens 3/3; EOS stop in run/gen.

**M1: PASSED.**

## M1.5 — hygiene  ✅

- [x] parity harness scripted (scripts/parity-gen.sh, path-matched oracle
      via ORACLE_FA; CODPIECE_SUBCMD selects stateless/session path).
- [x] tokenizer perf closed by measurement: 297K tokens in 0.37 s wall —
      the pretokenizer keeps BPE pieces tiny; no heap needed.
- [x] SIGPIPE fixed. Builder image docker/builder.Dockerfile (CUDA devel +
      rustup + libclang); cargo cuda builds on llm-host in a named volume.

## M2 — Single-GPU CUDA + decode loop  ✅ correctness / ◕ speed 2026-08-19

Stateful engine (Session: KV cache + carried conv/GDN states) on one 3090,
inside the locked bench window (scripts/bench-window.sh — SAFETY.md as code).

Correctness (all PASSED, path-matched vs llama-completion @ b10423):
- [x] selftest 5/5 on CUDA (prefill-all, prefill+decode, decode-only,
      single-token, session-repeat determinism)
- [x] PPL on GPU 16.2114 vs CPU 16.2159 (same 4 chunks)
- [x] 64-token GPU generation parity IDENTICAL
- [x] **2334-token long-context parity IDENTICAL** (chunked prefill,
      positions at depth, KV across chunk boundaries, bucket rollovers)

Speed (0.8B BF16, GPU1, 256-token decode, ≥2 reps):

| step | tok/s | note |
|---|---|---|
| session decode, f32 KV, per-step graph | 235 | correctness baseline |
| + persistent gallocr | 239 | +1.7% |
| + flash attention | 247 | +3% |
| + in-graph argmax (4-byte readback) | 265 | +7%; unlocked the next one |
| + CUDA graphs (GGML_CUDA_GRAPHS=ON) | **295** | +11%; capture was compiled out AND masked by the logits copy |
| llama.cpp b10423, same GPU/model/path | 317 | codpiece = **93%** |

Three levers that did NOT pay: async input uploads (0%), KV bucket 64/128
(worse than 256 — rebuild cost), f32→f32 weight casts (already F32 in file).
Remaining 7% is unexplained; it is not launch overhead, not padding waste,
and not cast nodes. Deferred deliberately: the 27B has ~30× the compute per
step, so fixed overheads matter far less there — profile at real scale.

**Gate:** correctness ✅; speed ≥ oracle NOT met (93%).

## M3 — Qwen3.8-27B on 2 GPUs  ◕ in progress 2026-08-19

The 27B is 29.3 GiB of weights; one 24 GiB card cannot hold it. Multi-GPU is
therefore a correctness requirement, not a speed optimization.

Staged deliberately:
- [x] **M3a — layer split** (`Device::CudaSplit`, ggml_backend_sched):
      weights placed per layer, scheduler inserts cross-device copies over
      PCIe. Verified on the 0.8B: output identical to single-GPU.
      Costs speed (113 vs 295 tok/s on the 0.8B — sched re-plans every step,
      no cached graph, session state all on device 0) but it is the path
      that makes the real model runnable at all.
- [x] **M3b — the production 27B runs on codpiece**: 29.3 GiB placed as
      16.57 + 12.72 GiB across the two 3090s, coherent generation,
      prefill 105 tok/s, decode 19.0 tok/s
- [x] **M3c — 27B output IDENTICAL to llama.cpp**: short prompt under both
      `-sm layer` (matching sha256) and `-sm tensor`, plus a **7,521-token
      prompt under tensor parallel**. 27K remains for the M6 context work.
- [ ] M3d — placement fixes: session KV/state tensors on their layer's
      device (today they all sit on device 0, doubling bus traffic for the
      second half of the stack), cached decode graph under the scheduler
- [x] **M3e — tensor parallel DONE**: `Device::CudaTensorParallel` over
      ggml's meta device, codpiece supplying the split classification.
      **27B decode 39.8 tok/s (3 reps) vs llama.cpp's prod config 41.4
      (2 reps) = 96 %, output identical.** 2.1× the layer-split path.
- [ ] M3d — remaining placement work: session KV/state per device under the
      layer-split path (TP handles this itself), cached decode graph reuse
      measurements under TP.

**Gate: PASSED.** Parity holds on short and 8K prompts under prod's own
split mode, and decode is 39.8 vs the ~39 tok/s no-MTP baseline the gate
asked for (within 4% of llama.cpp measured side by side at 41.4).

## M4 — MTP speculative decoding  ✅ 2026-08-19

blk.64 draft head, hidden-state handoff, K-snapshot rollback across the 48
recurrent layers.

Measured on the 27B, tensor parallel, 96 tokens, all **lossless** (output
identical to plain greedy):

| mode | decode tok/s | acceptance | tokens/round |
|---|---|---|---|
| plain greedy | 40.1 | — | 1.00 |
| MTP depth 1 | 47.9 | 0.900 | 1.92 |
| MTP depth 2 | 52.9 | 0.713 | 2.40 |
| **MTP depth 3** | **59.3** | 0.667 | 3.00 |

Head to head against llama.cpp's own server running production's MTP config,
3 reps each in one window, that first measurement read **llama.cpp 65.0 tok/s,
codpiece 59.2 (91 %)** — codpiece *behind*, unlike the non-speculative case.

**Re-measured 2026-08-20**, after fixing the graph-reuse bugs described in
`docs/OPTIMIZATION-IDEAS.md` §4 (96 tokens, 3 reps, one window):

| engine | tok/s |
|---|---|
| llama.cpp b10423, prod MTP config | 64.52 / 64.78 / 64.28 |
| codpiece, fused round depth 2 | 64.51 / 64.47 / 64.49 |
| **codpiece, fused round depth 3** | **69.86 / 69.81 / 69.89** |

**Gate: PASSED.** codpiece is level with llama.cpp at depth 2 and ~8 % ahead at
depth 3, with output byte-identical to plain greedy decoding at every depth.
The best depth is prompt-dependent — on a 160-token generation depth 2 led at
64.3 while depth 3 gave 62.9 — so an adaptive depth is the obvious next step.

Acceptance-adaptive draft depth is implemented (`--depth auto`) and stable, and
now **matches fixed depth 3 within +-0.4% on four prompts** while choosing the
depth itself — depth is worth up to 14% between adjacent settings, so landing on
the right one matters even when adaptation adds no speed of its own. It stays
opt-in because it does not beat a correctly chosen fixed depth. The fixed
default moved from 1 to 3.

Still open here: prod's `p-min 0.75` idea — dropping low-confidence drafts
rather than always drafting n — measured as neutral earlier but never combined
with the fused cached round, where a dropped draft now actually shortens the
graph.

## M4.5 — Long context  ◕ 2026-08-20

Everything through M4 was measured at `-c 4096`; production serves at 196K.

- [x] **Decode is nearly context-free**, as the hybrid architecture implies —
      48 of 64 layers are recurrences with fixed-size state, so only 16
      attention layers grow. Plain decode falls just ~20% from 1K to 127K
      tokens (39.0 -> 31.4 tok/s); prefill holds ~1,000 tok/s at 127K.
- [x] **Fixed: speculative decoding aborted above ~9K tokens.** The fused round
      asked the trunk for predictions at every position, which at prefill is a
      logits tensor of `n_vocab x n_prompt` — 26 GiB on a 27K prompt. Prefill
      now runs in chunks, with earlier chunks at depth 1 since only the final
      chunk's drafts are read.
- [x] **Speculation at 130K context**: 52.3 tok/s vs plain's 32.1 (1.63x), with
      the prefill chunk scaled to context.
- [x] **Production context reached for plain decode**: a 186,162-token prompt at
      `-c 200704`, 28.1 tok/s decode, 813 tok/s prefill. Scaling the prefill
      chunk was the whole fix.
- [x] **Draft head's KV window bounded** (`CODPIECE_MTP_CTX`, default 16384),
      lossless by construction since drafts are verified; moved the speculative
      ceiling from ~127K to 145K tokens.
- [ ] **Speculation above ~145K**: blocked by `output.weight` (2.37 GiB) and
      `token_embd` (1.26 GiB) being replicated on both cards — 1.81 GiB/card —
      which is what makes in-graph argmax and the embedding gather possible
      under tensor parallelism. Splitting them costs nothing in speed but
      disables the fused round. See `notes/results-2026-08-19.md`.

**Gate: met for plain decode, partial for speculation.** Production's context
runs; speculation runs to 145K. Since M5 shares one KV pool across slots,
4-way at a 196K pool is ~49K per slot, which is inside the speculative range —
so this does not block M5 in practice.

## M5 — Serving layer + continuous batching  ◕ started 2026-08-20

OpenAI-compatible `/v1/chat/completions` + `/completions` + `/slots`-style
introspection, streaming, minijinja template from GGUF, reasoning_effort +
thinking budget, official sampling presets, multi-stream continuous batching.

- [x] **M5a — samplers** (`codpiece-sample`, zero deps). temperature, top-k, top-p,
      min-p, repeat/frequency/presence penalties, seeded xoshiro256++ draw.
      Filter semantics ported from llama.cpp's `llama-sampler.cpp` at the pin, so
      the same request parameters mean the same thing in both engines. 10 unit
      tests; on the 27B, `--top-k 1 --temp 1` reproduces greedy **byte-identically**
      through the host sampling path, which exercises the logits readback, the
      candidate construction and the draw at once. Cost of leaving the in-graph
      argmax: 38.46 vs 39.02 tok/s (~1.5%), so greedy still takes the fast path
      and only non-greedy parameters pay.
      *Parity note:* the draw cannot match llama.cpp token for token — it uses
      `std::mt19937` through `std::discrete_distribution`, whose stream is
      implementation-defined. Token-exact parity stays a temperature-0 claim.
- [x] **M5b — HTTP server** (`codpiece serve`, crate `codpiece-server`).
      `/v1/chat/completions`, `/v1/completions`, `/health`, `/v1/models`,
      `/slots`, SSE streaming, CORS. Hand-written HTTP/1.1 on `std::net`; the
      chat template renders through minijinja with the pycompat shim, because
      Hugging Face templates call Python string methods that are not Jinja.
      7 tests against the production template pin its behaviour, including that
      `enable_thinking` opens a `<think>` block in the generation prompt.
      **Gate: a greedy request over HTTP is byte-identical to `codpiece gen`.**
      Chat responses split `reasoning_content` from `content` in both shapes —
      non-streaming and SSE deltas (marker-across-chunks handled, 5 unit tests)
      — which is also what makes thinking-mode conversations warm across turns.
- [x] **M5c — speculative decoding at any temperature.** It turned out not to
      need draft probabilities at all. The draft head proposes its *argmax*, so
      the draft distribution is a point mass, and `min(1, p/q)` collapses to
      "accept with probability `p(x0)`, otherwise draw from `p` with `x0`
      removed" — which emits exactly `p`, using only the target distribution the
      sampler already builds. At temperature 0 that same rule reduces to "the
      draft equals the trunk's argmax", so one code path serves both.
      Measured through the server: greedy **39.6 -> 73.5 tok/s**, sampled
      (temp 0.7, top_p 0.9) **35.1 -> 62.5**. A property test asserts the
      emitted distribution matches the target to within 0.005 over 200k draws.
- [x] **M5d — continuous batching, end to end.** Compute path plus scheduler:
      overlapping requests are served through a lazily-created batch session —
      slot admission (with graph-side state zeroing), one 256-token prefill
      chunk per pending request between rounds, fixed-width decode rounds on
      one cached graph, per-slot samplers. Over HTTP on the 27B: 4-way 111
      tok/s aggregate (3.5 s wall for 4x96), **8-way 153.5 tok/s** — the gate
      asked >150 against llama.cpp's ~101. Raw compute path: 173.5 at 8-way.
      Single requests keep the speculative fast path (losslessness 8/8 after
      the dispatcher change). Compute-path details: `SeqMode::{Single, Slot, Batched}` through `build_inner`; the
      recurrent-state slot dimension doubles as the sequence dimension (batch
      mode does not speculate, so snapshots and sequences never coexist — the
      `ne[3]` conflict dissolved). Correctness: `codpiece batchtest` — every slot
      of an N-way batch byte-identical to the single-path reference, on CPU and
      on the 27B under TP. Throughput: **173.5 tok/s aggregate at 8-way**
      (2048/seq), 165.0 at 8x8192 — the gate asked >150 against llama.cpp's
      ~101. Remaining: the HTTP scheduler that feeds it (slot routing,
      prefill/decode interleaving, per-slot samplers) — engine-loop code with
      no graph unknowns left.
      Original notes, kept for the record: The op carries a sequence
      dimension (`q,k,v: [S, H, n_tokens, n_seqs]`, `state: [S_v, S_v, H_v,
      n_seqs]`), so the kernels are ready, but two things have to be settled
      first and both were found by reading rather than guessed:

      1. **`ne[3]` is already taken.** The op wants it for `n_seqs`; codpiece uses
         it for the K recurrent snapshots that make rejected drafts undoable
         (`qwen35.rs:951` offsets a slot by `n * nb[3]`). A ggml tensor has four
         dimensions and speculation needs both. The way out is a state tensor
         *per sequence* and one op call per sequence per recurrent layer — which
         costs kernel launches but not bandwidth, because the weight traffic is
         in the q/k/v/g/beta projections, and those batch across sequences
         normally. Worth measuring against the 129 AllReduce boundaries a step
         already pays.
      2. **Equal-token ubatches.** `n_tokens` is shared across sequences in one
         call, so a batch groups sequences of equal token count; a prefilling
         request and decoding requests cannot share a ubatch.

      The server's worker loop is the seam: it already owns the model and serves
      a queue, so the scheduler replaces `run_job` rather than reaching into the
      HTTP layer.

**Gate:** `~/llm/bench/db2.py` runs unmodified against codpiece ✅ (2026-08-20;
needed `/tokenize`, the `draft_n`/`draft_n_accepted`/`n_draft_calls` timings and
`chat_template_kwargs`); 4-way aggregate > 150 tok/s (llama.cpp ~101) with f16
KV at 196K — pending M5d; no request starvation (p95 TTFT < 2× p50 under 4-way).

Single-stream baseline on that benchmark, temperature 1.0, 400 tokens:

| depth | codpiece prefill / decode | prod prefill / decode |
|---|---|---|
| 0 | 278.7 / **44.7** | **350.1** / 32.4 |
| 32,000 | 820.1 / 48.7 | **1353.2** / **52.7** |

Two gaps it measured: prefill is 25-65% slower (the fused prefill runs the draft
head over every chunk), and acceptance under sampling is 0.35-0.45 against
prod's 0.78 (prod declines to draft below `p-min 0.75`; codpiece always drafts K).

## M6 — Session cache tier (the 70× feature)  ◔ started 2026-08-20

Host-RAM session store: KV + conv/GDN states + token list; restore path;
prefix reuse for shared system prompts.

- [x] **In-session prefix reuse**: a prompt that extends the tokens already in
      the session's caches prefills only the suffix. Correctness-gated on CPU
      (warm continuation byte-identical to a cold run); on the 27B the second
      turn of a 32K conversation went from 24.4 s to **0.9 s** — parity with
      prod's recorded 1.3 s revisit.
- [x] **Multi-conversation switching, as a VRAM session pool**: whole sessions
      resident on-device, longest-prefix slot match, LRU eviction. A/B/A
      alternation between two ~31K conversations on the 27B: return to A in
      **0.9 s** instead of 24.7. Two slots at 32K context, one above ~70K
      (`CODPIECE_SESSIONS` overrides). Chosen over the planned host-RAM tier
      because the split 4-D GDN state cannot be copied off-device under TP
      (meta backend limit #4) — and a pointer switch beats a PCIe copy anyway.
- [x] **Host-RAM snapshot tier** for single-device deployments (where the
      prefix copy works): built, CPU-gated byte-identical, engaged
      automatically when the pool misses and the device is not TP.

**Gate:** session revisit @ 27K ≤ 1.5 s; RAM budget respected under
eviction pressure; zero cross-session token leakage (state keyed by full
token-prefix hash).

## M7 — Production parity extras

Vision (mmproj qwen3vl_merger tower), tool-call grammar constraints,
`-np 2`-equivalent shared-pool policy, then the switch.sh profile so the box
can A/B prod↔codpiece in one command.

**Gate:** Shodan's real workload (screenshots + tool calls) runs a full day
on codpiece with zero correctness incidents, then a measured week.

**Progress (2026-08-20):** the encoder is ported and gated. `codpiece-vision`
builds the qwen3vl_merger ViT (27 layers, GELU, vision M-RoPE) + 2x2 merger
op-for-op from b10423's `models/qwen3vl.cpp`, single-device. Parity vs
`llama-mtmd-debug` on CPU (same build, `-t 20`, `-fa off`): the ported front
end — dual-conv patch embed, block reorder, position embedding — is
bit-exact at the native 48x48 grid and within 2 ULP through the bilinear
resize at 512; full-depth drift reaches only ~5e-4 by layer 26 and ~1e-3 at
the output on both gray and rainbow patterns, attributable to op-fusion
scheduling (the oracle's eval callback splits the graph per node; codpiece
computes fused) — well under BF16 weight precision. `codpiece vision
<mmproj> --pattern gray|rainbow` prints dumps diffable against the oracle.

**Gate defaults retuned for value (2026-08-20, late).** Acceptance ratio is
a conversion statistic, not accuracy — the output distribution is identical
at any setting — and the two draft pools price differently: carried chain
links are free to verify, re-drafts cost ~6 ms each. Defaults moved to
CHAIN_PMIN=0 / REDRAFT_PMIN=0.75 (measured on the shipped build: greedy 68.5,
short sampled 45.2, 32K sampled 47.4 tok/s, acceptance 0.64-0.73); 0.9/0.9
remains available by env for deployments that want the ratio to read >=0.90
(costs ~15% decode). Prod recreated on the new defaults.

**Vision SHIPPED end to end (2026-08-20, same day).** Preprocessing ported
from mtmd's dyn-size pipeline (smart-resize, PAD_CEIL bilinear, min-tokens
1024 like prod); the trunk gained an embd-input graph (`step_embd`) with the
Qwen-VL 2D M-RoPE rule and a block-visible mask; `Session::rope_off` carries
the RoPE-vs-rows gap (images advance RoPE by max(nx,ny) only); image spans
enter prompt history as content-hash pseudo-ids so prefix reuse is
byte-correct; `/v1/chat/completions` takes `image_url` data: URLs; the ViT
runs flash attention on CUDA (head size 72 has a kernel) because the non-FA
KQ is >1 GiB at a 1024-token image and did not fit beside the 64K-context
trunk. Gate (`scripts/vision-payload.sh`): the moon-landing front page
through llama.cpp b10423 + mmproj and codpiece — both answer
"MEN WALK ON MOON" from identically sized prompts; byline follow-up reads
"John Noble Wilford"; text parity gates unchanged. Deployed: prod serves
with `CODPIECE_MMPROJ`; llama.cpp is no longer needed for vision.

## Stretch — DFlash2 under multi-GPU

Block-diffusion drafter (arch `dflash`) through the pluggable drafter
interface. Prize: mean accepted length 4.80 vs MTP's 4.28 (+10–15% decode).

**Gate:** lossless (parity) + ≥ +8% decode over M4 on prose.

## M8 — Coding-first: context, tools, evaluation surface ◕ 2026-08-21

Owner's brief: "perfect speed and accuracy, while maintaining a good
context size for coding" — which reorders the standing priority to
**ACCURACY > CONTEXT ≈ SPEED** for the shipping configuration, since a
coding session is one long conversation, not thirty short ones.

- [x] **Tool calling, end to end.** Qwen3.8 emits an XML-ish framing
      (`<tool_call><function=name><parameter=key>`), not the JSON blob
      earlier Qwen generations used. Parsed into OpenAI `tool_calls`
      with `finish_reason: "tool_calls"`, streaming included. Writing the
      round-trip test found two blockers: the `tojson` filter was never
      registered (so *any* assistant turn carrying tool_calls failed to
      render — every agentic second turn would have 400'd), and the
      template refuses arguments passed as a JSON string, which is
      exactly the wire format clients send back.
- [x] **`echo` + `logprobs`.** Closes the harness gap: loglikelihood
      suites (MMLU, HellaSwag, ARC) can now run. Prompts may be sent as
      token ids so a harness keeps its own tokenization authoritative.
- [x] **Model-recommended sampling for thinking requests.** The GGUF
      carries `general.sampling.*` and it was ignored; greedy + thinking
      is a documented Qwen failure mode.
- [x] **VRAM reporting at startup**, which is what made the context
      sweep legible rather than guesswork.
- [x] **Long-context retrieval verified** — 9/9 needles to 97K tokens
      (see BENCHMARKS.md, including the harness that lied first).
- [ ] **Ship the larger context.** Measured, full stack resident
      (drafter + vision), 1 session:

      | ctx | card0 free | card1 free | verdict |
      |---|---|---|---|
      | 98304 | 2.00 GiB | 1.15 GiB | fits |
      | 114688 | 1.50 GiB | 0.65 GiB | borderline (peak 24.09/24.58 GiB) |
      | 131072 | — | — | 500s on first request |
      | 131072 (no vision) | 1.00 GiB | 1.01 GiB | fits; vision costs ~0.87 GiB |

**Gate:** 98304 shipped with drafter + vision + batch path, retrieval
verified at ~90K, vision working, greedy code decode within 10% of the
40960 figure.

## M3 design brief (from b10423 source reading, 2026-08-19)

llama.cpp's `-sm tensor` (prod's mode) = `ggml_backend_meta_device(devs, n,
get_split_state, ud)`: a Meta device wrapping both CUDA devices. All TP
execution + NCCL all-reduce (PHB-safe, NCCL_P2P_DISABLE honored) lives in
ggml's meta backend. The model supplies ONE callback classifying each tensor
by name into a `ggml_backend_meta_split_state` (row/col/segment splits; fused
tensors like attn_qkv described as segments). llama's callback
(`llama_meta_device_get_split_state`, llama-model.cpp:353) already covers the
whole qwen35 tensor family incl. ssm_*. codpiece M3 = port that classification
+ create the meta device + point Weights::load at it. Session caches/states
also classified (their `cache_[kv]_l\d` patterns → codpiece's own names).
DFlash2's historical crash ("buffer Meta() cannot run the operation") was
draft-model code colliding with this meta backend — codpiece's drafter design
must be meta-aware from day one.
