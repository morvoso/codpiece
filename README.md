# tandem

A from-scratch LLM inference engine for exactly one machine shape: **2× RTX 3090
(24 GiB, ~936 GB/s each) connected over PCIe with no P2P DMA**, serving
**Qwen3.8-27B** (hybrid Gated-DeltaNet/attention with an embedded MTP head).

tandem replaces the *engine layer* of llama.cpp and vLLM — scheduling,
batching, KV + recurrent-state management, speculative decoding orchestration,
session caching, and the server API — while inheriting the battle-tested
compute kernels underneath (vendored ggml CUDA, MIT-licensed, pinned to the
exact build production runs today). Accuracy is a bit-parity property by
construction, not a hope.

## Design priorities (standing, from the box's operating doc)

**ACCURACY > SPEED > CONTEXT.** Every milestone gates on temp-0 token-identical
output vs. llama.cpp before any speed claim counts. f16 KV is the default.
The GGUF-embedded chat template is authoritative.

## The physics (read ENGINE.md first)

Single-stream decode on this hardware is memory-bandwidth-bound at ~64 tok/s
per stream (Q8 weights, tensor-split): 14.65 GiB per GPU per token against
936 GB/s is 15.6 ms. Tuned llama.cpp already sits at 80-85% of that wall.

Speculation is the one thing that gets *past* it, and this is worth being
precise about rather than treating as a paradox: a round reads the weights
once and can commit several tokens, so the floor applies per *round*, not per
token. That is why the numbers below exceed 64 tok/s, and why the entire
optimization problem reduces to accepted tokens per round. tandem does not promise to break physics; it targets the wins
that measurement says exist:

| lever | evidence |
|---|---|
| Continuous batching / concurrency | vLLM: +245% at 4-way on this box |
| Single-stream decode overhead | vLLM: +33–49% vs llama.cpp, same GPUs |
| MTP verify batching ≥0.92 acceptance | llama.cpp measured, must not regress |
| Host-RAM session cache incl. GDN states | 70× on session revisit (llama.cpp `--cache-ram`) |
| f16-KV fidelity at 196K+ context | vLLM cannot (forced fp8); tandem must |
| DFlash2 block-diffusion drafting, multi-GPU | nobody has it; +10–15% measured prize |

The end state is one engine that holds all rows at once — which today no
engine does.

## Status — 2026-08-19

Running the production Qwen3.8-27B on 2×3090, tensor-parallel, with MTP
speculative decoding. Every speed number below has an accuracy gate behind
it: output identical to llama.cpp b10423, or it does not count.

| | tandem | llama.cpp b10423 |
|---|---|---|
| 27B decode, tensor parallel | 39.0 tok/s | 41.4 |
| 27B decode, + MTP speculation | **69.9 tok/s** | 64.5 (prod MTP config) |
| same, on a code-writing prompt | **67.6 tok/s** | 53.3 |
| 27B prefill, 7.5K prompt | 1,494 tok/s | — |
| output vs llama.cpp (short, 8K, 27B) | **identical** | reference |
| tokenizer, wikitext-2 | **297,193/297,193 identical** | reference |
| perplexity | 20.4453 | 20.4429 |

The speculative row is a head-to-head: both engines measured in the same
locked bench window, same GPUs, same model, same prompt, 96 tokens, three
repetitions each. llama.cpp ran 64.52 / 64.78 / 64.28 tok/s; tandem's fused
speculative round ran 64.51 / 64.47 / 64.49 at depth 2 and **69.86 / 69.81 /
69.89 at depth 3**. Acceptance is prompt-dependent, so depth 3 is not always
the best choice — on a longer 160-token generation depth 2 led at 64.3.

### What "lossless" does and does not mean here

Every token a speculative round commits is one the trunk itself predicted: a
draft is kept only when it equals the trunk's own argmax at that position, and
the first rejection is replaced by that argmax. Speculation cannot invent a
token.

It can still change the output, and the honest statement is that it sometimes
does. Verifying T tokens in one batch is not bit-identical to decoding them one
at a time — different GEMM shapes reduce in a different order — so where two
logits are nearly tied, the argmax can flip. Measured on three prompts at
depths 1-3, output was byte-identical to plain greedy on two of them and
diverged on the third (a code-writing prompt), at every depth.

That is a property of batched speculative decoding on this model, not of
tandem. **llama.cpp's own MTP speculation diverges from its own greedy output
on the same prompt, at the same token** (`scripts/specparity-payload.sh`
measures exactly this). Both engines' *non*-speculative greedy outputs agree
with each other word for word.

So: plain decode is gated on byte-identical output versus llama.cpp, and that
gate holds. Speculative decode is gated on committing only trunk-predicted
tokens, and on matching the reference implementation's behaviour — which it
does.

Draft depth is `--depth N` (default 3, the best single setting across the prompts
measured). `--depth auto` chooses per round from observed acceptance and measured
round cost, and matches fixed depth 3 within +-0.4% across four prompts while
picking the depth itself. It stays opt-in because it matches rather than beats a
correctly chosen fixed depth — but depth is worth up to 14% between adjacent
settings, so it is the right choice when the workload is unknown.

What closed the gap was not drafting at all. Three lifetime bugs were keeping
graphs from being reused across computes — including one where
`ggml_new_graph_custom` leaves `cgraph->uid` at 0, which the tensor-parallel
backend reads as "assume this graph changed", so **every compute rebuilt its
per-device mapping even for graphs tandem had carefully cached**. Fixing that
took plain decode from 20 backend rebuilds per 20 tokens to 2, and took the
speculative round from 53.6 to 64.3 tok/s. See
`docs/OPTIMIZATION-IDEAS.md` §4 and `notes/results-2026-08-19.md`.

Milestones M0–M4 are done and gated; see `docs/ROADMAP.md`. Next: closing the
speculative gap, then the serving layer (M5) and the host-RAM session cache
(M6).

## Layout

- `crates/tandem-gguf` — zero-dependency GGUF v2/v3 reader (done, tested
  against the production 31 GB file).
- `crates/tandem-cli` — `tandem inspect`, header-only model analysis (done).
- `crates/tandem-ggml-sys` — FFI bindings to vendored ggml (in progress).
- `docs/ARCHITECTURE.md` — full design and rationale.
- `docs/ROADMAP.md` — milestones M0–M7, each with a hard accuracy gate.
- `docs/SAFETY.md` — hardware- and production-safety protocol for the
  llm-host box. **Read before running anything on the server.**
- `notes/` — recon logs and reference snapshots (llama.cpp qwen35 sources,
  prod config, chat template).
- `ENGINE.md` — the measured knowledge base this project is built on.

## License

MIT. Vendored ggml/llama.cpp sources are MIT (ggml-org).
