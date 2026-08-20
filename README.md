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

**ACCURACY > CONTEXT > SPEED.** Every milestone gates on temp-0 token-identical
output vs. llama.cpp before any speed claim counts. f16 KV is the default.
The GGUF-embedded chat template is authoritative.

## The physics (read ENGINE.md first)

Single-stream decode on this hardware is memory-bandwidth-bound at ~65 tok/s
per stream (Q8 weights, tensor-split). Tuned llama.cpp already sits at 80–85%
of that wall. tandem does not promise to break physics; it targets the wins
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
| 27B decode, tensor parallel | 39.8 tok/s | 41.4 |
| 27B decode, + MTP speculation | 59.2 tok/s | **65.0** (prod MTP config, measured head to head) |
| 27B prefill, 7.5K prompt | 1,494 tok/s | — |
| output vs llama.cpp (short, 8K, 27B) | **identical** | reference |
| tokenizer, wikitext-2 | **297,193/297,193 identical** | reference |
| perplexity | 20.4453 | 20.4429 |

tandem is at 96 % of llama.cpp without speculation and ~91 % with it. Two
hypotheses for closing the speculative gap were implemented and measured this
session, and **both were refuted** (see `notes/results-2026-08-19.md`):
confidence-gated drafting moves acceptance but not throughput, and free
CPU-side drafts cost more than they return on the 27B. What measurement
*does* point at is per-draft graph construction: a draft from a one-layer
head costs ~8 ms against ~1.4 ms of actual bandwidth.

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
