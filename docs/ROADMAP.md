# tandem roadmap

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

## M0 — Ground truth + skeleton  ✅ 2026-08-19

- [x] SSH + recon of llm-host (notes/recon-2026-08-19.md)
- [x] Repo, workspace, safety protocol
- [x] `tandem-gguf` parser + `tandem inspect`, validated against all four
      production GGUFs (27B Q8_K_XL, Q6_K_XL-v3, DFlash2 Q8, mmproj)
- [x] qwen35 architecture fully mapped from GGUF metadata + llama.cpp source
- [ ] Vendored ggml pinned @ b10423; `tandem-ggml-sys` links and runs a CPU
      graph locally and on llm-host (container build, sm_86 compile proven)

**Gate:** inspector tensor census matches llama.cpp's loader expectations
(866 tensors, layer schedule 48 GDN / 16 attn / 1 MTP). ✅ + FFI smoke test.

## M1 — CPU correctness on a small dense sibling

Small Qwen3-family dense GGUF (0.6B/1.7B class) end to end on ggml CPU:
tokenizer → graph → sampler.

**Gate:** (a) tokenizer token-identical to `llama-tokenize` on ≥1 MB of mixed
corpus incl. CJK + code; (b) temp-0, 64-token continuation token-identical to
`llama-cli` b10423; max |logit diff| < 1e-3 on first 8 steps.

## M2 — Single-GPU CUDA + decode loop

Same small model on one 3090 **inside the locked bench window**. CUDA-graph
decode loop. First speed data point vs llama.cpp same-model-same-GPU.

**Gate:** token-parity as M1 (CUDA numerics may shift logits; parity = same
tokens temp-0 w/ tie tolerance) + decode ≥ llama.cpp single-GPU on that model.

## M3 — Qwen3.8-27B, 2-GPU tensor split, single stream

The real model: GDN + attention + IMROPE graph, f16 KV, 2-way row split via
backend-sched, recurrent-state checkpointing, greedy decode. No MTP yet.

**Gate:** temp-0 parity vs prod build on 3 prompts (short, 8K, 27K) AND
decode within 10% of llama.cpp-no-MTP baseline (~39 tok/s np1) at d0.

## M4 — MTP speculative decoding

blk.64 draft graph, hidden-state handoff, K-snapshot rollback on GDN layers,
acceptance-adaptive n-draft.

**Gate:** acceptance ≥ 0.90 @ n-max 3 equivalent AND ≥ 57 tok/s prose @ d0
(beat prod's best rep). Output remains temp-0 parity (spec decode is lossless).

## M5 — Serving layer + continuous batching

OpenAI-compatible `/v1/chat/completions` + `/completions` + `/slots`-style
introspection, streaming, minijinja template from GGUF, reasoning_effort +
thinking budget, official sampling presets, multi-stream continuous batching.

**Gate:** `~/llm/bench/db2.py` runs unmodified against tandem; 4-way
aggregate > 150 tok/s (llama.cpp ~101) with f16 KV at 196K; no request
starvation (p95 TTFT < 2× p50 under 4-way).

## M6 — Session cache tier (the 70× feature)

Host-RAM session store: KV + conv/GDN states + token list; restore path;
prefix reuse for shared system prompts.

**Gate:** session revisit @ 27K ≤ 1.5 s; RAM budget respected under
eviction pressure; zero cross-session token leakage (state keyed by full
token-prefix hash).

## M7 — Production parity extras

Vision (mmproj qwen3vl_merger tower), tool-call grammar constraints,
`-np 2`-equivalent shared-pool policy, then the switch.sh profile so the box
can A/B prod↔tandem in one command.

**Gate:** Shodan's real workload (screenshots + tool calls) runs a full day
on tandem with zero correctness incidents, then a measured week.

## Stretch — DFlash2 under multi-GPU

Block-diffusion drafter (arch `dflash`) through the pluggable drafter
interface. Prize: mean accepted length 4.80 vs MTP's 4.28 (+10–15% decode).

**Gate:** lossless (parity) + ≥ +8% decode over M4 on prose.
