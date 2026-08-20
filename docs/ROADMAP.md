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
- [x] Vendored ggml pinned @ b10423; `tandem-ggml-sys` links and runs CPU
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
      via ORACLE_FA; TANDEM_SUBCMD selects stateless/session path).
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
| llama.cpp b10423, same GPU/model/path | 317 | tandem = **93%** |

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
- [ ] M3b — 27B loaded and generating across both cards
- [ ] M3c — parity vs prod llama.cpp (temp-0, short / 8K / 27K prompts)
- [ ] M3d — placement fixes: session KV/state tensors on their layer's
      device (today they all sit on device 0, doubling bus traffic for the
      second half of the stack), cached decode graph under the scheduler
- [ ] M3e — tensor parallel (`split.rs` classification + meta device), which
      is what prod's `-sm tensor` uses and what makes single-stream decode
      competitive. All-reduce must assume PCIe host-bounce forever: the user
      confirmed there is no NVLink bridge on this machine.

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

## M3 design brief (from b10423 source reading, 2026-08-19)

llama.cpp's `-sm tensor` (prod's mode) = `ggml_backend_meta_device(devs, n,
get_split_state, ud)`: a Meta device wrapping both CUDA devices. All TP
execution + NCCL all-reduce (PHB-safe, NCCL_P2P_DISABLE honored) lives in
ggml's meta backend. The model supplies ONE callback classifying each tensor
by name into a `ggml_backend_meta_split_state` (row/col/segment splits; fused
tensors like attn_qkv described as segments). llama's callback
(`llama_meta_device_get_split_state`, llama-model.cpp:353) already covers the
whole qwen35 tensor family incl. ssm_*. tandem M3 = port that classification
+ create the meta device + point Weights::load at it. Session caches/states
also classified (their `cache_[kv]_l\d` patterns → tandem's own names).
DFlash2's historical crash ("buffer Meta() cannot run the operation") was
draft-model code colliding with this meta backend — tandem's drafter design
must be meta-aware from day one.
