# Building a Custom Inference Engine — What We Know, What It Takes

*Compiled 2026-08-19 from the llm-host tuning campaign. Every number in here was
measured on this hardware with `~/llm/bench/db2.py` (llm-host) unless cited
otherwise. This is the knowledge base for the "code our own engine" question.*

---

## 1. The Hardware Truth

| | |
|---|---|
| GPUs | 2× RTX 3090, 24 GiB each, ~936 GB/s DRAM bandwidth each |
| Interconnect | **PCIe via host bridge (PHB) — NVLink links INACTIVE** (`nvidia-smi topo -m` = PHB; `nvlink -s` = "all links are inActive"). GeForce driver also disables PCIe P2P DMA (`NCCL_P2P_DISABLE=1` required) |
| CPU | i9-12900KF, 24 threads |
| RAM | 62 GiB |
| GPU0 quirk | fan sits against GPU1's backplate; capped 260 W (default 350 W) via `gpu-powerlimit.service`. Cap costs ~5% decode / ~10% prefill under tensor-parallel because both cards run in lockstep |

### The bandwidth wall (why a custom engine can't win big)

Decode is memory-bandwidth-bound: every generated token streams the full weight
set. Qwen3.8-27B at Q8 ≈ 28.5 GiB of weights; tensor-split two ways, each 3090
reads ~14.3 GiB per token:

```
936 GB/s ÷ 14.3 GB/token ≈ 65 tok/s   ← hard ceiling per stream, before overhead
measured today: 50–57 tok/s with MTP  ← already ~80-85% of ceiling
```

**A perfect engine buys 20–40% at most** (all-reduce overhead, kernel-launch
gaps, better speculative verify batching). It cannot buy 3×. The DRAM is the
DRAM. This single paragraph is the reason "build our own engine for speed" is
the wrong trade — and specialization projects like ninfer confirm it: ~22%
over tuned llama.cpp master, on hardware it was purpose-built for.

---

## 2. The Model (what any engine must implement)

**Qwen3.8-27B** (released 2026-08-14 — five days old at time of writing):

- **Hybrid architecture: 64 layers — only 16 full-attention (with KV cache), 48
  Gated DeltaNet (GDN) recurrent layers** with fixed-size recurrent state.
  This is the research-grade part; reference implementations are llama.cpp's
  CUDA kernels and Qwen's Triton.
- **Embedded MTP head** (multi-token prediction, `blk.64.nextn.*` tensors) for
  built-in speculative decoding.
- Native context 262,144 (YaRN to 1M). We run 196,608 — set by f16-KV VRAM fit.
- Vision tower via separate `mmproj-BF16.gguf` (931 MB, wants ~1.2–1.5 GiB on
  one device at load time — a real packing constraint, see §5).
- KV arithmetic (16 of 64 layers cache; 4 KV heads): **f16 = 64 KiB/token →
  196K ctx = 12 GiB**; q8_0 = 34 KiB/token → 262K = 8.5 GiB.
- Session state is MORE than KV: ~82 KiB/token in the host prompt cache because
  the **48 DeltaNet recurrent states** ride along.

### Chat template is load-bearing

The template embedded in the GGUF implements `reasoning_effort`
(xhigh/medium/low/none), `enable_thinking`, `preserve_thinking`. The whole
efficiency story of this box rides on it (see §4). Verified 2026-08-19: the
Dynamic-v3 rollout changed **zero bytes** of the template (9,993 B, identical
in Q8_K_XL and Q6_K_XL-v3). Never adopt a webpage template over the embedded
one without diffing.

### Official sampling (Unsloth docs + Qwen model card — verified live)

| mode | temp | top_p | top_k | min_p | presence |
|---|---|---|---|---|---|
| thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

Off-spec sampling is not cosmetic: sub-recommended temps cause repetition
loops on this family. (`ai-agent` ran temp 0.2 for weeks; fixed 2026-08-19.)

---

## 3. Measured Results — the Campaign (2026-08-19)

Baseline prod config: llama.cpp `server-cuda-b10423` (Docker), Q8_K_XL, f16 KV,
196,608 ctx, `-sm tensor -ts 50,50`, `-b 4096 -ub 512`, `-np 2 -kvu`,
`--cache-ram 40960`, MTP `n-max 3, p-min 0.75`, GPU0 @ 260 W.

| experiment | result | verdict |
|---|---|---|
| baseline | decode 49.9 @ d0 / 52.3 @ 27K, prefill 1,387 tok/s, MTP accept 0.78–0.87 | — |
| b10450 image | 54.8 @ 27K | tied (noise) |
| b10499 image | 53.4 @ 27K | tied (noise) |
| PR-27342 source build | 51.5 @ 27K | tied (noise) |
| `-ub 1024` @ 196K ctx | **server crashed mid-prefill** (flash-attn VMM pool OOM) | ub 512 is load-bearing |
| threads 4/8/16/24 | 39.0/38.8/38.4/38.4 | flat; dead axis at `-ngl 999` |
| batch 2048 vs 4096 | 48.1 vs 48.1 mean | dead axis |
| ubatch 512 vs 1024 (small ctx) | 48.1 vs 48.1 mean | dead axis |
| `-np` 1→2→4 (concurrency matched) | 39.4 → 67.6 → 100.9 aggregate; 41.1 → 36.1 → 27.8 per-request | np 2 = right trade for 1 user + background daemons |
| Q6_K_XL **Dynamic v3** | decode 55.6/56.7 (+8–11%), prefill 1,339 (−3%), **7.9 GiB free vs 1.6** | validated fallback, not deployed (accuracy-first) |
| DFlash2 drafter | see §6 | parked — no multi-GPU support |

**Rep-to-rep MTP-acceptance noise is ±10% — bigger than any build delta.**
Single-rep benchmark wins are usually acceptance luck. Always ≥2 reps.

### Prior measured wins already banked in this config (from the compose docs)

- `--spec-draft-p-min 0.75`: MTP acceptance 0.60 → 0.92 (the big spec win)
- `--spec-draft-n-max 3` beats 4 and 5 (A/B/A: 52.0/48.7/49.1 prose;
  62.7/61.0/61.4 code) — more draft depth ⇒ lower acceptance ⇒ net loss
- f16 KV beats q8_0 **on speed** at depth (47.8 vs 41.8 @ 96K) — dequant inside
  the FA kernel costs more than halved bandwidth saves
- `--cache-ram 40960`: session revisit re-prefill 91 s → 1.3 s (~70×). *Decode
  was never the problem on this box; re-prefill was.*
- `reasoning_effort medium` default: same trivial task 53 s (xhigh) → 22 s
  (medium) → 3.9 s (off). **2.4× — an order of magnitude more than any engine,
  quant or cache change measured here.**

### The engine comparison already on record (2026-08-16, same harness)

| | llama.cpp (deployed) | vLLM 0.27.1 |
|---|---|---|
| prose / code decode | 51 / 67 | **76 / 90 (+49% / +33%)** |
| prefill 27K | 1,381 | **1,591 (+15%)** |
| 4-way concurrent | 73 | **251 (+245%)** |
| KV fidelity | **f16** | fp8 (forced — f16 caps ~65–80K ctx) |
| prefix cache | host-RAM ~512K tokens | GPU-only ~236K tokens, **experimental** "align" mode for hybrid archs |

vLLM's "feels slow" mystery was solved and documented: `--enable-prefix-caching`
is OFF by default for hybrid GDN models; without it TTFT was 26.9 s *every*
turn; with it 1.6 s. Still reverted on fidelity + cache-capacity grounds.
Watch item: **vLLM 0.28** collects an MTP+prefix-cache correctness fix
(PR #47861) and align-mode hardening; the OffloadingConnector (CPU host-RAM KV
tier — llama.cpp's cache-ram trick) already exists in 0.27 but is untested
against hybrid align mode, which has an open silent-0%-hit bug (#45238).

---

## 4. Where the Real Latency Lives (the altitude lesson)

Ranked by measured impact on *felt* speed, biggest first:

1. **Thinking budget** — 2.4× swing (`reasoning_effort`). Client-side. Free.
2. **Prompt-cache behavior** — 70× on session switches (`--cache-ram`, prefix
   stability, dynamic content in the *user* turn not the system prompt).
3. **Sampling correctness** — accuracy/repetition, i.e. fewer retries.
4. **Speculative decoding** — MTP ≈ 2.5–3× over greedy baseline.
5. Engine/build/kernel choice — ±10% at best on this hardware. **Last place.**

Any "new engine" conversation that starts anywhere but line 5 of this table is
starting in the wrong place — and a new engine only addresses line 5.

---

## 5. VRAM Packing Map (why everything is tight)

Prod layout at 196,608 ctx, f16 KV, Q8 weights (48 GiB total):

```
weights (tensor-split)   ~29.3 GiB
KV cache f16             ~12.0 GiB
GDN recurrent states     ~part of session state; GPU share modest
vision tower (mmproj)     ~1.2 GiB single-device at load
compute buffers/pools     ~2-3 GiB (flash-attn VMM pool grows with ub × ctx)
─────────────────────────────────
free                      ~1.6 GiB   ← nothing else fits
```

Load order matters: main model → draft model (if any) → **clip/mmproj last** —
so a draft model can evict the vision tower's slot (this is exactly how DFlash2
died its third death, §6). The Q6_K_XL-v3 fallback frees ~6.3 GiB and is the
documented lever when anything must co-reside.

---

## 6. Case Study: the DFlash2 Autopsy (what engine surgery looks like)

DFlash2 = block-diffusion drafter for Qwen3.8-27B (drafts 8-token blocks;
claimed mean accepted length 4.80 vs native MTP's 4.28; lossless; +1.3% cycle
overhead). GGUFs: 1.9B params — Q4_K_M 1.14 GB / Q8_0 2.06 GB (on disk in
`~/llm/models/qwen38/`). Needs llama.cpp **PR #27342 (open)**; the
`draft-dflash` type in released builds is DFlash**1** and expects a different
tensor layout ("expected 81 tensors, got 58" = version skew, not corruption).

Built the PR from source (no CUDA toolkit on host, no sudo — built inside
`nvidia/cuda:13.0.0-devel` with `--runtime nvidia` so `libcuda` stubs resolve;
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`). Then, in order:

1. `--device-draft CUDA1` → abort: *pre-allocated tensor (output.weight) in a
   buffer (Meta()) that cannot run the operation* — draft-on-one-device is
   incompatible with the tensor-split Meta backend.
2. Draft split across both GPUs → mmproj (931 MB) no longer fits CUDA0
   (load-order eviction, §5).
3. No vision → abort inside `ggml_gallocr_alloc_graph` via
   `common_context_can_seq_rm` — the DFlash2 draft graph cannot allocate on
   the multi-GPU scheduler at all.

**Conclusion: the PR is single-GPU-era code.** Every tester in its thread ran
one card. This is the highest-ROI "build our own engine" target on this box:
**make DFlash2 work under `-sm tensor` multi-GPU** — allocator + backend-sched
+ draft-graph placement work, a measured ~10–15% decode prize, an upstream
contribution, and the failing build + bench harness are already on disk.

---

## 7. So You Want to Build an Engine

### The honest scoping table

| component | difficulty | notes |
|---|---|---|
| GGUF loader, tokenizer, sampler, HTTP server | easy | the fun first month |
| Q8/Q6 dequant + GEMM CUDA kernels | hard, well-trodden | cuBLAS gets you started; custom kernels for the last 20% |
| Flash attention @ 192K | hard | FA kernel + VMM pool management (see the ub-1024 crash) |
| **GDN/DeltaNet recurrent kernels** | **research-grade** | 48 of 64 layers; 5-day-old arch; references: llama.cpp CUDA + Qwen Triton. Where forks go to die |
| Multi-GPU tensor split + all-reduce over PCIe (no P2P!) | hard | staged host-bounce all-reduce; NCCL needs `ipc: host` + shm in containers |
| MTP speculative verify | medium | and it must beat 0.92 acceptance to matter |
| KV + recurrent-state cache mgmt, host-RAM tier | medium-hard | the 70× feature; recurrent states make checkpointing mandatory (`--ctx-checkpoints`, default 32/8192-token spacing — rollback on recurrent layers is otherwise impossible) |
| Vision tower (mmproj) | medium | required for Shodan's screenshots |
| chat template + reasoning_effort plumbing | easy but critical | the 2.4× lever lives here |

Realistic timeline for parity with tuned llama.cpp on this exact model/hardware:
**months of full evenings.** Prize for beating it: **≤ 20–40%** (§1), and the
bandwidth wall means most of that is all-reduce/launch overhead, not magic.

### The three sane paths, ranked

1. **Targeted surgery (recommended): fix DFlash2 multi-GPU in llama.cpp.**
   Scoped, measurable (+10–15% decode), upstream-able, all materials on disk
   (`~/llm/llama.cpp-dflash2`, draft GGUFs, `~/llm/bench/`). This IS engine
   work — allocator, scheduler, graph placement — without rebuilding the 95%
   that's already optimal.
2. **Educational from-scratch, single GPU:** small Qwen (4B-class), one 3090,
   greedy decode, correctness-diffed against llama.cpp layer by layer, then
   speed. The ninfer model: specialize ruthlessly, audit honestly. Learning
   gold; production irrelevance.
3. **Full production engine (2×3090, 27B, 192K, vision, spec-decode, caching):**
   only if the *goal is the journey*. The destination is a rounding error.

### Non-negotiables any engine must reproduce (or the box gets *slower*)

- reasoning_effort plumbing (2.4×)
- host-RAM prompt/session cache incl. GDN states (70× session switch)
- MTP ≥ 0.9 acceptance at n-max 3 (≈2.5–3×)
- f16 KV at 196K (accuracy + measured speed at depth)
- vision, tool-call grammar, `-np 2` shared-pool concurrency

---

## 8. Operational Lore (paid for in downtime today)

- **Never benchmark against a busy server** — a concurrent prefill halves
  decode silently. `db2.py` refuses via `/slots`; respect it. Shodan's own
  daemons (attention, waybar-health) contaminated the first baseline today.
- **Two sweeps must never overlap** — each stops/restores prod; the overlap
  produced a bogus "parallel-4 = 19 tok/s" table and OOM'd every cell. flock
  (`/tmp/llama-sweep.lock`) now enforces this.
- llama.cpp exit-status traps: a server that dies at arg-parse (`--ubatch` is
  not a flag; it's `--ubatch-size`/`-ub`) with output routed to `/dev/null`
  looks exactly like a hang. **Tee server stderr somewhere, always**
  (`~/llm/llama-server-docker` shim does).
- `docker run -d --rm` deletes your crash evidence. Drop `--rm` while
  debugging.
- Filter by container ID, not image ancestry — prod and bench share an image;
  an `ancestor=` kill filter took production down once today.
- GGUF verify: HF `x-linked-etag` is the sha256; check before re-downloading
  30 GB (today's "update" was renames + low-tier v3 requants; Q8_K_XL
  unchanged, template byte-identical).
- Builder containers need `--runtime nvidia` or the `cuMem*` driver symbols
  won't link.
- `pgrep/pkill -f` matches your own wrapper shell. Four incidents today.
  Anchor patterns (`^bash /path`) or kill by PID.

---

## 9. Current State (post-campaign, 2026-08-19 evening)

- **Deployed:** llama.cpp b10423, Q8_K_XL (launch-gen; v3 never touched Q8),
  f16 KV @ 196,608, MTP 3/0.75, cache-ram 40960, np 2, GPU0 @ 260 W.
  Unchanged — measured at its ceiling.
- **Changed today:** `llm` + `ai-agent` sampling now mode-aware per the model
  card (was temp 0.2 flat — a real accuracy bug on every Shodan reply).
- **Shelf:** Q6_K_XL-v3 (validated: +8–11% decode, 7.9 GiB free), vLLM 0.27.1
  fast-profile via `./switch.sh vllm`, DFlash2 build + drafts.
- **User-action levers:** seat/replace the NVLink bridge (links inactive —
  biggest untapped win for `-sm tensor`); test GPU0 at 280–300 W with temps
  watched (`sudo nvidia-smi -i GPU-2763fe73… -pl 300`, revert `-pl 260`).
- **Watch:** llama.cpp PR #27342 (DFlash2) for multi-GPU support; vLLM 0.28
  (MTP-cache fix + align hardening + OffloadingConnector re-test); Unsloth
  regenerating Q8-tier with Dynamic v3 (nothing yet — v3 gains live at low
  bits).
