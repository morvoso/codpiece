# codpiece architecture

*Decision record. Started 2026-08-19. Everything here is falsifiable by the
gates in ROADMAP.md.*

## 1. The one decision that shapes everything

**codpiece rebuilds the engine layer and inherits the kernel layer.**

ENGINE.md §7's scoping table is the argument: GDN/DeltaNet kernels are
research-grade ("where forks go to die"), while every measured win on this box
lives above the kernels — scheduling (vLLM +245% concurrent), speculative
orchestration (MTP 0.60→0.92 acceptance from *flags*, not kernels), cache
policy (70× session revisit), template/sampling plumbing (2.4× felt latency).

So: vendor ggml (MIT) at the exact tag production runs (**b10423**), link it as
codpiece's compute substrate, and own everything above it. Kernel-level work
happens later, surgically, where profiling proves a win (§6).

What we verified ggml@b10423 already provides (read from the llama.cpp source
on llm-host, snapshots in `notes/reference/`):

- `GGML_OP_GATED_DELTA_NET` — fused GDN with CUDA impl
  (`ggml-cuda/gated_delta_net.cu`), autoregressive *and* chunked paths, and
  native **K-snapshot state rollback** (slot s = state s tokens back) —
  exactly the primitive speculative decoding needs on recurrent layers.
- `ggml_ssm_conv` (the 4-tap causal conv), `ggml_rope_multi` with
  `LLAMA_ROPE_TYPE_IMROPE` (interleaved M-RoPE, sections [11,11,10,0]),
  flash attention, all dequant paths for the prod file (Q8_0 + BF16 + F32 —
  the "Q8_K_XL" name is an Unsloth mix label; no K-quants are actually in it).
- `ggml_backend_sched` — multi-GPU graph placement, the same machinery behind
  llama.cpp `-sm tensor -ts 50,50`.
- CUDA graph capture for the decode loop.

## 2. Language: Rust control plane, C/CUDA compute plane

The layer codpiece rebuilds is a concurrency-heavy control plane: continuous
batching, cache eviction, rollback bookkeeping, streaming HTTP. These die by
data race and use-after-free, which is Rust's home turf. The compute plane
stays C/CUDA (vendored, pinned).

- `ggml.h` is a **C** API — bindgen binds it cleanly, no C++ interop tax.
- Graph construction is thousands of cheap C calls per graph; llama.cpp itself
  rebuilds graphs per ubatch, so FFI call overhead is a proven non-issue.
- HTTP/streaming: axum/tokio. Template: minijinja (the 9,993-byte embedded
  Qwen template is Jinja and is load-bearing — `reasoning_effort` lives there).
- Tokenizer: own implementation of GGUF-embedded BPE (`tokenizer.ggml.model =
  "gpt2"`, `pre = "qwen35"`, 248,320 tokens, 247,587 merges), validated
  token-for-token against `llama-tokenize` on real corpora.

## 3. Crate map

```
codpiece-gguf      GGUF v2/v3 reader (no deps)                 [done]
codpiece-ggml-sys  bindgen FFI over vendored ggml, cmake build [m0]
codpiece-tok       BPE tokenizer + template engine             [m1]
codpiece-model     qwen35 graph builder (trunk + MTP graph)    [m1-m3]
codpiece-runtime   scheduler, KV/recurrent cache, spec decode  [m3-m6]
codpiece-server    OpenAI-compatible API, streaming, sessions  [m5]
codpiece-cli       inspect / run / bench                       [rolling]
```

## 4. The model, as read from the production GGUF

`general.architecture = "qwen35"`, 65 blocks = 64 trunk + 1 MTP (`blk.64`,
`nextn.*` tensors). `full_attention_interval = 4`: layers 3,7,…,63 are full
attention (16 with KV cache), the other 48 are GDN recurrent.

Trunk GDN layer (from `notes/reference/qwen35.cpp`, codpiece must reproduce
exactly):

```
x → rms(attn_norm)
  → wqkv [5120→10240]  (q:128×16, k:128×16, v:128×48 after conv split)
  → wqkv_gate z [5120→6144]
  → beta = sigmoid(ssm_beta·x)          [48 heads]
  → g    = ssm_a ⊙ softplus(ssm_alpha·x + dt_bias)   (per-head log-decay)
  → causal conv1d (kernel 4, 10240 ch) over [q|k|v] with carried conv state, SiLU
  → split q,k,v; L2-normalize q,k per head
  → fused GDN(q,k,v,g,beta,state)  →  out, new_state (+K rollback snapshots)
  → rms-norm(out) ⊙ silu(z) → ssm_out [6144→5120]
residual; rms(post_attn_norm); SwiGLU FFN (17408); residual
```

Full-attention layer: `attn_q.weight` packs **Q and a per-head output gate
interleaved** (2×256 per head); q/k RMS-normed per head; IMROPE on 64 of 256
dims; GQA 24/4 heads at 256 wide; `sigmoid(gate) ⊙ attn_out` before `wo`.
KV per token: 4 heads × 256 × 2 (K,V) × 2 B = 64 KiB f16 across the 16 layers
(matches the measured 12 GiB at 196K).

MTP graph (`blk.64`): `concat(rms_e(emb(tok)), rms_h(h_trunk)) → eh_proj
[10240→5120] → one full-attention block (own KV cache) → shared output head`.
llama.cpp drives it via `LLM_GRAPH_TYPE_DECODER_MTP` with hidden-state handoff
(`t_h_nextn`) — codpiece's runtime owns this handoff and the verify batching.

Recurrent session state is why session caching must be engine-native:
48 layers × (conv state 3×10240 + GDN state 128×128×48) ≈ **~82 KiB/token-slot
equivalent** in the host tier (matches ENGINE.md's measured session overhead).

## 5. Runtime design (the part that beats llama.cpp)

**Continuous batching, one unified GPU pass per step.** llama.cpp's server
runs slot-based `-np N` with a shared context; vLLM proves this box rewards
real continuous batching (+245% at 4-way). codpiece composes each step's ubatch
from whatever sequences are runnable: prefill chunks and decode tokens ride
the same graph where shapes allow, MTP verify tokens batch with them.

**Cache manager owns three tiers:**
1. GPU KV (f16, 16 layers) + GDN/conv states (f32) with K rollback slots.
2. Host-RAM session tier — full sequence state (KV + recurrent states +
   token list), LRU by session, budget-capped (prod proves 40 GiB works).
   Restores must beat llama.cpp's 1.3 s @ 27K.
3. Disk is out of scope until measurement demands it.

Recurrent states make rollback checkpointing mandatory (can't recompute
backwards): codpiece checkpoints GDN+conv states every C tokens during prefill
(llama.cpp uses 32 checkpoints / 8192-token spacing as its default shape) and
keeps K=n_spec+1 rolling snapshots during speculative decode — both mechanisms
already supported by the fused op + our copy scheduling.

**Speculative pipeline.** MTP first (target: ≥0.92 acceptance @ n-max 3,
p-min 0.75 — prod's measured numbers are the bar). The drafter interface is
pluggable so DFlash2 (arch `dflash`, 5 blocks, 5120 embd — drafts 8-token
blocks) can slot in under multi-GPU, which llama.cpp PR-27342 structurally
cannot do (single-GPU-era allocator assumptions; see ENGINE.md §6 autopsy).

**Decode loop as CUDA graph.** Kernel-launch gaps are part of vLLM's edge;
ggml-cuda supports graph capture; codpiece's fixed-shape decode step is designed
to stay capture-legal (no shape churn on the hot path).

**Multi-GPU.** Two modes, both implemented:

- *Layer split* (`Device::CudaSplit`): contiguous layer ranges per device,
  ggml_backend_sched moves activations. Simple and correct; makes the 27B
  runnable. Slow, because the GPUs alternate rather than share each token.
- *Tensor parallel* (`Device::CudaTensorParallel`): ggml's meta device wraps
  both GPUs and presents one backend, so codpiece keeps its fast path while
  every matrix is sliced. This is prod's `-sm tensor`. codpiece supplies the
  split classification (`split.rs`, unit-tested offline against the real 27B
  shapes) and ggml inserts the all-reduce.

The user owns **no NVLink bridge**, and GeForce P2P is driver-disabled, so
all inter-GPU traffic is PCIe host-bounce — permanently. Design accordingly:
prefer one reduction per split point over per-layer activation handoff, and
never assume peer memory.

Two constraints the meta backend imposes on graph construction, learned on
hardware: a split tensor's per-device extents must sum exactly to its own
extent (it asserts), and it cannot map a split axis through a
stride-reordered view — so cache reads are built with monotonically
increasing strides and then permuted, never viewed into permuted form.

## 5b. Designing for *this* box, not for a generic one

Two resource facts drive everything, and they point opposite to the defaults a
portable engine picks:

- **PCIe is scarce, VRAM is not.** No NVLink bridge exists on this machine and
  GeForce P2P is driver-disabled, so every inter-GPU byte takes a host bounce.
  Meanwhile each card has ~9 GiB free after weights and KV.
- **The CPU is idle.** 24 threads and 62 GiB of RAM do nothing during a decode
  step, which is entirely GPU-bandwidth-bound.

Consequences codpiece acts on:

**Replicate the LM head rather than split it** (`split.rs`). The memory-optimal
choice column-splits `output.weight`, but then the vocabulary is spread across
devices, argmax cannot run in the graph, and every scored position ships 3.9 MB
of logits over the scarce link. Replicating costs ~0.7 GiB per card of the
plentiful resource and reduces the readback to 4 bytes per position. This is
the single change that makes wide speculative verification affordable.

**Draft on the CPU** (`oracle.rs`). Decode is bandwidth-bound, so a verify pass
costs one weight-read whether it checks 1 token or 12: the quantity to optimize
is *accepted tokens per weight-read*, and draft cost is the enemy. The MTP head
is a real transformer block (~8 ms per draft on the 27B), so its deeper drafts
are a poor trade at falling acceptance. A bounded-order predictor over the live
token stream runs on the idle CPU in microseconds and is strong exactly where
transformers spend tokens — quotation, repeated identifiers, structural
scaffolding. It declines to predict on novel text, and a self-tuning gate keeps
it from paying for width it is not earning.

**What does NOT work here, and why it is worth writing down:** tree
speculation. Verifying a branching tree of candidates in one pass is standard
elsewhere, but 48 of this model'"'"'s 64 layers are Gated DeltaNet recurrences
whose state advances strictly along one sequence. A tree has no single
sequence, so the recurrent layers cannot evaluate its branches in one pass.
The hybrid architecture forecloses the technique regardless of hardware.

## 6. Later, surgical kernel work (only after profiles demand it)

- Fused GDN in-projection (wqkv + conv + split + L2-norm are separate ops).
- All-reduce tuned for PHB (staged pinned-buffer bounce, stream-ordered).
- Speculative verify: batched MTP verify with per-position early-exit.

## 7. Accuracy invariants (non-negotiable, from the box's standing priority)

1. Temp-0 token-parity vs llama.cpp b10423 at every milestone gate.
2. f16 KV default; q8_0 KV only as an explicit context-stretch mode.
3. GGUF-embedded chat template, byte-identical
   (sha256 12827f24… — verified equal to prod's live template).
4. Official sampling presets wired as defaults (thinking: t=1.0/top_p=0.95/
   top_k=20; non-thinking: t=0.7/top_p=0.8/presence=1.5).
5. `reasoning_effort` + thinking-budget plumbing (prod runs budget 4096 with a
   budget-exhaustion message — replicate).

## 8. What codpiece does NOT do

- No training, no quantization tooling (Unsloth's files are the input).
- No CPU inference path beyond what correctness testing needs.
- No architectures beyond qwen35 (+dflash drafter, +qwen3vl_merger mmproj)
  until the box runs something else.
