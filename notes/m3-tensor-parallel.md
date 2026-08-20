# M3: tensor-parallel port notes (from b10423 source, 2026-08-19)

Reference extract: `notes/reference/meta-split-state.cpp.txt`
(llama-model.cpp:353-600, the whole split-state callback).

## Mechanism

`-sm tensor` builds ONE meta device wrapping both CUDA devices:

```c
ggml_backend_dev_t meta = ggml_backend_meta_device(
    devs, n_devs, get_split_state_callback, userdata);
```

Everything else (per-device execution, NCCL all-reduce, PHB host-bounce)
lives inside ggml's meta backend. The model supplies exactly one callback:
given a tensor by name, return a `ggml_backend_meta_split_state`
(split axis + per-device segment sizes + repetition counts).

## Split axis per tensor class (qwen35)

| tensor | axis | axis-0 reference |
|---|---|---|
| attn_q/k/v.weight, attn_qkv.weight, attn_gate.weight | AXIS_1 | attn_output.weight ?? ssm_out.weight |
| attn_q/k_norm.weight | ne[1]==1 ? MIRRORED : AXIS_1 | attn_output.weight |
| attn_output.weight, ssm_out.weight | AXIS_0 | (self) |
| ssm_dt.bias, ssm_a | AXIS_0 | ssm_out.weight |
| ssm_alpha/beta.weight, ssm_conv1d.weight | AXIS_1 | ssm_out.weight |
| KV cache (cache_k/v_l*) | AXIS_0 | attn_output.weight |
| conv state (cache_r_l*), GDN state (cache_s_l*) | AXIS_0 | ssm_out.weight |
| ffn_up/gate.weight | AXIS_1 | ffn_down.weight |
| ffn_down.weight | AXIS_0 | ffn_down.weight |
| output.weight | AXIS_1 | (self) |
| everything else (norms, embeddings) | MIRRORED | |

## The qwen35 segmentation landmine

Qwen3.5 broadcasts GDN heads differently from Qwen3-Next:
- Qwen3-Next: `[k0_v0, k0_v1, k1_v2, k1_v3]` (default contiguous split)
- **Qwen3.5: `[k0_v0, k1_v1, k0_v2, k1_v3]`** — V must be segmented at K's
  scale or the head pairing silently scrambles.

With `head_ratio = n_v_heads / n_k_heads` (27B: 48/16 = 3):

| tensor | segments {size, repeats} |
|---|---|
| attn_qkv.weight, ssm_conv1d.weight | `{key_dim, 2 + head_ratio}` |
| attn_gate.weight, ssm_out.weight | `{key_dim, head_ratio}` |
| ssm_dt.bias, ssm_a, ssm_alpha, ssm_beta | `{n_k_heads, head_ratio}` |
| conv state (r_cache) | `{key_dim * (d_conv - 1), 2 + head_ratio}` |
| GDN state (s_cache) | `{n_k_heads * head_v_dim^2, head_ratio}` |

27B numbers: head_k=head_v=128 (ssm.state_size), n_k_heads=16,
n_v_heads=48, key_dim=2048, value_dim=6144, conv_dim=10240.

## Rotation (load balance)

Each tensor gets `rotation = (index of same-type previous layers) % n_devices`
so rounding leftovers alternate between GPUs instead of always landing on
GPU0. Same-type = same is_recr and is_swa. Non-layer tensors use
`n_layer % n_devices`.

## tandem port plan

1. `tandem-model`: `split_state(name, hp, n_devices) -> SplitState` mirroring
   the table above; unit-test the segment arithmetic against the 27B's real
   tensor shapes (checkable offline with `tandem inspect`).
2. `Device::Meta(vec![0,1])` in `Weights::load`; the meta backend handles
   per-device placement of the loaded bytes.
3. Session tensors get classified too (our names, same axes as the
   cache_* patterns).
4. Prod-matching env: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `NCCL_P2P_DISABLE=1`,
   `NCCL_SHM_DISABLE=0`. No NVLink (user owns no bridge) — PHB forever.
5. Only then: 27B (29.3 GiB Q8) fits 2×24 GiB with f16 KV. It cannot fit on
   one card, so M3 has no single-GPU fallback — the meta path IS the gate.
