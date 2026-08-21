//! DFlash2 block-diffusion drafter (arch `dflash`), ported from llama.cpp
//! PR-27342 (`src/models/dflash.cpp` + the `draft_dflash` driver).
//!
//! The drafter is a 5-layer non-causal transformer over a sliding window of
//! the conversation, whose KV cache is filled not from tokens but from the
//! TRUNK's own layer-input hidden states (layers `target_layers`), fused by
//! a small encoder. One draft call feeds `[committed, MASK x block-1]` and
//! denoises the whole block in a single forward; sampling goes through a
//! learned top-16 transition lattice rather than raw logits. Up to 7 drafts
//! per call for roughly the weight-read cost of one MTP chain link.
//!
//! Portability deviations from the reference, all behavior-preserving:
//! - candidate ids are read back as their own I32 tensor instead of being
//!   f32-cast into the packed lattice row (the cast op has no reliable
//!   CUDA/TP path; see the ggml_arange incident in the vision port);
//! - the draft KV ring and its mask live here, not in a llama_kv_cache.

use codpiece_ggml_sys as ffi;
use codpiece_gguf::Value;

use crate::{Device, ModelError, Weights};

#[derive(Debug, Clone)]
pub struct DflashHparams {
    pub n_layer: usize,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub n_ff: i64,
    pub eps: f32,
    pub n_swa: i64,
    pub block_size: usize,
    pub conv_kernel: i64,
    pub conv_group: i64,
    pub selector_rank: i64,
    pub selector_top_k: usize,
    pub rope_base: f32,
    pub n_ctx_train: i64,
    pub target_layers: Vec<usize>,
    pub mask_token: u32,
}

impl DflashHparams {
    fn from_gguf(g: &codpiece_gguf::GgufFile) -> Result<DflashHparams, ModelError> {
        let u = |k: &str| -> Result<i64, ModelError> {
            g.kv(k)
                .and_then(Value::as_u64)
                .map(|v| v as i64)
                .ok_or_else(|| ModelError::Load(format!("dflash missing {k}")))
        };
        let layers = g
            .kv("dflash.target_layers")
            .and_then(Value::as_array)
            .ok_or_else(|| ModelError::Load("dflash missing target_layers".into()))?
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as usize)
            .collect::<Vec<_>>();
        Ok(DflashHparams {
            n_layer: u("dflash.block_count")? as usize,
            n_embd: u("dflash.embedding_length")?,
            n_head: u("dflash.attention.head_count")?,
            n_head_kv: u("dflash.attention.head_count_kv")?,
            head_dim: u("dflash.attention.key_length")?,
            n_ff: u("dflash.feed_forward_length")?,
            eps: g
                .kv("dflash.attention.layer_norm_rms_epsilon")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            n_swa: u("dflash.attention.sliding_window")?,
            block_size: u("dflash.block_size")? as usize,
            conv_kernel: u("dflash.conv_kernel_size")?,
            conv_group: u("dflash.conv_group_size")?,
            selector_rank: u("dflash.selector_rank")?,
            selector_top_k: u("dflash.selector_top_k")? as usize,
            rope_base: g
                .kv("dflash.rope.freq_base")
                .and_then(Value::as_f64)
                .unwrap_or(1e7) as f32,
            n_ctx_train: u("dflash.context_length")?,
            target_layers: layers,
            mask_token: g
                .kv("tokenizer.ggml.mask_token_id")
                .and_then(Value::as_u64)
                .ok_or_else(|| ModelError::Load("dflash missing mask token".into()))?
                as u32,
        })
    }

    /// concatenated trunk features per token: |target_layers| x trunk n_embd
    pub fn n_feat(&self) -> i64 {
        self.target_layers.len() as i64 * self.n_embd
    }
}

/// The drafter's weights plus borrowed pointers into the trunk it drafts
/// for: DFlash2 ships no token embedding and no lm head of its own.
pub struct Dflash {
    pub weights: Weights,
    pub hp: DflashHparams,
    /// trunk token_embd.weight (same backend as `weights`)
    tok_embd: *mut ffi::ggml_tensor,
    /// trunk output.weight
    output: *mut ffi::ggml_tensor,
}

unsafe impl Send for Dflash {}

/// Sliding-window K/V ring for the drafter, plus the block scratch. Slots
/// are `position % ring`; a slot's position is recoverable from the ring
/// arithmetic, so nothing tracks it. Rollback is positional overwrite.
pub struct DflashCache {
    ctx: *mut ffi::ggml_context,
    buffer: ffi::ggml_backend_buffer_t,
    galloc: ffi::ggml_gallocr_t,
    /// [head_dim * n_head_kv, ring] f16 per layer
    k: Vec<*mut ffi::ggml_tensor>,
    v: Vec<*mut ffi::ggml_tensor>,
    pub ring: usize,
    /// highest injected position + 1 (the drafter's view of history)
    pub n_seen: usize,
}

impl Drop for DflashCache {
    fn drop(&mut self) {
        unsafe {
            ffi::ggml_gallocr_free(self.galloc);
            ffi::ggml_backend_buffer_free(self.buffer);
            ffi::ggml_free(self.ctx);
        }
    }
}

/// One draft call's readback: per-position candidate ids and the transition
/// score matrix, walked on the host.
pub struct Lattice {
    /// [block][top_k] candidate token ids (positions 1..block are drafts)
    pub cand: Vec<Vec<u32>>,
    /// [block][pred * top_k + succ] transition scores
    pub scores: Vec<Vec<f32>>,
    pub top_k: usize,
}

impl Lattice {
    /// Greedy walk: at each drafted position take the best transition from
    /// the current predecessor slot; the anchor conditions position 1.
    pub fn walk_greedy(&self, n_max: usize) -> Vec<u32> {
        self.walk(n_max, |scores| {
            let mut best = 0usize;
            for (i, s) in scores.iter().enumerate() {
                if *s > scores[best] {
                    best = i;
                }
            }
            best
        })
    }

    /// Temperature walk with an explicit uniform per step (caller owns RNG).
    pub fn walk_sampled(&self, n_max: usize, temp: f32, mut uniform: impl FnMut() -> f32) -> Vec<u32> {
        self.walk(n_max, |scores| {
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ps: Vec<f32> = scores.iter().map(|s| ((s - m) / temp).exp()).collect();
            let sum: f32 = ps.iter().sum();
            let mut r = uniform() * sum;
            for (i, p) in ps.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    return i;
                }
            }
            ps.len() - 1
        })
    }

    fn walk(&self, n_max: usize, mut pick: impl FnMut(&[f32]) -> usize) -> Vec<u32> {
        let k = self.top_k;
        let mut out = Vec::new();
        let mut pred = 0usize;
        for pos in 1..self.cand.len().min(n_max + 1) {
            let row = &self.scores[pos][pred * k..(pred + 1) * k];
            let j = pick(row);
            out.push(self.cand[pos][j]);
            pred = j;
        }
        out
    }
}

impl Dflash {
    /// `trunk` supplies the shared token embedding and lm head; it must live
    /// on the same device as `path` is loaded to.
    pub fn load(
        path: &std::path::Path,
        device: Device,
        trunk: &crate::qwen35::Qwen35,
    ) -> Result<Dflash, ModelError> {
        let weights = Weights::load(path, device)?;
        let hp = DflashHparams::from_gguf(&weights.gguf)?;
        if hp.n_embd != trunk.hp.n_embd {
            return Err(ModelError::Load(format!(
                "dflash n_embd {} != trunk {}",
                hp.n_embd, trunk.hp.n_embd
            )));
        }
        for name in [
            "fc.weight",
            "enc.output_norm.weight",
            "output_norm.weight",
            "selector_hidden.weight",
            "selector_predecessor.weight",
            "selector_successor.weight",
        ] {
            if weights.tensor(name).is_none() {
                return Err(ModelError::Load(format!("dflash missing {name}")));
            }
        }
        let tok_embd = trunk
            .weights
            .tensor("token_embd.weight")
            .ok_or_else(|| ModelError::Load("trunk has no token_embd".into()))?;
        let output = trunk
            .weights
            .tensor("output.weight")
            .unwrap_or(tok_embd);
        Ok(Dflash { weights, hp, tok_embd, output })
    }

    fn t(&self, name: &str) -> Result<*mut ffi::ggml_tensor, ModelError> {
        self.weights
            .tensor(name)
            .ok_or_else(|| ModelError::Load(format!("dflash missing tensor {name}")))
    }

    fn blk(&self, il: usize, part: &str) -> Result<*mut ffi::ggml_tensor, ModelError> {
        self.t(&format!("blk.{il}.{part}"))
    }

    pub fn new_cache(&self) -> Result<DflashCache, ModelError> {
        let hp = &self.hp;
        // block queries can look `n_swa` behind the block start
        let ring = (hp.n_swa as usize) + hp.block_size;
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: (hp.n_layer * 2 + 8) * ffi::ggml_tensor_overhead(),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(ModelError::Load("dflash cache ctx".into()));
            }
            let f16t = ffi::ggml_type_GGML_TYPE_F16;
            let kv_dim = hp.head_dim * hp.n_head_kv;
            let mut k = Vec::new();
            let mut v = Vec::new();
            for il in 0..hp.n_layer {
                let kt = ffi::ggml_new_tensor_2d(ctx, f16t, kv_dim, ring as i64);
                let vt = ffi::ggml_new_tensor_2d(ctx, f16t, kv_dim, ring as i64);
                let name = |t: *mut ffi::ggml_tensor, n: String| {
                    if let Ok(c) = std::ffi::CString::new(n) {
                        ffi::ggml_set_name(t, c.as_ptr());
                    }
                };
                name(kt, format!("dflash_k_l{il}"));
                name(vt, format!("dflash_v_l{il}"));
                k.push(kt);
                v.push(vt);
            }
            let buffer = ffi::ggml_backend_alloc_ctx_tensors(ctx, self.weights.backend());
            if buffer.is_null() {
                ffi::ggml_free(ctx);
                return Err(ModelError::Load("dflash cache alloc".into()));
            }
            ffi::ggml_backend_buffer_clear(buffer, 0);
            let galloc = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(
                self.weights.backend(),
            ));
            if galloc.is_null() {
                ffi::ggml_backend_buffer_free(buffer);
                ffi::ggml_free(ctx);
                return Err(ModelError::Load("dflash cache gallocr".into()));
            }
            Ok(DflashCache { ctx, buffer, galloc, k, v, ring, n_seen: 0 })
        }
    }

    /// Inject trunk features for positions `[pos0, pos0 + t)` into the draft
    /// KV ring. `features` is `[n_feat, t]` row-major per position — the
    /// concatenated layer-input hiddens the trunk graph tapped.
    pub fn inject(
        &self,
        cache: &mut DflashCache,
        features: &[f32],
        pos0: usize,
        n_threads: i32,
    ) -> Result<(), ModelError> {
        let hp = &self.hp;
        let n_feat = hp.n_feat();
        let t = features.len() / n_feat as usize;
        if t == 0 || features.len() != t * n_feat as usize {
            return Err(ModelError::Load("dflash inject: bad feature buffer".into()));
        }
        unsafe {
            let graph_nodes = hp.n_layer * 16 + 32;
            let params = ffi::ggml_init_params {
                mem_size: (graph_nodes * 2) * ffi::ggml_tensor_overhead()
                    + ffi::ggml_graph_overhead_custom(graph_nodes, false),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            let gf = ffi::ggml_new_graph_custom(ctx, graph_nodes, false);
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _g = G(ctx);
            let f32t = ffi::ggml_type_GGML_TYPE_F32;

            let inp = ffi::ggml_new_tensor_2d(ctx, f32t, n_feat, t as i64);
            ffi::ggml_set_input(inp);
            let inp_pos = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t as i64);
            ffi::ggml_set_input(inp_pos);
            let rows = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I64, t as i64);
            ffi::ggml_set_input(rows);

            let rms = |ctx: *mut ffi::ggml_context,
                       x: *mut ffi::ggml_tensor,
                       w: *mut ffi::ggml_tensor| {
                let n = ffi::ggml_rms_norm(ctx, x, hp.eps);
                ffi::ggml_mul(ctx, n, w)
            };

            // encoder: fused features -> draft hidden
            let mut h = ffi::ggml_mul_mat(ctx, self.t("fc.weight")?, inp);
            h = rms(ctx, h, self.t("enc.output_norm.weight")?);

            for il in 0..hp.n_layer {
                let wk = self.blk(il, "attn_k.weight")?;
                let wv = self.blk(il, "attn_v.weight")?;
                let k_norm = self.blk(il, "attn_k_norm.weight")?;

                let mut kc = ffi::ggml_mul_mat(ctx, wk, h);
                kc = ffi::ggml_reshape_3d(ctx, kc, hp.head_dim, hp.n_head_kv, t as i64);
                kc = rms(ctx, kc, k_norm);
                kc = ffi::ggml_rope_ext(
                    ctx,
                    kc,
                    inp_pos,
                    std::ptr::null_mut(),
                    hp.head_dim as i32,
                    ffi::GGML_ROPE_TYPE_NEOX as i32,
                    hp.n_ctx_train as i32,
                    hp.rope_base,
                    1.0,
                    0.0,
                    1.0,
                    32.0,
                    1.0,
                );
                let vc = ffi::ggml_mul_mat(ctx, wv, h);

                let kv_dim = hp.head_dim * hp.n_head_kv;
                let kf = ffi::ggml_cast(
                    ctx,
                    ffi::ggml_reshape_2d(ctx, kc, kv_dim, t as i64),
                    ffi::ggml_type_GGML_TYPE_F16,
                );
                let vf = ffi::ggml_cast(ctx, vc, ffi::ggml_type_GGML_TYPE_F16);
                let kw = ffi::ggml_set_rows(ctx, cache.k[il], kf, rows);
                let vw = ffi::ggml_set_rows(ctx, cache.v[il], vf, rows);
                ffi::ggml_build_forward_expand(gf, kw);
                ffi::ggml_build_forward_expand(gf, vw);
            }

            if !ffi::ggml_gallocr_alloc_graph(cache.galloc, gf) {
                return Err(ModelError::Load("dflash inject alloc".into()));
            }
            ffi::ggml_backend_tensor_set(inp, features.as_ptr().cast(), 0, features.len() * 4);
            let pos: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
            ffi::ggml_backend_tensor_set(inp_pos, pos.as_ptr().cast(), 0, t * 4);
            let ring_rows: Vec<i64> = (0..t).map(|i| ((pos0 + i) % cache.ring) as i64).collect();
            ffi::ggml_backend_tensor_set(rows, ring_rows.as_ptr().cast(), 0, t * 8);
            self.compute(gf, n_threads)?;
        }
        cache.n_seen = cache.n_seen.max(pos0 + t);
        Ok(())
    }

    /// One block-diffusion draft call: `[anchor @ pos0, MASK x (block-1)]`,
    /// non-causal over the injected window plus the block itself. Returns
    /// the selector lattice for the host walk.
    pub fn draft_block(
        &self,
        cache: &DflashCache,
        anchor: u32,
        pos0: usize,
        n_threads: i32,
    ) -> Result<Lattice, ModelError> {
        let hp = &self.hp;
        let b = hp.block_size as i64;
        let top_k = hp.selector_top_k as i64;
        unsafe {
            let graph_nodes = hp.n_layer * 120 + hp.block_size * 40 + 128;
            let params = ffi::ggml_init_params {
                mem_size: (graph_nodes * 2) * ffi::ggml_tensor_overhead()
                    + ffi::ggml_graph_overhead_custom(graph_nodes, false)
                    + (4 << 20),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            let gf = ffi::ggml_new_graph_custom(ctx, graph_nodes, false);
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _g = G(ctx);
            let f32t = ffi::ggml_type_GGML_TYPE_F32;

            let inp_tokens = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, b);
            ffi::ggml_set_input(inp_tokens);
            let inp_pos = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, b);
            ffi::ggml_set_input(inp_pos);
            // keys: the whole ring plus the in-batch block; mask selects
            let n_kv = cache.ring as i64;
            let kq_mask = ffi::ggml_new_tensor_2d(ctx, f32t, n_kv + b, b);
            ffi::ggml_set_input(kq_mask);

            let rms = |ctx: *mut ffi::ggml_context,
                       x: *mut ffi::ggml_tensor,
                       w: *mut ffi::ggml_tensor| {
                let n = ffi::ggml_rms_norm(ctx, x, hp.eps);
                ffi::ggml_mul(ctx, n, w)
            };

            let mut inp_l = ffi::ggml_get_rows(ctx, self.tok_embd, inp_tokens);

            // grouped dynamic conv, block-internal shift (reference
            // build_dflash2_conv with n_blocks = 1)
            let conv = |ctx: *mut ffi::ggml_context,
                        gf: *mut ffi::ggml_cgraph,
                        hidden: *mut ffi::ggml_tensor,
                        dynamic: *mut ffi::ggml_tensor,
                        base: *mut ffi::ggml_tensor,
                        side: i64|
             -> *mut ffi::ggml_tensor {
                let _ = gf;
                let n_embd = hp.n_embd;
                let n_groups = n_embd / hp.conv_group;
                let kernel = hp.conv_kernel;
                let hidden = ffi::ggml_cont_2d(ctx, hidden, n_embd, b);
                let coeffs = ffi::ggml_reshape_4d(ctx, dynamic, n_groups, kernel, 2, b);
                let mut result: *mut ffi::ggml_tensor = std::ptr::null_mut();
                for tap in 0..kernel {
                    // values: hidden shifted down `tap` positions inside the block
                    let values = if tap == 0 {
                        hidden
                    } else {
                        let zeros = ffi::ggml_scale(
                            ctx,
                            ffi::ggml_view_2d(
                                ctx,
                                hidden,
                                n_embd,
                                tap.min(b),
                                ffi::ggml_row_size(f32t, n_embd),
                                0,
                            ),
                            0.0,
                        );
                        if tap < b {
                            let prev = ffi::ggml_view_2d(
                                ctx,
                                hidden,
                                n_embd,
                                b - tap,
                                ffi::ggml_row_size(f32t, n_embd),
                                0,
                            );
                            ffi::ggml_concat(ctx, zeros, prev, 1)
                        } else {
                            zeros
                        }
                    };
                    // coeff for this tap/side: [n_groups] per token -> repeat
                    // to n_embd rows
                    let c4 = coeffs;
                    let coeff = ffi::ggml_cont(
                        ctx,
                        ffi::ggml_view_2d(
                            ctx,
                            c4,
                            n_groups,
                            b,
                            (*c4).nb[3],
                            (tap as usize) * (*c4).nb[1] + (side as usize) * (*c4).nb[2],
                        ),
                    );
                    let coeff = ffi::ggml_reshape_3d(ctx, coeff, 1, n_groups, b);
                    let grouped_shape =
                        ffi::ggml_new_tensor_3d(ctx, f32t, hp.conv_group, n_groups, b);
                    let coeff = ffi::ggml_repeat(ctx, coeff, grouped_shape);
                    let coeff = ffi::ggml_reshape_2d(ctx, coeff, n_embd, b);
                    // static base for this tap/side: [n_embd]
                    let base_tap = ffi::ggml_view_1d(
                        ctx,
                        base,
                        n_embd,
                        (tap as usize) * (*base).nb[1] + (side as usize) * (*base).nb[2],
                    );
                    let weight =
                        ffi::ggml_add(ctx, coeff, ffi::ggml_repeat(ctx, base_tap, hidden));
                    let term = ffi::ggml_mul(ctx, weight, values);
                    result = if result.is_null() {
                        term
                    } else {
                        ffi::ggml_add(ctx, result, term)
                    };
                }
                result
            };

            let kq_scale = 1.0f32 / (hp.head_dim as f32).sqrt();
            for il in 0..hp.n_layer {
                let residual = inp_l;
                let mut x = rms(ctx, inp_l, self.blk(il, "attn_norm.weight")?);
                let dynamic = ffi::ggml_mul_mat(ctx, self.blk(il, "attn_conv_proj.weight")?, x);
                x = conv(ctx, gf, x, dynamic, self.blk(il, "attn_conv_base")?, 0);

                let mut q = ffi::ggml_mul_mat(ctx, self.blk(il, "attn_q.weight")?, x);
                let mut k = ffi::ggml_mul_mat(ctx, self.blk(il, "attn_k.weight")?, x);
                let v = ffi::ggml_mul_mat(ctx, self.blk(il, "attn_v.weight")?, x);
                q = ffi::ggml_reshape_3d(ctx, q, hp.head_dim, hp.n_head, b);
                k = ffi::ggml_reshape_3d(ctx, k, hp.head_dim, hp.n_head_kv, b);
                q = rms(ctx, q, self.blk(il, "attn_q_norm.weight")?);
                k = rms(ctx, k, self.blk(il, "attn_k_norm.weight")?);
                let rope = |ctx: *mut ffi::ggml_context, t: *mut ffi::ggml_tensor| {
                    ffi::ggml_rope_ext(
                        ctx,
                        t,
                        inp_pos,
                        std::ptr::null_mut(),
                        hp.head_dim as i32,
                        ffi::GGML_ROPE_TYPE_NEOX as i32,
                        hp.n_ctx_train as i32,
                        hp.rope_base,
                        1.0,
                        0.0,
                        1.0,
                        32.0,
                        1.0,
                    )
                };
                q = rope(ctx, q);
                k = rope(ctx, k);

                // keys/values = cached ring (f16 -> f32) ++ this block
                let kv_dim = hp.head_dim * hp.n_head_kv;
                let ring_k = ffi::ggml_cast(ctx, cache.k[il], f32t);
                let ring_v = ffi::ggml_cast(ctx, cache.v[il], f32t);
                let blk_k = ffi::ggml_reshape_2d(ctx, k, kv_dim, b);
                let all_k = ffi::ggml_concat(ctx, ring_k, blk_k, 1);
                let all_v = ffi::ggml_concat(ctx, ring_v, v, 1);
                let n_keys = n_kv + b;

                let kh = ffi::ggml_permute(
                    ctx,
                    ffi::ggml_reshape_3d(ctx, all_k, hp.head_dim, hp.n_head_kv, n_keys),
                    0,
                    2,
                    1,
                    3,
                );
                let vh = ffi::ggml_cont(
                    ctx,
                    ffi::ggml_permute(
                        ctx,
                        ffi::ggml_reshape_3d(ctx, all_v, hp.head_dim, hp.n_head_kv, n_keys),
                        1,
                        2,
                        0,
                        3,
                    ),
                );
                let qh = ffi::ggml_permute(ctx, q, 0, 2, 1, 3);
                let kq = ffi::ggml_mul_mat(ctx, kh, qh);
                let kq = ffi::ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0);
                let kqv = ffi::ggml_mul_mat(ctx, vh, kq);
                let mut o = ffi::ggml_permute(ctx, kqv, 0, 2, 1, 3);
                o = ffi::ggml_cont_2d(ctx, o, hp.head_dim * hp.n_head, b);
                o = ffi::ggml_mul_mat(ctx, self.blk(il, "attn_output.weight")?, o);
                o = conv(ctx, gf, o, dynamic, self.blk(il, "attn_conv_base")?, 1);

                let ffn_inp = ffi::ggml_add(ctx, o, residual);
                let mut f = rms(ctx, ffn_inp, self.blk(il, "ffn_norm.weight")?);
                let ffn_dynamic =
                    ffi::ggml_mul_mat(ctx, self.blk(il, "ffn_conv_proj.weight")?, f);
                f = conv(ctx, gf, f, ffn_dynamic, self.blk(il, "ffn_conv_base")?, 0);
                let up = ffi::ggml_mul_mat(ctx, self.blk(il, "ffn_up.weight")?, f);
                let gate = ffi::ggml_mul_mat(ctx, self.blk(il, "ffn_gate.weight")?, f);
                let act = ffi::ggml_mul(ctx, ffi::ggml_silu(ctx, gate), up);
                let mut down = ffi::ggml_mul_mat(ctx, self.blk(il, "ffn_down.weight")?, act);
                down = conv(ctx, gf, down, ffn_dynamic, self.blk(il, "ffn_conv_base")?, 1);
                inp_l = ffi::ggml_add(ctx, down, ffn_inp);
            }

            let h_out = rms(ctx, inp_l, self.t("output_norm.weight")?);
            let logits = ffi::ggml_mul_mat(ctx, self.output, h_out);

            // selector lattice
            let cand = ffi::ggml_top_k(ctx, logits, top_k as i32);
            ffi::ggml_set_output(cand);
            ffi::ggml_build_forward_expand(gf, cand);
            let vocab_rows = (*logits).ne[0];
            let unary = ffi::ggml_get_rows(
                ctx,
                ffi::ggml_reshape_3d(ctx, logits, 1, vocab_rows, b),
                ffi::ggml_reshape_2d(ctx, cand, top_k, b),
            );
            let unary = ffi::ggml_reshape_2d(ctx, unary, top_k, b);
            let hidden =
                ffi::ggml_mul_mat(ctx, self.t("selector_hidden.weight")?, h_out); // [rank, b]

            let sel_prev = self.t("selector_predecessor.weight")?;
            let sel_next = self.t("selector_successor.weight")?;
            let anchor_ids = ffi::ggml_view_1d(ctx, inp_tokens, 1, 0);

            let mut packed: Vec<*mut ffi::ggml_tensor> = Vec::new();
            let rank = hp.selector_rank;
            for pos in 1..hp.block_size as i64 {
                let ids_pos = ffi::ggml_cont(
                    ctx,
                    ffi::ggml_view_1d(ctx, cand, top_k, (pos as usize) * (*cand).nb[1]),
                );
                let successor = ffi::ggml_get_rows(ctx, sel_next, ids_pos); // [rank, k]
                let hidden_pos = ffi::ggml_view_2d(
                    ctx,
                    hidden,
                    rank,
                    1,
                    ffi::ggml_row_size(f32t, rank),
                    (pos as usize) * (*hidden).nb[1],
                );
                let predecessor = if pos == 1 {
                    ffi::ggml_get_rows(ctx, sel_prev, anchor_ids) // [rank, 1]
                } else {
                    let prev_ids = ffi::ggml_cont(
                        ctx,
                        ffi::ggml_view_1d(
                            ctx,
                            cand,
                            top_k,
                            (pos as usize - 1) * (*cand).nb[1],
                        ),
                    );
                    ffi::ggml_get_rows(ctx, sel_prev, prev_ids) // [rank, k]
                };
                let conditioned = ffi::ggml_mul(
                    ctx,
                    predecessor,
                    ffi::ggml_repeat(ctx, hidden_pos, predecessor),
                );
                let mut scores = ffi::ggml_mul_mat(ctx, successor, conditioned); // [k, P]
                if pos == 1 {
                    scores = ffi::ggml_repeat_4d(ctx, scores, top_k, top_k, 1, 1);
                }
                let unary_pos = ffi::ggml_view_2d(
                    ctx,
                    unary,
                    top_k,
                    1,
                    ffi::ggml_row_size(f32t, top_k),
                    (pos as usize) * (*unary).nb[1],
                );
                // scores[succ j, pred i] += unary[j]
                let scores = ffi::ggml_add(
                    ctx,
                    scores,
                    ffi::ggml_repeat(
                        ctx,
                        ffi::ggml_reshape_2d(ctx, ffi::ggml_cont(ctx, unary_pos), top_k, 1),
                        scores,
                    ),
                );
                let flat = ffi::ggml_reshape_1d(ctx, ffi::ggml_cont(ctx, scores), top_k * top_k);
                ffi::ggml_set_output(flat);
                ffi::ggml_build_forward_expand(gf, flat);
                packed.push(flat);
            }

            if !ffi::ggml_gallocr_alloc_graph(cache.galloc, gf) {
                return Err(ModelError::Load("dflash draft alloc".into()));
            }

            // inputs
            let mut toks = vec![hp.mask_token as i32; hp.block_size];
            toks[0] = anchor as i32;
            ffi::ggml_backend_tensor_set(inp_tokens, toks.as_ptr().cast(), 0, toks.len() * 4);
            let pos: Vec<i32> = (0..hp.block_size).map(|i| (pos0 + i) as i32).collect();
            ffi::ggml_backend_tensor_set(inp_pos, pos.as_ptr().cast(), 0, pos.len() * 4);

            // mask: ring slots hold positions in (n_seen - ring, n_seen);
            // a key at position pk is visible to query at position pq iff
            // pk < pos0 (already injected; the block itself provides pos0..)
            // and pq - pk < n_swa. Block keys are all mutually visible.
            let n_keys = (cache.ring + hp.block_size) as usize;
            let mut mask = vec![f32::NEG_INFINITY; n_keys * hp.block_size];
            for qi in 0..hp.block_size {
                let pq = pos0 + qi;
                for s in 0..cache.ring {
                    // the position currently held by ring slot s
                    let held = if cache.n_seen == 0 {
                        None
                    } else {
                        let top = cache.n_seen - 1;
                        let cand = top - ((top + cache.ring - s) % cache.ring);
                        // cand is the largest pos <= top with pos % ring == s
                        if cand + cache.ring > cache.n_seen || cand > top {
                            Some(cand)
                        } else {
                            None
                        }
                    };
                    if let Some(pk) = held {
                        if pk < pos0 && pq >= pk && (pq - pk) < hp.n_swa as usize {
                            mask[qi * n_keys + s] = 0.0;
                        }
                    }
                }
                for bi in 0..hp.block_size {
                    mask[qi * n_keys + cache.ring + bi] = 0.0;
                }
            }
            ffi::ggml_backend_tensor_set(kq_mask, mask.as_ptr().cast(), 0, mask.len() * 4);

            self.compute(gf, n_threads)?;

            // readback
            let mut cand_ids = vec![0i32; hp.block_size * hp.selector_top_k];
            ffi::ggml_backend_tensor_get(
                cand,
                cand_ids.as_mut_ptr().cast(),
                0,
                cand_ids.len() * 4,
            );
            let k = hp.selector_top_k;
            let mut lat = Lattice {
                cand: (0..hp.block_size)
                    .map(|p| cand_ids[p * k..(p + 1) * k].iter().map(|&x| x as u32).collect())
                    .collect(),
                scores: vec![Vec::new(); hp.block_size],
                top_k: k,
            };
            for (i, t) in packed.iter().enumerate() {
                let mut s = vec![0f32; k * k];
                ffi::ggml_backend_tensor_get(*t, s.as_mut_ptr().cast(), 0, s.len() * 4);
                // graph layout: [succ, pred]; host walk wants pred-major rows
                let mut rows = vec![0f32; k * k];
                for pred in 0..k {
                    for succ in 0..k {
                        rows[pred * k + succ] = s[pred * k + succ];
                    }
                }
                lat.scores[i + 1] = rows;
            }
            Ok(lat)
        }
    }

    unsafe fn compute(&self, gf: *mut ffi::ggml_cgraph, n_threads: i32) -> Result<(), ModelError> {
        let backend = self.weights.backend();
        if self.weights.is_cpu() && n_threads > 0 {
            ffi::ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
        if ffi::ggml_backend_graph_compute(backend, gf) != ffi::ggml_status_GGML_STATUS_SUCCESS {
            return Err(ModelError::Load("dflash compute".into()));
        }
        Ok(())
    }
}
