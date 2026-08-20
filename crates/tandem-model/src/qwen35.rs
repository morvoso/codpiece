//! qwen35 forward graph (trunk only: GDN + full-attention layers).
//!
//! Faithful port of llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp`
//! (snapshot in notes/reference/). This is the M1 *stateless* build: the whole
//! token prefix is recomputed each step with zero-initialized recurrent
//! states and no KV cache. O(n²) and proud — it exists to be diffed against
//! llama.cpp, not to be fast. State carry and caching land in M3.

use std::collections::HashMap;

use tandem_ggml_sys as ffi;
use tandem_gguf::Value;

use crate::{ModelError, Weights};

pub struct Hparams {
    pub n_layer: usize,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_k: i64,
    pub head_v: i64,
    pub n_ff: i64,
    pub rms_eps: f32,
    pub n_rot: i32,
    pub rope_sections: [i32; 4],
    pub freq_base: f32,
    pub n_ctx_train: i32,
    // GDN
    pub d_conv: i64,
    pub n_k_heads: i64, // ssm.group_count
    pub n_v_heads: i64, // ssm.time_step_rank
    pub d_state: i64,   // ssm.state_size (= gdn head dim)
    pub d_inner: i64,   // ssm.inner_size
    pub full_attn_interval: usize,
    pub n_vocab: i64,
}

impl Hparams {
    pub fn from_gguf(g: &tandem_gguf::GgufFile) -> Result<Hparams, ModelError> {
        let arch = g
            .architecture()
            .ok_or_else(|| ModelError::Load("no architecture".into()))?;
        if arch != "qwen35" {
            return Err(ModelError::Load(format!("arch {arch:?}, want qwen35")));
        }
        let k = |key: &str| -> Option<u64> {
            g.kv(&format!("qwen35.{key}")).and_then(Value::as_u64)
        };
        let kf = |key: &str| -> Option<f64> {
            g.kv(&format!("qwen35.{key}")).and_then(Value::as_f64)
        };
        let req = |key: &'static str| -> Result<u64, ModelError> {
            k(key).ok_or(ModelError::Load(format!("missing qwen35.{key}")))
        };

        let n_layer_all = req("block_count")? as usize;
        let n_nextn = k("nextn_predict_layers").unwrap_or(0) as usize;
        let n_embd = req("embedding_length")? as i64;
        let n_head = req("attention.head_count")? as i64;
        let head_k = k("attention.key_length").map(|v| v as i64).unwrap_or(n_embd / n_head);
        let head_v = k("attention.value_length").map(|v| v as i64).unwrap_or(head_k);

        let mut sections = [0i32; 4];
        if let Some(arr) = g.kv("qwen35.rope.dimension_sections").and_then(Value::as_array) {
            for (i, v) in arr.iter().take(4).enumerate() {
                sections[i] = v.as_u64().unwrap_or(0) as i32;
            }
        }

        let tokens_len = g
            .kv("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .map(|a| a.len() as i64)
            .ok_or_else(|| ModelError::Load("missing tokenizer tokens".into()))?;

        Ok(Hparams {
            n_layer: n_layer_all - n_nextn,
            n_embd,
            n_head,
            n_head_kv: req("attention.head_count_kv")? as i64,
            head_k,
            head_v,
            n_ff: req("feed_forward_length")? as i64,
            rms_eps: kf("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32,
            n_rot: k("rope.dimension_count").map(|v| v as i32).unwrap_or(head_k as i32),
            rope_sections: sections,
            freq_base: kf("rope.freq_base").unwrap_or(10000.0) as f32,
            n_ctx_train: req("context_length")? as i32,
            d_conv: req("ssm.conv_kernel")? as i64,
            n_k_heads: req("ssm.group_count")? as i64,
            n_v_heads: req("ssm.time_step_rank")? as i64,
            d_state: req("ssm.state_size")? as i64,
            d_inner: req("ssm.inner_size")? as i64,
            full_attn_interval: k("full_attention_interval").unwrap_or(4) as usize,
            n_vocab: tokens_len,
        })
    }

    pub fn is_recurrent(&self, il: usize) -> bool {
        (il + 1) % self.full_attn_interval != 0
    }

    pub fn key_dim(&self) -> i64 {
        self.d_state * self.n_k_heads
    }

    pub fn value_dim(&self) -> i64 {
        // == d_inner; head_v_dim per GDN head is d_inner / n_v_heads == d_state
        self.d_inner
    }

    pub fn conv_dim(&self) -> i64 {
        2 * self.key_dim() + self.value_dim()
    }

    pub fn gdn_head_v(&self) -> i64 {
        self.d_inner / self.n_v_heads
    }
}

pub struct Qwen35 {
    pub weights: Weights,
    pub hp: Hparams,
}

/// Layer tensor handles, resolved once at load.
struct Layer {
    attn_norm: *mut ffi::ggml_tensor,
    post_attn_norm: *mut ffi::ggml_tensor,
    // full attention
    wq: *mut ffi::ggml_tensor,
    wk: *mut ffi::ggml_tensor,
    wv: *mut ffi::ggml_tensor,
    wo: *mut ffi::ggml_tensor,
    q_norm: *mut ffi::ggml_tensor,
    k_norm: *mut ffi::ggml_tensor,
    // gdn
    wqkv: *mut ffi::ggml_tensor,
    wqkv_gate: *mut ffi::ggml_tensor,
    conv1d: *mut ffi::ggml_tensor,
    dt_bias: *mut ffi::ggml_tensor,
    ssm_a: *mut ffi::ggml_tensor,
    ssm_beta: *mut ffi::ggml_tensor,
    ssm_alpha: *mut ffi::ggml_tensor,
    ssm_norm: *mut ffi::ggml_tensor,
    ssm_out: *mut ffi::ggml_tensor,
    // ffn
    ffn_gate: *mut ffi::ggml_tensor,
    ffn_up: *mut ffi::ggml_tensor,
    ffn_down: *mut ffi::ggml_tensor,
}

impl Qwen35 {
    pub fn load(path: &std::path::Path) -> Result<Qwen35, ModelError> {
        let weights = Weights::load(path, crate::Device::Cpu)?;
        let hp = Hparams::from_gguf(&weights.gguf)?;
        Ok(Qwen35 { weights, hp })
    }

    fn t(&self, name: &str) -> Result<*mut ffi::ggml_tensor, ModelError> {
        self.weights
            .tensor(name)
            .ok_or_else(|| ModelError::Load(format!("missing tensor {name}")))
    }

    fn layer(&self, il: usize) -> Result<Layer, ModelError> {
        let n = |suffix: &str| format!("blk.{il}.{suffix}");
        let opt = |name: &str| self.weights.tensor(name).unwrap_or(std::ptr::null_mut());
        let recurrent = self.hp.is_recurrent(il);
        Ok(Layer {
            attn_norm: self.t(&n("attn_norm.weight"))?,
            post_attn_norm: self.t(&n("post_attention_norm.weight"))?,
            wq: if recurrent { std::ptr::null_mut() } else { self.t(&n("attn_q.weight"))? },
            wk: if recurrent { std::ptr::null_mut() } else { self.t(&n("attn_k.weight"))? },
            wv: if recurrent { std::ptr::null_mut() } else { self.t(&n("attn_v.weight"))? },
            wo: if recurrent { std::ptr::null_mut() } else { self.t(&n("attn_output.weight"))? },
            q_norm: opt(&n("attn_q_norm.weight")),
            k_norm: opt(&n("attn_k_norm.weight")),
            wqkv: if recurrent { self.t(&n("attn_qkv.weight"))? } else { std::ptr::null_mut() },
            wqkv_gate: if recurrent { self.t(&n("attn_gate.weight"))? } else { std::ptr::null_mut() },
            conv1d: if recurrent { self.t(&n("ssm_conv1d.weight"))? } else { std::ptr::null_mut() },
            dt_bias: if recurrent { self.t(&n("ssm_dt.bias"))? } else { std::ptr::null_mut() },
            ssm_a: if recurrent { self.t(&n("ssm_a"))? } else { std::ptr::null_mut() },
            ssm_beta: if recurrent { self.t(&n("ssm_beta.weight"))? } else { std::ptr::null_mut() },
            ssm_alpha: if recurrent { self.t(&n("ssm_alpha.weight"))? } else { std::ptr::null_mut() },
            ssm_norm: if recurrent { self.t(&n("ssm_norm.weight"))? } else { std::ptr::null_mut() },
            ssm_out: if recurrent { self.t(&n("ssm_out.weight"))? } else { std::ptr::null_mut() },
            ffn_gate: self.t(&n("ffn_gate.weight"))?,
            ffn_up: self.t(&n("ffn_up.weight"))?,
            ffn_down: self.t(&n("ffn_down.weight"))?,
        })
    }

    /// Stateless forward over `tokens`; returns logits for the LAST position.
    pub fn forward_logits(&self, tokens: &[u32], n_threads: i32) -> Result<Vec<f32>, ModelError> {
        assert!(!tokens.is_empty());
        let hp = &self.hp;
        let t_len = tokens.len() as i64;

        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: 64 << 20, // graph metadata only (no_alloc)
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(ModelError::Load("graph ctx init".into()));
            }
            // Free ctx + galloc on every exit path.
            struct CtxGuard(*mut ffi::ggml_context, ffi::ggml_gallocr_t);
            impl Drop for CtxGuard {
                fn drop(&mut self) {
                    unsafe {
                        if !self.1.is_null() {
                            ffi::ggml_gallocr_free(self.1);
                        }
                        ffi::ggml_free(self.0);
                    }
                }
            }
            let mut guard = CtxGuard(ctx, std::ptr::null_mut());

            let gf = ffi::ggml_new_graph_custom(ctx, 8192, false);

            // ---- inputs ----
            let inp_tokens = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len);
            ffi::ggml_set_input(inp_tokens);
            let inp_pos = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len * 4);
            ffi::ggml_set_input(inp_pos);
            let kq_mask = ffi::ggml_new_tensor_2d(ctx, ffi::ggml_type_GGML_TYPE_F32, t_len, t_len);
            ffi::ggml_set_input(kq_mask);
            let conv_zero = ffi::ggml_new_tensor_3d(
                ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                hp.d_conv - 1,
                hp.conv_dim(),
                1,
            );
            ffi::ggml_set_input(conv_zero);
            let state_zero = ffi::ggml_new_tensor_4d(
                ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                hp.gdn_head_v(),
                hp.gdn_head_v(),
                hp.n_v_heads,
                1,
            );
            ffi::ggml_set_input(state_zero);
            let out_ids = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, 1);
            ffi::ggml_set_input(out_ids);

            // Elementwise ops (mul/add/ssm_conv) need f32 operands; quantized
            // files may store small tensors as bf16/f16. Cast in-graph.
            let as_f32 = |ctx: *mut ffi::ggml_context, t: *mut ffi::ggml_tensor| {
                if (*t).type_ == ffi::ggml_type_GGML_TYPE_F32 {
                    t
                } else {
                    ffi::ggml_cast(ctx, t, ffi::ggml_type_GGML_TYPE_F32)
                }
            };
            let rms = |ctx: *mut ffi::ggml_context, x: *mut ffi::ggml_tensor, w: *mut ffi::ggml_tensor| {
                let n = ffi::ggml_rms_norm(ctx, x, hp.rms_eps);
                ffi::ggml_mul(ctx, n, as_f32(ctx, w))
            };

            // ---- trunk ----
            let tok_embd = self.t("token_embd.weight")?;
            let mut cur;
            let mut inp_l = ffi::ggml_get_rows(ctx, tok_embd, inp_tokens); // [n_embd, T]

            let mut sections = hp.rope_sections;
            let elt = ffi::ggml_type_size(ffi::ggml_type_GGML_TYPE_F32);

            for il in 0..hp.n_layer {
                let l = self.layer(il)?;
                let inp_sa = inp_l;

                cur = rms(ctx, inp_l, l.attn_norm);

                if hp.is_recurrent(il) {
                    // ---- gated delta net ----
                    let key_dim = hp.key_dim();
                    let value_dim = hp.value_dim();
                    let head_v = hp.gdn_head_v();

                    let qkv_mixed = ffi::ggml_mul_mat(ctx, l.wqkv, cur); // [2*key+value, T]
                    let qkv_mixed = ffi::ggml_reshape_3d(ctx, qkv_mixed, hp.conv_dim(), t_len, 1);
                    let z = ffi::ggml_mul_mat(ctx, l.wqkv_gate, cur); // [value_dim, T]

                    let beta = ffi::ggml_mul_mat(ctx, l.ssm_beta, cur); // [H_v, T]
                    let beta = ffi::ggml_reshape_4d(ctx, beta, 1, hp.n_v_heads, t_len, 1);
                    let beta = ffi::ggml_sigmoid(ctx, beta);

                    let alpha = ffi::ggml_mul_mat(ctx, l.ssm_alpha, cur); // [H_v, T]
                    let alpha = ffi::ggml_reshape_3d(ctx, alpha, hp.n_v_heads, t_len, 1);
                    let alpha = ffi::ggml_add(ctx, alpha, as_f32(ctx, l.dt_bias));
                    let alpha = ffi::ggml_softplus(ctx, alpha);
                    let g = ffi::ggml_mul(ctx, alpha, as_f32(ctx, l.ssm_a)); // -A.exp() * softplus
                    let g = ffi::ggml_reshape_4d(ctx, g, 1, hp.n_v_heads, t_len, 1);

                    // causal conv over [q|k|v] with zero initial state
                    let qkv_t = ffi::ggml_transpose(ctx, qkv_mixed); // [T, conv_dim, 1]
                    let conv_input = ffi::ggml_concat(ctx, conv_zero, qkv_t, 0);
                    let conv_out = ffi::ggml_ssm_conv(ctx, conv_input, as_f32(ctx, l.conv1d)); // [conv_dim, T, 1]
                    let conv_out = ffi::ggml_silu(ctx, conv_out);

                    let nb1_qkv = ffi::ggml_row_size(ffi::ggml_type_GGML_TYPE_F32, hp.conv_dim());
                    let row = |n: i64| ffi::ggml_row_size(ffi::ggml_type_GGML_TYPE_F32, n);

                    let q = ffi::ggml_view_4d(
                        ctx, conv_out,
                        hp.d_state, hp.n_k_heads, t_len, 1,
                        row(hp.d_state), nb1_qkv, nb1_qkv * t_len as usize, 0,
                    );
                    let k = ffi::ggml_view_4d(
                        ctx, conv_out,
                        hp.d_state, hp.n_k_heads, t_len, 1,
                        row(hp.d_state), nb1_qkv, nb1_qkv * t_len as usize,
                        key_dim as usize * elt,
                    );
                    let v = ffi::ggml_view_4d(
                        ctx, conv_out,
                        head_v, hp.n_v_heads, t_len, 1,
                        row(head_v), nb1_qkv, nb1_qkv * t_len as usize,
                        row(2 * key_dim),
                    );

                    let q = ffi::ggml_l2_norm(ctx, q, hp.rms_eps);
                    let k = ffi::ggml_l2_norm(ctx, k, hp.rms_eps);
                    let v = ffi::ggml_cont(ctx, v);

                    // fused GDN, K=1 (final state only; discarded in stateless mode)
                    let gdn = ffi::ggml_gated_delta_net(ctx, q, k, v, g, beta, state_zero, 1);

                    let out = ffi::ggml_view_4d(
                        ctx, gdn,
                        head_v, hp.n_v_heads, t_len, 1,
                        row(head_v), row(head_v * hp.n_v_heads),
                        row(head_v * hp.n_v_heads * t_len), 0,
                    );

                    // gated rms norm with z, then out-projection
                    let z4 = ffi::ggml_reshape_4d(ctx, z, head_v, hp.n_v_heads, t_len, 1);
                    let normed = rms(ctx, out, l.ssm_norm);
                    let gated = ffi::ggml_mul(ctx, normed, ffi::ggml_silu(ctx, z4));
                    let flat = ffi::ggml_reshape_3d(ctx, ffi::ggml_cont(ctx, gated), value_dim, t_len, 1);
                    cur = ffi::ggml_mul_mat(ctx, l.ssm_out, flat); // [n_embd, T, 1]
                    cur = ffi::ggml_reshape_2d(ctx, cur, hp.n_embd, t_len);
                } else {
                    // ---- full attention (Q+gate packed in wq, IMROPE, GQA) ----
                    let hd = hp.head_k;
                    let q_full = ffi::ggml_mul_mat(ctx, l.wq, cur); // [hd*2*n_head, T]

                    let qcur = ffi::ggml_view_3d(
                        ctx, q_full, hd, hp.n_head, t_len,
                        elt * (hd * 2) as usize,
                        elt * (hd * 2 * hp.n_head) as usize,
                        0,
                    );
                    let qcur = rms(ctx, qcur, l.q_norm);

                    let gate = ffi::ggml_view_3d(
                        ctx, q_full, hd, hp.n_head, t_len,
                        elt * (hd * 2) as usize,
                        elt * (hd * 2 * hp.n_head) as usize,
                        elt * hd as usize,
                    );
                    let gate = ffi::ggml_cont_2d(ctx, gate, hd * hp.n_head, t_len);

                    let kcur = ffi::ggml_mul_mat(ctx, l.wk, cur); // [hd*n_head_kv, T]
                    let kcur = ffi::ggml_reshape_3d(ctx, kcur, hd, hp.n_head_kv, t_len);
                    let kcur = rms(ctx, kcur, l.k_norm);

                    let vcur = ffi::ggml_mul_mat(ctx, l.wv, cur);
                    let vcur = ffi::ggml_reshape_3d(ctx, vcur, hp.head_v, hp.n_head_kv, t_len);

                    let qcur = ffi::ggml_rope_multi(
                        ctx, qcur, inp_pos, std::ptr::null_mut(),
                        hp.n_rot, sections.as_mut_ptr(), ffi::GGML_ROPE_TYPE_IMROPE as i32,
                        hp.n_ctx_train, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0,
                    );
                    let kcur = ffi::ggml_rope_multi(
                        ctx, kcur, inp_pos, std::ptr::null_mut(),
                        hp.n_rot, sections.as_mut_ptr(), ffi::GGML_ROPE_TYPE_IMROPE as i32,
                        hp.n_ctx_train, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0,
                    );

                    // attention: kq = k·q, masked softmax, kqv = v_t·p
                    let q = ffi::ggml_permute(ctx, qcur, 0, 2, 1, 3); // [hd, T, n_head]
                    let k = ffi::ggml_permute(ctx, kcur, 0, 2, 1, 3); // [hd, T, n_head_kv]
                    let kq = ffi::ggml_mul_mat(ctx, k, q); // [T_kv, T, n_head]
                    let kq_scale = 1.0f32 / (hd as f32).sqrt();
                    let p = ffi::ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0);

                    let v = ffi::ggml_permute(ctx, vcur, 0, 2, 1, 3); // [hd, T, n_head_kv]
                    let v_t = ffi::ggml_cont(ctx, ffi::ggml_transpose(ctx, v)); // [T, hd, n_head_kv]
                    let kqv = ffi::ggml_mul_mat(ctx, v_t, p); // [hd, T, n_head]
                    let merged = ffi::ggml_permute(ctx, kqv, 0, 2, 1, 3); // [hd, n_head, T]
                    let merged = ffi::ggml_cont_2d(ctx, merged, hd * hp.n_head, t_len);

                    let gated = ffi::ggml_mul(ctx, merged, ffi::ggml_sigmoid(ctx, gate));
                    cur = ffi::ggml_mul_mat(ctx, l.wo, gated);
                }

                // residual + post-attn norm + swiglu ffn + residual
                cur = ffi::ggml_add(ctx, cur, inp_sa);
                let ffn_residual = cur;
                let normed = rms(ctx, cur, l.post_attn_norm);
                let up = ffi::ggml_mul_mat(ctx, l.ffn_up, normed);
                let gate = ffi::ggml_mul_mat(ctx, l.ffn_gate, normed);
                let act = ffi::ggml_mul(ctx, ffi::ggml_silu(ctx, gate), up);
                cur = ffi::ggml_mul_mat(ctx, l.ffn_down, act);
                cur = ffi::ggml_add(ctx, cur, ffn_residual);

                inp_l = cur;
            }

            // final norm → select last position → lm head
            let output_norm = self.t("output_norm.weight")?;
            let output_w = self
                .weights
                .tensor("output.weight")
                .unwrap_or(tok_embd); // tied embeddings fallback
            cur = rms(ctx, inp_l, output_norm);
            cur = ffi::ggml_get_rows(ctx, cur, out_ids);
            cur = ffi::ggml_mul_mat(ctx, output_w, cur); // [n_vocab, 1]
            ffi::ggml_set_output(cur);
            ffi::ggml_build_forward_expand(gf, cur);

            // ---- allocate + set inputs + compute ----
            let backend = self.weights.backend();
            let galloc = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(backend));
            guard.1 = galloc;
            if !ffi::ggml_gallocr_alloc_graph(galloc, gf) {
                return Err(ModelError::Load("graph alloc failed".into()));
            }

            let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            ffi::ggml_backend_tensor_set(inp_tokens, toks_i32.as_ptr().cast(), 0, t_len as usize * 4);

            // M-RoPE text positions: first 3 streams = position, 4th = 0.
            let mut pos = vec![0i32; tokens.len() * 4];
            for i in 0..tokens.len() {
                pos[i] = i as i32;
                pos[tokens.len() + i] = i as i32;
                pos[2 * tokens.len() + i] = i as i32;
            }
            ffi::ggml_backend_tensor_set(inp_pos, pos.as_ptr().cast(), 0, pos.len() * 4);

            // causal mask: [n_kv, n_q] with 0 where kv <= q else -inf
            let mut mask = vec![f32::NEG_INFINITY; tokens.len() * tokens.len()];
            for q in 0..tokens.len() {
                for kv in 0..=q {
                    mask[q * tokens.len() + kv] = 0.0;
                }
            }
            ffi::ggml_backend_tensor_set(kq_mask, mask.as_ptr().cast(), 0, mask.len() * 4);

            let zeros_conv = vec![0f32; ((hp.d_conv - 1) * hp.conv_dim()) as usize];
            ffi::ggml_backend_tensor_set(conv_zero, zeros_conv.as_ptr().cast(), 0, zeros_conv.len() * 4);
            let zeros_state =
                vec![0f32; (hp.gdn_head_v() * hp.gdn_head_v() * hp.n_v_heads) as usize];
            ffi::ggml_backend_tensor_set(state_zero, zeros_state.as_ptr().cast(), 0, zeros_state.len() * 4);

            let last = [(tokens.len() - 1) as i32];
            ffi::ggml_backend_tensor_set(out_ids, last.as_ptr().cast(), 0, 4);

            ffi::ggml_backend_cpu_set_n_threads(backend, n_threads);
            let st = ffi::ggml_backend_graph_compute(backend, gf);
            if st != ffi::ggml_status_GGML_STATUS_SUCCESS {
                return Err(ModelError::Load(format!("graph compute status {st}")));
            }

            let mut logits = vec![0f32; hp.n_vocab as usize];
            ffi::ggml_backend_tensor_get(cur, logits.as_mut_ptr().cast(), 0, logits.len() * 4);
            Ok(logits)
        }
    }

    /// Greedy generation, recomputing the full prefix each step (M1 rig).
    pub fn greedy(&self, prompt: &[u32], n_gen: usize, n_threads: i32) -> Result<Vec<u32>, ModelError> {
        let mut toks = prompt.to_vec();
        let mut out = Vec::with_capacity(n_gen);
        for _ in 0..n_gen {
            let logits = self.forward_logits(&toks, n_threads)?;
            let best = argmax(&logits);
            toks.push(best);
            out.push(best);
        }
        Ok(out)
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi as u32
}

/// Convenience: map with layer tensor names — used by tests/tools.
pub fn expected_layer_tensors(hp: &Hparams) -> HashMap<usize, &'static str> {
    let mut m = HashMap::new();
    for il in 0..hp.n_layer {
        m.insert(il, if hp.is_recurrent(il) { "gdn" } else { "attn" });
    }
    m
}
