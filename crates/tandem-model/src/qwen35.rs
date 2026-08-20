//! qwen35 forward graph (trunk only: GDN + full-attention layers).
//!
//! Faithful port of llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp`
//! (snapshots in notes/reference/). One graph builder serves two modes:
//!
//! - **Stateless** (M1 reference rig): whole prefix recomputed, zero initial
//!   recurrent states, no KV cache. O(n²), exists to be diffed against
//!   llama.cpp. All M1 gates passed on this path.
//! - **Session** (the engine path): per-layer KV caches (attention layers,
//!   llama.cpp 2-D layout with transposed V), carried conv + GDN states with
//!   in-graph write-back. Prefill once, then O(1)-per-token decode.
//!
//! Write-after-read ordering inside a step relies on ggml executing nodes in
//! build_forward_expand insertion order: cache/state writes are expanded
//! before downstream readers are inserted, and every state write's source
//! subtree contains the state read, so dependencies force the safe order —
//! the same contract llama.cpp's KV cache uses.

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

/// Layer tensor handles, resolved per graph build.
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

/// Persistent per-sequence state: KV caches for attention layers, conv + GDN
/// recurrent states for GDN layers. Lives in its own backend buffer, distinct
/// from the per-step compute buffer, so graph allocation can never reuse it.
pub struct Session {
    pub n_ctx_max: usize,
    pub n_past: usize,
    ctx: *mut ffi::ggml_context,
    buffer: ffi::ggml_backend_buffer_t,
    /// attn layers: k [hd*nhkv, n_ctx_max], v [n_ctx_max, hv*nhkv] (transposed)
    k_cache: Vec<*mut ffi::ggml_tensor>,
    v_cache: Vec<*mut ffi::ggml_tensor>,
    /// gdn layers: conv [d_conv-1, conv_dim], gdn [S, S, H_v]
    conv_state: Vec<*mut ffi::ggml_tensor>,
    gdn_state: Vec<*mut ffi::ggml_tensor>,
}

impl Session {
    pub fn new(model: &Qwen35, n_ctx_max: usize) -> Result<Session, ModelError> {
        let hp = &model.hp;
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: (hp.n_layer * 4 + 8) * ffi::ggml_tensor_overhead(),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(ModelError::Load("session ctx init".into()));
            }
            let f32t = ffi::ggml_type_GGML_TYPE_F32;
            let mut k_cache = vec![std::ptr::null_mut(); hp.n_layer];
            let mut v_cache = vec![std::ptr::null_mut(); hp.n_layer];
            let mut conv_state = vec![std::ptr::null_mut(); hp.n_layer];
            let mut gdn_state = vec![std::ptr::null_mut(); hp.n_layer];
            for il in 0..hp.n_layer {
                if hp.is_recurrent(il) {
                    conv_state[il] =
                        ffi::ggml_new_tensor_2d(ctx, f32t, hp.d_conv - 1, hp.conv_dim());
                    gdn_state[il] = ffi::ggml_new_tensor_3d(
                        ctx, f32t, hp.gdn_head_v(), hp.gdn_head_v(), hp.n_v_heads,
                    );
                } else {
                    k_cache[il] = ffi::ggml_new_tensor_2d(
                        ctx, f32t, hp.head_k * hp.n_head_kv, n_ctx_max as i64,
                    );
                    v_cache[il] = ffi::ggml_new_tensor_2d(
                        ctx, f32t, n_ctx_max as i64, hp.head_v * hp.n_head_kv,
                    );
                }
            }
            let buffer = ffi::ggml_backend_alloc_ctx_tensors(ctx, model.weights.backend());
            if buffer.is_null() {
                ffi::ggml_free(ctx);
                return Err(ModelError::Load("session buffer alloc".into()));
            }
            ffi::ggml_backend_buffer_clear(buffer, 0);
            Ok(Session {
                n_ctx_max,
                n_past: 0,
                ctx,
                buffer,
                k_cache,
                v_cache,
                conv_state,
                gdn_state,
            })
        }
    }

    pub fn reset(&mut self) {
        unsafe {
            ffi::ggml_backend_buffer_clear(self.buffer, 0);
        }
        self.n_past = 0;
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            ffi::ggml_backend_buffer_free(self.buffer);
            ffi::ggml_free(self.ctx);
        }
    }
}

enum StateSrc<'a> {
    /// Zero states, no KV cache, positions start at 0 (reference rig).
    Stateless,
    /// Session caches/states, positions start at session.n_past.
    Session(&'a Session),
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

    /// Stateless forward over `tokens`; logits for the LAST position.
    pub fn forward_logits(&self, tokens: &[u32], n_threads: i32) -> Result<Vec<f32>, ModelError> {
        self.forward(tokens, &[(tokens.len() - 1) as i32], n_threads)
    }

    /// Stateless forward; logits at `out_positions` (ubatch-relative),
    /// row-major [n_out][n_vocab].
    pub fn forward(
        &self,
        tokens: &[u32],
        out_positions: &[i32],
        n_threads: i32,
    ) -> Result<Vec<f32>, ModelError> {
        self.forward_impl(tokens, StateSrc::Stateless, out_positions, n_threads)
    }

    /// Stateful step: consume `tokens` at positions [session.n_past, ..),
    /// update caches/states, return logits at `out_positions` (ubatch-
    /// relative). Advances session.n_past on success.
    pub fn step(
        &self,
        session: &mut Session,
        tokens: &[u32],
        out_positions: &[i32],
        n_threads: i32,
    ) -> Result<Vec<f32>, ModelError> {
        if session.n_past + tokens.len() > session.n_ctx_max {
            return Err(ModelError::Load(format!(
                "context overflow: {} + {} > {}",
                session.n_past,
                tokens.len(),
                session.n_ctx_max
            )));
        }
        let out =
            self.forward_impl(tokens, StateSrc::Session(session), out_positions, n_threads)?;
        session.n_past += tokens.len();
        Ok(out)
    }

    fn forward_impl(
        &self,
        tokens: &[u32],
        state: StateSrc<'_>,
        out_positions: &[i32],
        n_threads: i32,
    ) -> Result<Vec<f32>, ModelError> {
        assert!(!tokens.is_empty());
        assert!(!out_positions.is_empty());
        let hp = &self.hp;
        let t_len = tokens.len() as i64;
        let n_past = match &state {
            StateSrc::Stateless => 0usize,
            StateSrc::Session(s) => s.n_past,
        };
        let n_kv = (n_past + tokens.len()) as i64;

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
            let f32t = ffi::ggml_type_GGML_TYPE_F32;

            // ---- inputs ----
            let inp_tokens = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len);
            ffi::ggml_set_input(inp_tokens);
            let inp_pos = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len * 4);
            ffi::ggml_set_input(inp_pos);
            let kq_mask = ffi::ggml_new_tensor_2d(ctx, f32t, n_kv, t_len);
            ffi::ggml_set_input(kq_mask);
            let (conv_zero, state_zero) = match &state {
                StateSrc::Stateless => {
                    let c = ffi::ggml_new_tensor_3d(ctx, f32t, hp.d_conv - 1, hp.conv_dim(), 1);
                    ffi::ggml_set_input(c);
                    let s = ffi::ggml_new_tensor_4d(
                        ctx, f32t, hp.gdn_head_v(), hp.gdn_head_v(), hp.n_v_heads, 1,
                    );
                    ffi::ggml_set_input(s);
                    (c, s)
                }
                StateSrc::Session(_) => (std::ptr::null_mut(), std::ptr::null_mut()),
            };
            let out_ids = ffi::ggml_new_tensor_1d(
                ctx,
                ffi::ggml_type_GGML_TYPE_I32,
                out_positions.len() as i64,
            );
            ffi::ggml_set_input(out_ids);

            let as_f32 = |ctx: *mut ffi::ggml_context, t: *mut ffi::ggml_tensor| {
                if (*t).type_ == f32t {
                    t
                } else {
                    ffi::ggml_cast(ctx, t, f32t)
                }
            };
            let rms = |ctx: *mut ffi::ggml_context,
                       x: *mut ffi::ggml_tensor,
                       w: *mut ffi::ggml_tensor| {
                let n = ffi::ggml_rms_norm(ctx, x, hp.rms_eps);
                ffi::ggml_mul(ctx, n, as_f32(ctx, w))
            };

            // ---- trunk ----
            let tok_embd = self.t("token_embd.weight")?;
            let mut cur;
            let mut inp_l = ffi::ggml_get_rows(ctx, tok_embd, inp_tokens); // [n_embd, T]

            let mut sections = hp.rope_sections;
            let elt = ffi::ggml_type_size(f32t);
            let row = |n: i64| ffi::ggml_row_size(f32t, n);

            for il in 0..hp.n_layer {
                let l = self.layer(il)?;
                let inp_sa = inp_l;

                cur = rms(ctx, inp_l, l.attn_norm);

                if hp.is_recurrent(il) {
                    // ---- gated delta net ----
                    let key_dim = hp.key_dim();
                    let value_dim = hp.value_dim();
                    let head_v = hp.gdn_head_v();

                    let qkv_mixed = ffi::ggml_mul_mat(ctx, l.wqkv, cur); // [conv_dim, T]
                    let qkv_mixed = ffi::ggml_reshape_3d(ctx, qkv_mixed, hp.conv_dim(), t_len, 1);
                    let z = ffi::ggml_mul_mat(ctx, l.wqkv_gate, cur); // [value_dim, T]

                    let beta = ffi::ggml_mul_mat(ctx, l.ssm_beta, cur); // [H_v, T]
                    let beta = ffi::ggml_reshape_4d(ctx, beta, 1, hp.n_v_heads, t_len, 1);
                    let beta = ffi::ggml_sigmoid(ctx, beta);

                    let alpha = ffi::ggml_mul_mat(ctx, l.ssm_alpha, cur); // [H_v, T]
                    let alpha = ffi::ggml_reshape_3d(ctx, alpha, hp.n_v_heads, t_len, 1);
                    let alpha = ffi::ggml_add(ctx, alpha, as_f32(ctx, l.dt_bias));
                    let alpha = ffi::ggml_softplus(ctx, alpha);
                    let g = ffi::ggml_mul(ctx, alpha, as_f32(ctx, l.ssm_a));
                    let g = ffi::ggml_reshape_4d(ctx, g, 1, hp.n_v_heads, t_len, 1);

                    // conv over [q|k|v] with carried (or zero) state
                    let (conv_in_state, gdn_in_state) = match &state {
                        StateSrc::Stateless => (conv_zero, state_zero),
                        StateSrc::Session(s) => {
                            let c3 = ffi::ggml_reshape_3d(
                                ctx, s.conv_state[il], hp.d_conv - 1, hp.conv_dim(), 1,
                            );
                            let s4 = ffi::ggml_reshape_4d(
                                ctx, s.gdn_state[il], head_v, head_v, hp.n_v_heads, 1,
                            );
                            (c3, s4)
                        }
                    };

                    let qkv_t = ffi::ggml_transpose(ctx, qkv_mixed); // [T, conv_dim, 1]
                    let conv_input = ffi::ggml_concat(ctx, conv_in_state, qkv_t, 0);

                    // write updated conv state (last d_conv-1 columns) back
                    if let StateSrc::Session(s) = &state {
                        let tail = ffi::ggml_view_3d(
                            ctx, conv_input,
                            hp.d_conv - 1, hp.conv_dim(), 1,
                            (*conv_input).nb[1], (*conv_input).nb[2],
                            row(t_len),
                        );
                        let dst = ffi::ggml_reshape_3d(
                            ctx, s.conv_state[il], hp.d_conv - 1, hp.conv_dim(), 1,
                        );
                        ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, tail, dst));
                    }

                    let conv_out = ffi::ggml_ssm_conv(ctx, conv_input, as_f32(ctx, l.conv1d));
                    let conv_out = ffi::ggml_silu(ctx, conv_out);

                    let nb1_qkv = row(hp.conv_dim());
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

                    let gdn =
                        ffi::ggml_gated_delta_net(ctx, q, k, v, g, beta, gdn_in_state, 1);

                    let out = ffi::ggml_view_4d(
                        ctx, gdn,
                        head_v, hp.n_v_heads, t_len, 1,
                        row(head_v), row(head_v * hp.n_v_heads),
                        row(head_v * hp.n_v_heads * t_len), 0,
                    );

                    // write final GDN state back (snapshot slot 0)
                    if let StateSrc::Session(s) = &state {
                        let new_state = ffi::ggml_view_4d(
                            ctx, gdn,
                            head_v, head_v, hp.n_v_heads, 1,
                            row(head_v), row(head_v * head_v),
                            row(head_v * head_v * hp.n_v_heads),
                            row(head_v * hp.n_v_heads * t_len),
                        );
                        let dst = ffi::ggml_reshape_4d(
                            ctx, s.gdn_state[il], head_v, head_v, hp.n_v_heads, 1,
                        );
                        ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, new_state, dst));
                    }

                    let z4 = ffi::ggml_reshape_4d(ctx, z, head_v, hp.n_v_heads, t_len, 1);
                    let normed = rms(ctx, out, l.ssm_norm);
                    let gated = ffi::ggml_mul(ctx, normed, ffi::ggml_silu(ctx, z4));
                    let flat =
                        ffi::ggml_reshape_3d(ctx, ffi::ggml_cont(ctx, gated), value_dim, t_len, 1);
                    cur = ffi::ggml_mul_mat(ctx, l.ssm_out, flat);
                    cur = ffi::ggml_reshape_2d(ctx, cur, hp.n_embd, t_len);
                } else {
                    // ---- full attention (packed Q+gate, IMROPE, GQA) ----
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

                    let kcur = ffi::ggml_mul_mat(ctx, l.wk, cur); // [hd*nhkv, T]
                    let kcur = ffi::ggml_reshape_3d(ctx, kcur, hd, hp.n_head_kv, t_len);
                    let kcur = rms(ctx, kcur, l.k_norm);

                    let vcur = ffi::ggml_mul_mat(ctx, l.wv, cur); // [hv*nhkv, T]

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

                    let (k_all, v_t_all) = match &state {
                        StateSrc::Stateless => {
                            let k = ffi::ggml_permute(ctx, kcur, 0, 2, 1, 3); // [hd, T, nhkv]
                            let v3 =
                                ffi::ggml_reshape_3d(ctx, vcur, hp.head_v, hp.n_head_kv, t_len);
                            let v = ffi::ggml_permute(ctx, v3, 0, 2, 1, 3); // [hv, T, nhkv]
                            let v_t = ffi::ggml_cont(ctx, ffi::ggml_transpose(ctx, v));
                            (k, v_t)
                        }
                        StateSrc::Session(s) => {
                            let kc = s.k_cache[il];
                            let vc = s.v_cache[il];

                            // write new K columns (post-rope, post-norm) at n_past
                            let k2 = ffi::ggml_reshape_2d(
                                ctx,
                                ffi::ggml_cont(ctx, kcur),
                                hd * hp.n_head_kv,
                                t_len,
                            );
                            let k_dst = ffi::ggml_view_2d(
                                ctx, kc, hd * hp.n_head_kv, t_len,
                                (*kc).nb[1], n_past * (*kc).nb[1],
                            );
                            ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, k2, k_dst));

                            // write new V rows transposed at column offset n_past
                            let v_t_new = ffi::ggml_transpose(ctx, vcur); // [T, hv*nhkv]
                            let v_dst = ffi::ggml_view_2d(
                                ctx, vc, t_len, hp.head_v * hp.n_head_kv,
                                (*vc).nb[1], n_past * elt,
                            );
                            ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, v_t_new, v_dst));

                            // read views over [0, n_kv)
                            let k_all = ffi::ggml_view_3d(
                                ctx, kc, hd, n_kv, hp.n_head_kv,
                                (*kc).nb[1], hd as usize * elt, 0,
                            );
                            let v_t_all = ffi::ggml_view_3d(
                                ctx, vc, n_kv, hp.head_v, hp.n_head_kv,
                                (*vc).nb[1],
                                s.n_ctx_max * hp.head_v as usize * elt,
                                0,
                            );
                            (k_all, v_t_all)
                        }
                    };

                    let q = ffi::ggml_permute(ctx, qcur, 0, 2, 1, 3); // [hd, T, n_head]
                    let kq = ffi::ggml_mul_mat(ctx, k_all, q); // [n_kv, T, n_head]
                    let kq_scale = 1.0f32 / (hd as f32).sqrt();
                    let p = ffi::ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0);
                    let kqv = ffi::ggml_mul_mat(ctx, v_t_all, p); // [hv, T, n_head]
                    let merged = ffi::ggml_permute(ctx, kqv, 0, 2, 1, 3); // [hv, n_head, T]
                    let merged = ffi::ggml_cont_2d(ctx, merged, hp.head_v * hp.n_head, t_len);

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

            // final norm → select requested positions → lm head
            let output_norm = self.t("output_norm.weight")?;
            let output_w = self.weights.tensor("output.weight").unwrap_or(tok_embd);
            cur = rms(ctx, inp_l, output_norm);
            cur = ffi::ggml_get_rows(ctx, cur, out_ids);
            cur = ffi::ggml_mul_mat(ctx, output_w, cur); // [n_vocab, n_out]
            ffi::ggml_set_output(cur);
            ffi::ggml_build_forward_expand(gf, cur);

            // ---- allocate + set inputs + compute ----
            let backend = self.weights.backend();
            let galloc =
                ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(backend));
            guard.1 = galloc;
            if !ffi::ggml_gallocr_alloc_graph(galloc, gf) {
                return Err(ModelError::Load("graph alloc failed".into()));
            }

            let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            ffi::ggml_backend_tensor_set(
                inp_tokens,
                toks_i32.as_ptr().cast(),
                0,
                t_len as usize * 4,
            );

            // M-RoPE text positions: 3 streams = absolute position, 4th = 0
            let mut pos = vec![0i32; tokens.len() * 4];
            for i in 0..tokens.len() {
                let p = (n_past + i) as i32;
                pos[i] = p;
                pos[tokens.len() + i] = p;
                pos[2 * tokens.len() + i] = p;
            }
            ffi::ggml_backend_tensor_set(inp_pos, pos.as_ptr().cast(), 0, pos.len() * 4);

            // causal mask over cache: query at n_past+q sees kv <= n_past+q
            let nkv = n_kv as usize;
            let mut mask = vec![f32::NEG_INFINITY; nkv * tokens.len()];
            for q in 0..tokens.len() {
                for kv in 0..=(n_past + q) {
                    mask[q * nkv + kv] = 0.0;
                }
            }
            ffi::ggml_backend_tensor_set(kq_mask, mask.as_ptr().cast(), 0, mask.len() * 4);

            if let StateSrc::Stateless = &state {
                let zc = vec![0f32; ((hp.d_conv - 1) * hp.conv_dim()) as usize];
                ffi::ggml_backend_tensor_set(conv_zero, zc.as_ptr().cast(), 0, zc.len() * 4);
                let zs =
                    vec![0f32; (hp.gdn_head_v() * hp.gdn_head_v() * hp.n_v_heads) as usize];
                ffi::ggml_backend_tensor_set(state_zero, zs.as_ptr().cast(), 0, zs.len() * 4);
            }

            ffi::ggml_backend_tensor_set(
                out_ids,
                out_positions.as_ptr().cast(),
                0,
                out_positions.len() * 4,
            );

            ffi::ggml_backend_cpu_set_n_threads(backend, n_threads);
            let st = ffi::ggml_backend_graph_compute(backend, gf);
            if st != ffi::ggml_status_GGML_STATUS_SUCCESS {
                return Err(ModelError::Load(format!("graph compute status {st}")));
            }

            let mut logits = vec![0f32; hp.n_vocab as usize * out_positions.len()];
            ffi::ggml_backend_tensor_get(cur, logits.as_mut_ptr().cast(), 0, logits.len() * 4);
            Ok(logits)
        }
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
