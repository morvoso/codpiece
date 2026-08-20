//! MTP (multi-token prediction) draft head — `blk.<n_layer>` of the qwen35
//! GGUF, the `nextn.*` tensors.
//!
//! Port of llama.cpp's `llama_model_qwen35::graph_mtp`. The head is one full
//! attention block with its own KV cache, fronted by a projection that fuses
//! the *next* token's embedding with the trunk's hidden state:
//!
//! ```text
//!   concat( rms(embed(tok), enorm), rms(h_trunk, hnorm) )   [2*n_embd, T]
//!     -> eh_proj                                            [n_embd, T]
//!     -> attention block (own cache) -> FFN -> shared_head_norm -> LM head
//! ```
//!
//! `h_trunk` is the trunk's pre-LM-head hidden (`Built::h_out`), so a draft
//! costs one small block instead of a whole 64-layer pass. Prod measures 0.78
//! to 0.92 acceptance at draft depth 3, which is where the ~2.5x comes from.
//!
//! The 27B carries neither `nextn.embed_tokens` nor `nextn.shared_head_head`,
//! so both fall back to the trunk's own tensors, exactly as llama.cpp does.

use tandem_ggml_sys as ffi;

use crate::qwen35::{Layer, Qwen35};
use crate::ModelError;

/// A built MTP graph plus its input handles.
pub struct MtpGraph {
    pub ctx: *mut ffi::ggml_context,
    pub gf: *mut ffi::ggml_cgraph,
    pub inp_tokens: *mut ffi::ggml_tensor,
    pub inp_h: *mut ffi::ggml_tensor,
    pub inp_pos: *mut ffi::ggml_tensor,
    pub kq_mask: *mut ffi::ggml_tensor,
    pub out_ids: *mut ffi::ggml_tensor,
    pub out: *mut ffi::ggml_tensor,
    pub n_kv: i64,
    pub t_len: i64,
    pub fa_mask: bool,
}

impl Drop for MtpGraph {
    fn drop(&mut self) {
        unsafe { ffi::ggml_free(self.ctx) }
    }
}

impl Qwen35 {
    /// Tensors of the MTP block. Returns None when the file has no MTP head.
    pub(crate) fn mtp_layer(&self) -> Option<(Layer, MtpExtras)> {
        let il = self.hp.n_layer;
        let l = self.layer_pub(il).ok()?;
        let g = |n: &str| self.weights.tensor(&format!("blk.{il}.{n}"));
        Some((
            l,
            MtpExtras {
                eh_proj: g("nextn.eh_proj.weight")?,
                enorm: g("nextn.enorm.weight")?,
                hnorm: g("nextn.hnorm.weight")?,
                embed_tokens: g("nextn.embed_tokens.weight"),
                shared_head_norm: g("nextn.shared_head_norm.weight"),
                shared_head_head: g("nextn.shared_head_head.weight"),
            },
        ))
    }
}

pub(crate) struct MtpExtras {
    pub eh_proj: *mut ffi::ggml_tensor,
    pub enorm: *mut ffi::ggml_tensor,
    pub hnorm: *mut ffi::ggml_tensor,
    pub embed_tokens: Option<*mut ffi::ggml_tensor>,
    pub shared_head_norm: Option<*mut ffi::ggml_tensor>,
    pub shared_head_head: Option<*mut ffi::ggml_tensor>,
}

impl Qwen35 {
    /// Build the MTP draft graph for `t_len` tokens against a KV window of
    /// `n_kv`, writing into the session's MTP cache at `n_past`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn build_mtp(
        &self,
        t_len: i64,
        n_kv: i64,
        n_past: usize,
        mtp_k: *mut ffi::ggml_tensor,
        mtp_v: *mut ffi::ggml_tensor,
        n_ctx_max: usize,
        fa: bool,
    ) -> Result<MtpGraph, ModelError> {
        let hp = &self.hp;
        let (l, x) = self
            .mtp_layer()
            .ok_or_else(|| ModelError::Load("model has no MTP head".into()))?;

        let params = ffi::ggml_init_params {
            mem_size: 16 << 20,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return Err(ModelError::Load("mtp ctx init".into()));
        }
        let gf = ffi::ggml_new_graph_custom(ctx, 512, false);
        let f32t = ffi::ggml_type_GGML_TYPE_F32;

        let inp_tokens = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len);
        ffi::ggml_set_input(inp_tokens);
        let inp_h = ffi::ggml_new_tensor_2d(ctx, f32t, hp.n_embd, t_len);
        ffi::ggml_set_input(inp_h);
        let inp_pos = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len * 4);
        ffi::ggml_set_input(inp_pos);
        let mask_t = if fa { ffi::ggml_type_GGML_TYPE_F16 } else { f32t };
        let kq_mask = ffi::ggml_new_tensor_2d(ctx, mask_t, n_kv, t_len);
        ffi::ggml_set_input(kq_mask);
        let out_ids = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, 1);
        ffi::ggml_set_input(out_ids);

        let as_f32 = |t: *mut ffi::ggml_tensor| {
            if (*t).type_ == f32t {
                t
            } else {
                ffi::ggml_cast(ctx, t, f32t)
            }
        };
        let rms = |x: *mut ffi::ggml_tensor, w: *mut ffi::ggml_tensor| {
            let n = ffi::ggml_rms_norm(ctx, x, hp.rms_eps);
            ffi::ggml_mul(ctx, n, as_f32(w))
        };

        // token embedding (the MTP head may carry its own table; the 27B does not)
        let embd_w = x.embed_tokens.unwrap_or(self.t_pub("token_embd.weight")?);
        let tok_embd = ffi::ggml_get_rows(ctx, embd_w, inp_tokens);

        let e_norm = rms(tok_embd, x.enorm);
        let h_norm = rms(inp_h, x.hnorm);
        let concat = ffi::ggml_concat(ctx, e_norm, h_norm, 0);
        let mut cur = ffi::ggml_mul_mat(ctx, x.eh_proj, concat);

        let inp_sa = cur;
        cur = rms(cur, l.attn_norm);
        cur = self.build_attn_block(
            ctx,
            gf,
            cur,
            &l,
            inp_pos,
            kq_mask,
            std::ptr::null_mut(),
            Some((mtp_k, mtp_v, n_ctx_max)),
            t_len,
            n_kv,
            n_past,
            fa,
            fa,
            false,
            Default::default(),
        );
        cur = ffi::ggml_add(ctx, cur, inp_sa);

        let ffn_residual = cur;
        let normed = rms(cur, l.post_attn_norm);
        let up = ffi::ggml_mul_mat(ctx, l.ffn_up, normed);
        let gate = ffi::ggml_mul_mat(ctx, l.ffn_gate, normed);
        let act = ffi::ggml_mul(ctx, ffi::ggml_silu(ctx, gate), up);
        cur = ffi::ggml_mul_mat(ctx, l.ffn_down, act);
        cur = ffi::ggml_add(ctx, cur, ffn_residual);

        let head_norm = x
            .shared_head_norm
            .unwrap_or(self.t_pub("output_norm.weight")?);
        cur = rms(cur, head_norm);
        cur = ffi::ggml_get_rows(ctx, cur, out_ids);
        let head_w = match x.shared_head_head {
            Some(w) => w,
            None => self
                .weights
                .tensor("output.weight")
                .unwrap_or(self.t_pub("token_embd.weight")?),
        };
        cur = ffi::ggml_mul_mat(ctx, head_w, cur);
        ffi::ggml_set_output(cur);
        ffi::ggml_build_forward_expand(gf, cur);

        Ok(MtpGraph {
            ctx,
            gf,
            inp_tokens,
            inp_h,
            inp_pos,
            kq_mask,
            out_ids,
            out: cur,
            n_kv,
            t_len,
            fa_mask: fa,
        })
    }
}
