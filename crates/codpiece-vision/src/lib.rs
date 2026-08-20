//! Qwen3-VL vision encoder (ViT + spatial-merge projector) for the 27B's
//! mmproj GGUF, ported op-for-op from llama.cpp b10423
//! `tools/mtmd/models/qwen3vl.cpp` (projector type `qwen3vl_merger`).
//!
//! The encoder maps a preprocessed image to one embedding per 2x2 patch
//! block in the trunk's hidden size, ready to be injected in place of image
//! placeholder tokens. It runs on a single backend (CPU or one CUDA device)
//! — the ViT is ~0.9 GiB in BF16 and never tensor-parallel; the trunk's
//! meta-backend split rules do not apply here.
//!
//! Parity contract: every graph op below mirrors the reference builder,
//! including the 2x2 block reorder applied to BOTH the patch embeddings and
//! the resized position embeddings, and the M-RoPE position layout
//! ([y, x, y, x] per patch, filled in 2x2 block order). The reference's
//! deepstack branch is intentionally absent: this mmproj declares
//! `clip.vision.is_deepstack_layers = [false; 27]`, and `load` rejects any
//! file where that does not hold rather than silently dropping features.

use std::path::Path;

use codpiece_ggml_sys as ffi;
use codpiece_model::{Device, ModelError, Weights};

pub mod preprocess;

/// Vision hparams, read from mmproj GGUF metadata (`clip.vision.*`).
#[derive(Debug, Clone)]
pub struct VisionHparams {
    pub n_layer: usize,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_ff: i64,
    pub patch: i64,
    /// spatial merge factor per side (2 => 2x2 blocks -> one output embedding)
    pub merge: i64,
    pub eps: f32,
    /// side length the learned position grid was trained at, in patches
    pub pos_side: i64,
    /// trunk hidden size the projector emits
    pub proj_dim: i64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl VisionHparams {
    pub fn d_head(&self) -> i64 {
        self.n_embd / self.n_head
    }

    fn from_gguf(gguf: &codpiece_gguf::GgufFile) -> Result<VisionHparams, ModelError> {
        let u = |k: &str| -> Result<i64, ModelError> {
            gguf.kv(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as i64)
                .ok_or_else(|| ModelError::Load(format!("mmproj missing {k}")))
        };
        let f3 = |k: &str| -> Result<[f32; 3], ModelError> {
            let a = gguf
                .kv(k)
                .and_then(|v| v.as_array())
                .ok_or_else(|| ModelError::Load(format!("mmproj missing {k}")))?;
            let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
            v.try_into()
                .map_err(|_| ModelError::Load(format!("mmproj {k} is not 3 floats")))
        };
        let proj = gguf.kv("clip.projector_type").and_then(|v| v.as_str());
        if proj != Some("qwen3vl_merger") {
            return Err(ModelError::Load(format!(
                "mmproj projector {proj:?}, expected qwen3vl_merger"
            )));
        }
        if let Some(ds) = gguf.kv("clip.vision.is_deepstack_layers").and_then(|v| v.as_array()) {
            if ds.iter().any(|v| v.as_bool() == Some(true)) {
                return Err(ModelError::Load(
                    "mmproj has deepstack layers; this port does not build that branch".into(),
                ));
            }
        }
        let image_size = u("clip.vision.image_size")?;
        let patch = u("clip.vision.patch_size")?;
        Ok(VisionHparams {
            n_layer: u("clip.vision.block_count")? as usize,
            n_embd: u("clip.vision.embedding_length")?,
            n_head: u("clip.vision.attention.head_count")?,
            n_ff: u("clip.vision.feed_forward_length")?,
            patch,
            merge: u("clip.vision.spatial_merge_size")?,
            eps: gguf
                .kv("clip.vision.attention.layer_norm_epsilon")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            pos_side: image_size / patch,
            proj_dim: u("clip.vision.projection_dim")?,
            image_mean: f3("clip.vision.image_mean")?,
            image_std: f3("clip.vision.image_std")?,
        })
    }
}

pub struct VisionModel {
    pub weights: Weights,
    pub hp: VisionHparams,
}

// Raw ggml pointers keep this !Send by default; like the trunk, the encoder
// is owned by one engine thread.
unsafe impl Send for VisionModel {}

impl VisionModel {
    pub fn load(path: &Path, device: Device) -> Result<VisionModel, ModelError> {
        if matches!(device, Device::CudaTensorParallel(_) | Device::CudaSplit(_)) {
            return Err(ModelError::Load(
                "vision encoder runs on a single device (Cpu or Cuda(i))".into(),
            ));
        }
        let weights = Weights::load(path, device)?;
        let hp = VisionHparams::from_gguf(&weights.gguf)?;
        if hp.merge != 2 {
            // the block reorder (and the reference graph it mirrors) hardcodes
            // 2x2 blocks; Qwen publishes no other merge factor
            return Err(ModelError::Load(format!(
                "spatial merge {} unsupported (only 2)",
                hp.merge
            )));
        }
        // Fail at load, not mid-graph, if the file is missing anything we use.
        for name in Self::required(hp.n_layer) {
            if weights.tensor(&name).is_none() {
                return Err(ModelError::Load(format!("mmproj missing tensor {name}")));
            }
        }
        Ok(VisionModel { weights, hp })
    }

    fn required(n_layer: usize) -> Vec<String> {
        let mut v = vec![
            "v.patch_embd.weight".into(),
            "v.patch_embd.weight.1".into(),
            "v.patch_embd.bias".into(),
            "v.position_embd.weight".into(),
            "v.post_ln.weight".into(),
            "v.post_ln.bias".into(),
            "mm.0.weight".into(),
            "mm.0.bias".into(),
            "mm.2.weight".into(),
            "mm.2.bias".into(),
        ];
        for il in 0..n_layer {
            for t in ["attn_qkv", "attn_out", "ffn_up", "ffn_down", "ln1", "ln2"] {
                v.push(format!("v.blk.{il}.{t}.weight"));
                v.push(format!("v.blk.{il}.{t}.bias"));
            }
        }
        v
    }

    /// Number of output embeddings for an image of `w` x `h` pixels.
    pub fn n_out(&self, w: usize, h: usize) -> usize {
        let m = (self.hp.patch * self.hp.merge) as usize;
        (w / m) * (h / m)
    }

    /// Encode one preprocessed image into `[n_out, proj_dim]` trunk embeddings
    /// (row-major, one row per merged 2x2 patch block, in 2x2 block order —
    /// the same order the trunk's vision M-RoPE positions expect).
    ///
    /// `img` is planar f32, channel-major: `img[c*w*h + y*w + x]`, already
    /// resized so `w` and `h` are multiples of `patch*merge` (32), and
    /// normalized with `image_mean`/`image_std`.
    pub fn encode(&self, img: &[f32], w: usize, h: usize) -> Result<Vec<f32>, ModelError> {
        self.encode_t(img, w, h, 0)
    }

    /// `encode` with an explicit CPU thread count (0 = backend default).
    /// Thread count changes float summation order, so CPU parity checks
    /// against llama.cpp must match its `-t` to compare beyond ~1e-3.
    pub fn encode_t(
        &self,
        img: &[f32],
        w: usize,
        h: usize,
        n_threads: i32,
    ) -> Result<Vec<f32>, ModelError> {
        let hp = &self.hp;
        let m = (hp.patch * hp.merge) as usize;
        if w == 0 || h == 0 || w % m != 0 || h % m != 0 {
            return Err(ModelError::Load(format!(
                "image {w}x{h} is not a multiple of {m}"
            )));
        }
        if img.len() != w * h * 3 {
            return Err(ModelError::Load(format!(
                "image buffer {} floats, expected {}",
                img.len(),
                w * h * 3
            )));
        }
        let n_px = (w as i64) / hp.patch;
        let n_py = (h as i64) / hp.patch;
        let n_pos = n_px * n_py;
        let n_embd = hp.n_embd;
        let d_head = hp.d_head();
        let n_head = hp.n_head;
        let kq_scale = 1.0f32 / (d_head as f32).sqrt();
        let f32t = ffi::ggml_type_GGML_TYPE_F32;
        let i32t = ffi::ggml_type_GGML_TYPE_I32;
        // FA on CUDA (llama.cpp's clip default there — the memory difference
        // decides whether big images fit at all); the exact non-FA path on
        // CPU, where the parity harness runs. CODPIECE_VISION_NO_FA=1 forces
        // the reference path everywhere.
        let use_fa = !self.weights.is_cpu() && std::env::var("CODPIECE_VISION_NO_FA").is_err();

        let wt = |name: &str| self.weights.tensor(name).expect("checked at load");
        let blk = |il: usize, t: &str, wb: &str| {
            self.weights
                .tensor(&format!("v.blk.{il}.{t}.{wb}"))
                .expect("checked at load")
        };

        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: ffi::ggml_tensor_overhead() * 8192
                    + ffi::ggml_graph_overhead_custom(8192, false),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(ModelError::Load("vision ggml_init".into()));
            }
            // Everything below either returns through `finish` or aborts the
            // process (ggml asserts); no early Rust returns that leak ctx.
            let gf = ffi::ggml_new_graph_custom(ctx, 8192, false);

            let inp_raw = ffi::ggml_new_tensor_4d(ctx, f32t, w as i64, h as i64, 3, 1);
            ffi::ggml_set_input(inp_raw);

            // patch embedding: two conv2d over the same frame (the reference's
            // temporal merge with a still image duplicates the frame)
            let mut inp = ffi::ggml_add(
                ctx,
                ffi::ggml_conv_2d(
                    ctx,
                    wt("v.patch_embd.weight"),
                    inp_raw,
                    hp.patch as i32,
                    hp.patch as i32,
                    0,
                    0,
                    1,
                    1,
                ),
                ffi::ggml_conv_2d(
                    ctx,
                    wt("v.patch_embd.weight.1"),
                    inp_raw,
                    hp.patch as i32,
                    hp.patch as i32,
                    0,
                    0,
                    1,
                    1,
                ),
            );

            // 2x2 block reorder: rows become [b00, b01, b10, b11] per block so
            // the merger's reshape to n_embd*4 sees one spatial block per row
            let reorder = |ctx: *mut ffi::ggml_context, t: *mut ffi::ggml_tensor| {
                let t = ffi::ggml_cont_4d(ctx, t, n_embd * 2, n_px / 2, n_py, 1);
                let t = ffi::ggml_reshape_4d(ctx, t, n_embd * 2, n_px / 2, 2, n_py / 2);
                let t = ffi::ggml_permute(ctx, t, 0, 2, 1, 3);
                ffi::ggml_cont_3d(ctx, t, n_embd, n_pos, 1)
            };

            // [w, h, c, b] -> [c, w, h, b], then the block reorder
            inp = ffi::ggml_permute(ctx, inp, 1, 2, 0, 3);
            inp = reorder(ctx, inp);
            inp = ffi::ggml_add(ctx, inp, wt("v.patch_embd.bias"));

            // learned position embedding, bilinearly resized to this grid,
            // then reordered exactly like the patches
            let mut pe = wt("v.position_embd.weight");
            if n_px != hp.pos_side || n_py != hp.pos_side {
                let side = hp.pos_side;
                pe = ffi::ggml_reshape_3d(ctx, pe, n_embd, side, side);
                pe = ffi::ggml_permute(ctx, pe, 2, 0, 1, 3);
                pe = ffi::ggml_interpolate(
                    ctx,
                    pe,
                    n_px,
                    n_py,
                    n_embd,
                    1,
                    ffi::ggml_scale_mode_GGML_SCALE_MODE_BILINEAR
                        | ffi::ggml_scale_flag_GGML_SCALE_FLAG_ALIGN_CORNERS,
                );
                pe = ffi::ggml_permute(ctx, pe, 1, 2, 0, 3);
                pe = ffi::ggml_cont_2d(ctx, pe, n_embd, n_pos);
            }
            let pe = reorder(ctx, pe);
            inp = ffi::ggml_add(ctx, inp, pe);

            let positions = ffi::ggml_new_tensor_1d(ctx, i32t, n_pos * 4);
            ffi::ggml_set_input(positions);

            let norm = |ctx: *mut ffi::ggml_context,
                        t: *mut ffi::ggml_tensor,
                        w: *mut ffi::ggml_tensor,
                        b: *mut ffi::ggml_tensor| {
                let t = ffi::ggml_norm(ctx, t, hp.eps);
                let t = ffi::ggml_mul(ctx, t, w);
                ffi::ggml_add(ctx, t, b)
            };

            let mut sections: [i32; 4] = [(d_head / 4) as i32; 4];
            let rope = |ctx: *mut ffi::ggml_context,
                        t: *mut ffi::ggml_tensor,
                        sections: &mut [i32; 4]| {
                ffi::ggml_rope_multi(
                    ctx,
                    t,
                    positions,
                    std::ptr::null_mut(),
                    (d_head / 2) as i32,
                    sections.as_mut_ptr(),
                    ffi::GGML_ROPE_TYPE_VISION as i32,
                    32768,
                    10000.0,
                    1.0,
                    0.0,
                    1.0,
                    32.0,
                    1.0,
                )
            };

            // CODPIECE_VISION_DEBUG=1: read checkpoint sums back after compute
            // to bisect any divergence against llama.cpp's cb_eval node dump.
            let debug = std::env::var("CODPIECE_VISION_DEBUG").is_ok();
            let mut checkpoints: Vec<(String, *mut ffi::ggml_tensor)> = Vec::new();
            let mut mark = |name: String, t: *mut ffi::ggml_tensor| {
                if debug {
                    ffi::ggml_set_output(t);
                    checkpoints.push((name, t));
                }
            };
            mark("inp_pos_emb".into(), inp);

            let mut cur = inp;
            for il in 0..hp.n_layer {
                let residual = cur;
                let mut x = norm(ctx, cur, blk(il, "ln1", "weight"), blk(il, "ln1", "bias"));

                // fused qkv, then head views into the [3*n_embd] rows
                x = ffi::ggml_mul_mat(ctx, blk(il, "attn_qkv", "weight"), x);
                x = ffi::ggml_add(ctx, x, blk(il, "attn_qkv", "bias"));
                let row = ffi::ggml_row_size((*x).type_, n_embd) as usize;
                let nb1 = ffi::ggml_row_size((*x).type_, d_head);
                let nb2 = (*x).nb[1];
                let q = ffi::ggml_view_3d(ctx, x, d_head, n_head, n_pos, nb1, nb2, 0);
                let k = ffi::ggml_view_3d(ctx, x, d_head, n_head, n_pos, nb1, nb2, row);
                let v = ffi::ggml_view_3d(ctx, x, d_head, n_head, n_pos, nb1, nb2, 2 * row);

                let q = rope(ctx, q, &mut sections);
                let k = rope(ctx, k, &mut sections);

                let mut o = if use_fa {
                    // Flash attention, as llama.cpp's clip runs on CUDA (head
                    // size 72 has a dedicated kernel). This is what makes big
                    // images affordable: the non-FA path materializes an
                    // n_pos^2 x n_head f32 KQ — over 1 GiB at a 1024-token
                    // image — where FA streams it.
                    let qp = ffi::ggml_permute(ctx, q, 0, 2, 1, 3);
                    let kp = ffi::ggml_cast(
                        ctx,
                        ffi::ggml_permute(ctx, k, 0, 2, 1, 3),
                        ffi::ggml_type_GGML_TYPE_F16,
                    );
                    let vp = ffi::ggml_cast(
                        ctx,
                        ffi::ggml_permute(ctx, v, 0, 2, 1, 3),
                        ffi::ggml_type_GGML_TYPE_F16,
                    );
                    let cur = ffi::ggml_flash_attn_ext(
                        ctx,
                        qp,
                        kp,
                        vp,
                        std::ptr::null_mut(),
                        kq_scale,
                        0.0,
                        0.0,
                    );
                    ffi::ggml_flash_attn_ext_set_prec(cur, ffi::ggml_prec_GGML_PREC_F32);
                    ffi::ggml_reshape_2d(ctx, cur, n_embd, n_pos)
                } else {
                    // exact non-FA reference path (the CPU parity harness)
                    let qp = ffi::ggml_permute(ctx, q, 0, 2, 1, 3);
                    let kp = ffi::ggml_permute(ctx, k, 0, 2, 1, 3);
                    let vp = ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, v, 1, 2, 0, 3));
                    let kq = ffi::ggml_mul_mat(ctx, kp, qp);
                    let kq = ffi::ggml_soft_max_ext(
                        ctx,
                        kq,
                        std::ptr::null_mut(),
                        kq_scale,
                        0.0,
                    );
                    let kqv = ffi::ggml_mul_mat(ctx, vp, kq);
                    let o = ffi::ggml_permute(ctx, kqv, 0, 2, 1, 3);
                    ffi::ggml_cont_2d(ctx, o, n_embd, n_pos)
                };
                o = ffi::ggml_mul_mat(ctx, blk(il, "attn_out", "weight"), o);
                o = ffi::ggml_add(ctx, o, blk(il, "attn_out", "bias"));

                cur = ffi::ggml_add(ctx, o, residual);
                let residual = cur;

                let mut f = norm(ctx, cur, blk(il, "ln2", "weight"), blk(il, "ln2", "bias"));
                f = ffi::ggml_mul_mat(ctx, blk(il, "ffn_up", "weight"), f);
                f = ffi::ggml_add(ctx, f, blk(il, "ffn_up", "bias"));
                f = ffi::ggml_gelu(ctx, f);
                f = ffi::ggml_mul_mat(ctx, blk(il, "ffn_down", "weight"), f);
                f = ffi::ggml_add(ctx, f, blk(il, "ffn_down", "bias"));

                cur = ffi::ggml_add(ctx, residual, f);
                mark(format!("layer_out.{il}"), cur);
            }

            cur = norm(ctx, cur, wt("v.post_ln.weight"), wt("v.post_ln.bias"));
            mark("post_ln".into(), cur);

            // merger: one row per 2x2 block -> GELU MLP into the trunk width
            let merge_sq = hp.merge * hp.merge;
            let mut emb = ffi::ggml_reshape_3d(ctx, cur, n_embd * merge_sq, n_pos / merge_sq, 1);
            emb = ffi::ggml_mul_mat(ctx, wt("mm.0.weight"), emb);
            emb = ffi::ggml_add(ctx, emb, wt("mm.0.bias"));
            emb = ffi::ggml_gelu(ctx, emb);
            emb = ffi::ggml_mul_mat(ctx, wt("mm.2.weight"), emb);
            emb = ffi::ggml_add(ctx, emb, wt("mm.2.bias"));
            ffi::ggml_set_output(emb);
            ffi::ggml_build_forward_expand(gf, emb);

            let finish = |galloc: ffi::ggml_gallocr_t, r: Result<Vec<f32>, ModelError>| {
                if !galloc.is_null() {
                    ffi::ggml_gallocr_free(galloc);
                }
                ffi::ggml_free(ctx);
                r
            };

            if n_threads > 0 && self.weights.is_cpu() {
                ffi::ggml_backend_cpu_set_n_threads(self.weights.backend(), n_threads);
            }
            let galloc = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(
                self.weights.backend(),
            ));
            if galloc.is_null() {
                return finish(galloc, Err(ModelError::Load("vision gallocr".into())));
            }
            if !ffi::ggml_gallocr_alloc_graph(galloc, gf) {
                return finish(galloc, Err(ModelError::Load("vision graph alloc".into())));
            }

            ffi::ggml_backend_tensor_set(
                inp_raw,
                img.as_ptr().cast(),
                0,
                img.len() * std::mem::size_of::<f32>(),
            );
            // M-RoPE positions: [y, x, y, x] per patch, section-major, filled
            // in the same 2x2 block order the reorder above put the rows in
            let n = n_pos as usize;
            let mut pos = vec![0i32; n * 4];
            let mut ptr = 0usize;
            for y in (0..n_py as usize).step_by(hp.merge as usize) {
                for x in (0..n_px as usize).step_by(hp.merge as usize) {
                    for dy in 0..hp.merge as usize {
                        for dx in 0..hp.merge as usize {
                            pos[ptr] = (y + dy) as i32;
                            pos[n + ptr] = (x + dx) as i32;
                            pos[2 * n + ptr] = (y + dy) as i32;
                            pos[3 * n + ptr] = (x + dx) as i32;
                            ptr += 1;
                        }
                    }
                }
            }
            ffi::ggml_backend_tensor_set(
                positions,
                pos.as_ptr().cast(),
                0,
                pos.len() * std::mem::size_of::<i32>(),
            );

            if ffi::ggml_backend_graph_compute(self.weights.backend(), gf)
                != ffi::ggml_status_GGML_STATUS_SUCCESS
            {
                return finish(galloc, Err(ModelError::Load("vision compute".into())));
            }

            for (name, t) in &checkpoints {
                let n = ffi::ggml_nelements(*t) as usize;
                let mut v = vec![0f32; n];
                ffi::ggml_backend_tensor_get(*t, v.as_mut_ptr().cast(), 0, n * 4);
                let sum: f32 = v.iter().sum();
                eprintln!("checkpoint {name}: sum = {sum:.6} (first {:.6})", v[0]);
            }

            let n_out = (n_pos / merge_sq) as usize * hp.proj_dim as usize;
            let mut out = vec![0f32; n_out];
            ffi::ggml_backend_tensor_get(
                emb,
                out.as_mut_ptr().cast(),
                0,
                n_out * std::mem::size_of::<f32>(),
            );
            finish(galloc, Ok(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_layout_matches_reference() {
        // the reference loop from clip.cpp, verbatim, against our fill
        let (pw, ph, merge) = (4usize, 4usize, 2usize);
        let n = pw * ph;
        let mut ours = vec![0i32; n * 4];
        let mut ptr = 0usize;
        for y in (0..ph).step_by(merge) {
            for x in (0..pw).step_by(merge) {
                for dy in 0..merge {
                    for dx in 0..merge {
                        ours[ptr] = (y + dy) as i32;
                        ours[n + ptr] = (x + dx) as i32;
                        ours[2 * n + ptr] = (y + dy) as i32;
                        ours[3 * n + ptr] = (x + dx) as i32;
                        ptr += 1;
                    }
                }
            }
        }
        // first block: (0,0) (0,1) (1,0) (1,1); second block starts at x=2
        assert_eq!(&ours[0..6], &[0, 0, 1, 1, 0, 0]);
        assert_eq!(&ours[n..n + 6], &[0, 1, 0, 1, 2, 3]);
        // sections 2 and 3 mirror 0 and 1
        assert_eq!(ours[..n], ours[2 * n..3 * n]);
        assert_eq!(ours[n..2 * n], ours[3 * n..]);
    }
}
