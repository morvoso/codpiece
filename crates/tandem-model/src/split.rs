//! Tensor-parallel split classification for qwen35.
//!
//! Port of llama.cpp's `llama_meta_device_get_split_state`
//! (llama-model.cpp:353-600 @ b10423, extracted to
//! `notes/reference/meta-split-state.cpp.txt`). Given a tensor name and the
//! model hparams, decide how that tensor is divided across devices.
//!
//! The classification is pure and shape-independent, so it is unit-testable
//! offline against the production GGUF's real tensor directory — no GPU, no
//! bench window. That matters because a wrong split is silently wrong: it
//! produces plausible garbage rather than an error.

use crate::qwen35::Hparams;

/// How a tensor is divided across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// split along ne[0] (rows within a row-major matrix: the "contraction"
    /// side; used by output projections that end in an all-reduce)
    Axis0,
    /// split along ne[1] (columns: input projections, each device owns whole
    /// output features)
    Axis1,
    /// split along ne[2] (used by tandem's 3-D GDN state, one [S,S] matrix
    /// per value head — the heads are what get distributed)
    Axis2,
    /// full copy on every device (norms, embeddings, scalars)
    Mirrored,
    /// present on one device only (per-expert biases)
    Partial,
}

/// A tensor's split: the axis plus the segment pattern along it.
///
/// `segments` is a list of (segment size, repeat count). The segments must sum
/// (with repeats) to the tensor's extent along `axis`. Fused tensors need
/// several segments so each device gets a slice of every part — e.g. a fused
/// QKV matrix is split so both devices own some Q, some K and some V.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub axis: Axis,
    pub segments: Vec<(i64, u32)>,
    /// Suffix of the tensor that is split along axis 0 for this same
    /// operation (e.g. an input projection's reference is the block's output
    /// projection). Its quantization block size sets the split granularity,
    /// so a column-split matrix and its row-split partner divide identically
    /// — otherwise the halves of a matmul chain stop lining up.
    pub axis0_ref: Option<&'static str>,
}

impl Split {
    fn mirrored() -> Split {
        Split { axis: Axis::Mirrored, segments: vec![], axis0_ref: None }
    }

    /// Total extent described by the segment list.
    pub fn extent(&self) -> i64 {
        self.segments.iter().map(|(sz, rep)| sz * *rep as i64).sum()
    }
}

/// Parse `blk.<N>.<rest>` → (N, rest).
fn layer_of(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (num, suffix) = rest.split_once('.')?;
    Some((num.parse().ok()?, suffix))
}

/// Classify a tensor. `name` is the GGUF tensor name; session cache tensors
/// use tandem's own names (`cache_k_l<N>`, `cache_v_l<N>`, `cache_conv_l<N>`,
/// `cache_gdn_l<N>`), classified the same way llama.cpp classifies its
/// `cache_[kvrs]_l*`.
pub fn classify(name: &str, hp: &Hparams) -> Split {
    // GDN head broadcast for qwen35 is [k0_v0, k1_v1, k0_v2, k1_v3] — V must
    // be segmented at K's granularity or head pairing scrambles silently.
    // (Qwen3-Next uses the contiguous pattern; do not copy its rules.)
    let head_ratio = (hp.n_v_heads / hp.n_k_heads) as u32;
    let key_dim = hp.key_dim();

    let seg = |axis: Axis, segments: Vec<(i64, u32)>| Split {
        axis,
        segments,
        axis0_ref: None,
    };
    let seg_ref = |axis: Axis, segments: Vec<(i64, u32)>, r: &'static str| Split {
        axis,
        segments,
        axis0_ref: Some(r),
    };

    // cache tensors first (they carry a layer suffix but not a blk. prefix)
    if let Some(rest) = name.strip_prefix("cache_") {
        let kind = rest.split("_l").next().unwrap_or("");
        return match kind {
            // KV cache rows are per-head: split like attn_output's input side
            "k" | "v" => seg_ref(Axis::Axis0, vec![], "attn_output.weight"),
            // NOTE: llama.cpp keeps its recurrent caches FLAT (one row per
            // sequence), so its axis/segment rules do not transfer to
            // tandem's shaped tensors. These follow tandem's own layouts:
            //
            // conv state is [d_conv-1, conv_dim]: the [q|k|v] channels live
            // on ne[1] and split exactly like ssm_conv1d.weight's ne[1].
            "conv" => seg_ref(
                Axis::Axis1,
                vec![(key_dim, 2 + head_ratio)],
                "ssm_out.weight",
            ),
            // GDN state is [S, S, n_v_heads]: whole per-head matrices move
            // with their head, so the split is over ne[2], like ssm_a.
            "gdn" => seg_ref(
                Axis::Axis2,
                vec![(hp.n_k_heads, head_ratio)],
                "ssm_out.weight",
            ),
            _ => Split::mirrored(),
        };
    }

    let Some((_il, suffix)) = layer_of(name) else {
        // non-layer tensors
        return match name {
            "output.weight" => seg(Axis::Axis1, vec![]),
            "output.bias" => seg(Axis::Axis0, vec![]),
            // token_embd, output_norm, and anything else: replicated
            _ => Split::mirrored(),
        };
    };

    match suffix {
        // ---- attention ----
        // fused QKV: 2 key-sized parts (Q,K) plus head_ratio value parts
        "attn_qkv.weight" => {
            seg_ref(Axis::Axis1, vec![(key_dim, 2 + head_ratio)], "ssm_out.weight")
        }
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" => {
            seg_ref(Axis::Axis1, vec![], "attn_output.weight")
        }
        "attn_q.bias" | "attn_k.bias" | "attn_v.bias" | "attn_qkv.bias" => {
            seg(Axis::Axis0, vec![])
        }
        // per-head q/k norms are vectors of head_dim → replicated
        "attn_q_norm.weight" | "attn_k_norm.weight" => Split::mirrored(),
        "attn_output.weight" => seg(Axis::Axis0, vec![]),
        "attn_output.bias" => Split::mirrored(),
        "attn_gate.weight" => {
            seg_ref(Axis::Axis1, vec![(key_dim, head_ratio)], "ssm_out.weight")
        }

        // ---- gated delta net ----
        "ssm_dt.bias" | "ssm_a" => {
            seg_ref(Axis::Axis0, vec![(hp.n_k_heads, head_ratio)], "ssm_out.weight")
        }
        "ssm_alpha.weight" | "ssm_beta.weight" => {
            seg_ref(Axis::Axis1, vec![(hp.n_k_heads, head_ratio)], "ssm_out.weight")
        }
        "ssm_conv1d.weight" => {
            seg_ref(Axis::Axis1, vec![(key_dim, 2 + head_ratio)], "ssm_out.weight")
        }
        "ssm_out.weight" => seg(Axis::Axis0, vec![(key_dim, head_ratio)]),
        "ssm_norm.weight" => Split::mirrored(),

        // ---- ffn ----
        "ffn_up.weight" | "ffn_gate.weight" => {
            seg_ref(Axis::Axis1, vec![], "ffn_down.weight")
        }
        "ffn_up.bias" | "ffn_gate.bias" => seg(Axis::Axis0, vec![]),
        "ffn_down.weight" => seg(Axis::Axis0, vec![]),
        "ffn_down.bias" => Split::mirrored(),

        // ---- norms and anything unrecognized ----
        _ => Split::mirrored(),
    }
}

/// Load-balancing rotation: leftover rows alternate between devices instead of
/// always landing on device 0. Counts only same-kind previous layers (llama.cpp
/// does the same) so an alternating GDN/attention stack doesn't alias.
pub fn rotation(name: &str, hp: &Hparams, n_devices: usize) -> usize {
    let il = layer_of(name)
        .map(|(il, _)| il)
        .or_else(|| {
            name.strip_prefix("cache_")
                .and_then(|r| r.split("_l").nth(1))
                .and_then(|n| n.parse().ok())
        });
    match il {
        Some(il) => {
            let is_recr = hp.is_recurrent(il);
            let same_kind_before = (0..il).filter(|&p| hp.is_recurrent(p) == is_recr).count();
            same_kind_before % n_devices
        }
        None => hp.n_layer % n_devices,
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

fn lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 { 0 } else { (a / gcd(a, b)).abs() * b.abs() }
}

/// Split granularity: each device's slice must be a multiple of this, so that
/// quantization blocks are never cut in half and the per-device shapes stay
/// kernel-friendly. `blck` is the block size of the axis-0 reference tensor's
/// type (see `Split::axis0_ref`).
pub fn granularity(name: &str, hp: &Hparams, blck: i64, n_devices: usize) -> i64 {
    let Some((il, suffix)) = layer_of(name).or_else(|| {
        name.strip_prefix("cache_")
            .and_then(|r| r.split_once("_l"))
            .and_then(|(kind, n)| n.parse::<usize>().ok().map(|il| (il, kind)))
            .map(|(il, kind)| (il, kind))
    }) else {
        return 1;
    };

    if hp.is_recurrent(il) {
        let head_dim = hp.d_state;
        let blck_perf = lcm(blck, 128);
        let g_qkv = lcm(blck_perf, head_dim);
        match suffix {
            "attn_qkv.weight" | "attn_gate.weight" | "ssm_conv1d.weight"
            | "ssm_out.weight" => return g_qkv,
            "ssm_dt.bias" | "ssm_a" | "ssm_alpha.weight" | "ssm_beta.weight" => {
                return g_qkv / head_dim
            }
            // cache_* suffixes arrive as the bare kind; tandem's conv state
            // splits its channel axis like the conv weight, and its GDN state
            // splits whole heads
            "conv" => return g_qkv,
            "gdn" => return g_qkv / head_dim,
            _ => {}
        }
    } else {
        let n_gqa = hp.n_head / hp.n_head_kv;
        let n_embd_q = n_gqa * hp.head_k;
        // raise granularity only while every device still gets work
        let mut blck_perf = blck;
        while blck_perf < 128 && blck_perf * (n_devices as i64) < n_embd_q {
            blck_perf *= 2;
        }
        let g_q = lcm(n_embd_q, blck_perf);
        match suffix {
            // qwen35 packs a gate beside Q, so its granularity doubles
            "attn_q.weight" | "attn_q.bias" => return lcm(2 * n_embd_q, blck_perf),
            "attn_output.weight" => return g_q,
            "attn_k.weight" | "attn_v.weight" | "attn_k.bias" | "attn_v.bias" | "k" | "v" => {
                return g_q / n_gqa
            }
            _ => {}
        }
    }

    match suffix {
        "ffn_up.weight" | "ffn_gate.weight" | "ffn_down.weight" | "ffn_up.bias"
        | "ffn_gate.bias" => lcm(blck, 128),
        _ => 1,
    }
}

/// The C-facing split description: per-(segment, device) extents.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitState {
    pub axis: Axis,
    /// ne[segment * n_devices + device]
    pub ne: Vec<i64>,
    /// repeats per segment
    pub nr: Vec<u32>,
    pub n_segments: usize,
}

/// Divide `tensor_extent` (the tensor's size along the split axis) across
/// devices, honoring segments, granularity and rotation.
pub fn split_state(
    name: &str,
    hp: &Hparams,
    tensor_extent: i64,
    blck: i64,
    n_devices: usize,
) -> SplitState {
    let split = classify(name, hp);
    if !matches!(split.axis, Axis::Axis0 | Axis::Axis1 | Axis::Axis2) {
        return SplitState { axis: split.axis, ne: vec![], nr: vec![1], n_segments: 1 };
    }

    // an empty segment list means "one segment covering the whole axis"
    let segments: Vec<(i64, u32)> = if split.segments.is_empty() {
        vec![(tensor_extent, 1)]
    } else {
        split.segments.clone()
    };
    let g = granularity(name, hp, blck, n_devices).max(1);
    let rot = rotation(name, hp, n_devices);

    let mut ne = vec![0i64; segments.len() * n_devices];
    let mut nr = vec![0u32; segments.len()];
    for (is, &(ne_s, nr_s)) in segments.iter().enumerate() {
        let mut low = 0i64;
        for j in 0..n_devices - 1 {
            let mut high = ne_s * (j as i64 + 1) / n_devices as i64;
            high -= high % g;
            ne[is * n_devices + (j + rot) % n_devices] = high - low;
            low = high;
        }
        ne[is * n_devices + (n_devices - 1 + rot) % n_devices] = ne_s - low;
        nr[is] = nr_s;
    }
    SplitState { axis: split.axis, ne, nr, n_segments: segments.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Production 27B hyperparameters (verified from the GGUF metadata).
    fn hp_27b() -> Hparams {
        Hparams {
            n_layer: 64,
            n_embd: 5120,
            n_head: 24,
            n_head_kv: 4,
            head_k: 256,
            head_v: 256,
            n_ff: 17408,
            rms_eps: 1e-6,
            n_rot: 64,
            rope_sections: [11, 11, 10, 0],
            freq_base: 1e7,
            n_ctx_train: 262144,
            d_conv: 4,
            n_k_heads: 16,
            n_v_heads: 48,
            d_state: 128,
            d_inner: 6144,
            full_attn_interval: 4,
            n_vocab: 248320,
        }
    }

    #[test]
    fn qwen35_head_ratio_segments_cover_the_tensor() {
        let hp = hp_27b();
        let key_dim = hp.key_dim(); // 16 * 128 = 2048
        assert_eq!(key_dim, 2048);
        assert_eq!(hp.conv_dim(), 10240); // 2*2048 + 6144

        // fused qkv / conv1d: segments must tile the whole conv dim
        for n in ["blk.0.attn_qkv.weight", "blk.0.ssm_conv1d.weight"] {
            let s = classify(n, &hp);
            assert_eq!(s.axis, Axis::Axis1, "{n}");
            assert_eq!(s.extent(), hp.conv_dim(), "{n} segments must tile conv_dim");
        }

        // gate / out: value_dim worth of value-head segments
        for n in ["blk.0.attn_gate.weight", "blk.0.ssm_out.weight"] {
            let s = classify(n, &hp);
            assert_eq!(s.extent(), hp.value_dim(), "{n} segments must tile value_dim");
        }

        // per-head scalars: one entry per V head
        for n in ["blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_alpha.weight"] {
            let s = classify(n, &hp);
            assert_eq!(s.extent(), hp.n_v_heads, "{n} segments must tile n_v_heads");
        }
    }

    #[test]
    fn split_axes_match_llama_cpp() {
        let hp = hp_27b();
        // input projections split by columns, output projections by rows:
        // the row split is what an all-reduce sums back together.
        assert_eq!(classify("blk.3.attn_q.weight", &hp).axis, Axis::Axis1);
        assert_eq!(classify("blk.3.attn_output.weight", &hp).axis, Axis::Axis0);
        assert_eq!(classify("blk.0.ffn_up.weight", &hp).axis, Axis::Axis1);
        assert_eq!(classify("blk.0.ffn_gate.weight", &hp).axis, Axis::Axis1);
        assert_eq!(classify("blk.0.ffn_down.weight", &hp).axis, Axis::Axis0);
        assert_eq!(classify("blk.0.ssm_out.weight", &hp).axis, Axis::Axis0);
        assert_eq!(classify("output.weight", &hp).axis, Axis::Axis1);
        // replicated
        for n in [
            "token_embd.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ssm_norm.weight",
            "blk.3.attn_q_norm.weight",
        ] {
            assert_eq!(classify(n, &hp).axis, Axis::Mirrored, "{n}");
        }
    }

    #[test]
    fn session_caches_classified() {
        let hp = hp_27b();
        assert_eq!(classify("cache_k_l3", &hp).axis, Axis::Axis0);
        assert_eq!(classify("cache_v_l3", &hp).axis, Axis::Axis0);
        // conv state splits its channel axis (ne[1] == conv_dim)
        let conv = classify("cache_conv_l0", &hp);
        assert_eq!(conv.axis, Axis::Axis1);
        assert_eq!(conv.extent(), hp.conv_dim());
        // gdn state splits whole heads (ne[2] == n_v_heads)
        let gdn = classify("cache_gdn_l0", &hp);
        assert_eq!(gdn.axis, Axis::Axis2);
        assert_eq!(gdn.extent(), hp.n_v_heads);
    }

    #[test]
    fn split_state_conserves_every_element() {
        let hp = hp_27b();
        // (name, extent along the split axis, block size of the axis-0 ref)
        let cases: &[(&str, i64, i64)] = &[
            // GDN layer 0
            ("blk.0.attn_qkv.weight", 10240, 32),
            ("blk.0.attn_gate.weight", 6144, 32),
            ("blk.0.ssm_conv1d.weight", 10240, 1),
            ("blk.0.ssm_out.weight", 6144, 32),
            ("blk.0.ssm_a", 48, 32),
            ("blk.0.ssm_dt.bias", 48, 32),
            ("blk.0.ssm_alpha.weight", 48, 32),
            ("blk.0.ssm_beta.weight", 48, 32),
            // attention layer 3
            ("blk.3.attn_q.weight", 12288, 32), // 24 heads * 256 * 2 (Q+gate)
            ("blk.3.attn_k.weight", 1024, 32),  // 4 kv heads * 256
            ("blk.3.attn_v.weight", 1024, 32),
            ("blk.3.attn_output.weight", 6144, 32),
            // ffn
            ("blk.0.ffn_up.weight", 17408, 32),
            ("blk.0.ffn_gate.weight", 17408, 32),
            ("blk.0.ffn_down.weight", 17408, 32),
            // output head
            ("output.weight", 248320, 32),
            // session caches
            ("cache_k_l3", 1024, 32),
            ("cache_v_l3", 1024, 32),
            // tandem layouts: conv [d_conv-1, conv_dim] splits ne[1];
            // gdn [S, S, n_v_heads] splits ne[2]
            ("cache_conv_l0", 10240, 32),
            ("cache_gdn_l0", 48, 32),
        ];
        for &(name, extent, blck) in cases {
            let st = split_state(name, &hp, extent, blck, 2);
            // every element lands on exactly one device
            let total: i64 = (0..st.n_segments)
                .map(|is| {
                    let per: i64 = (0..2).map(|d| st.ne[is * 2 + d]).sum();
                    per * st.nr[is] as i64
                })
                .sum();
            assert_eq!(total, extent, "{name}: split must conserve the axis");
            // no device gets a negative or absurd share
            for d in 0..2 {
                for is in 0..st.n_segments {
                    let v = st.ne[is * 2 + d];
                    assert!(v >= 0, "{name}: negative share on device {d}");
                }
            }
        }
    }

    #[test]
    fn split_state_respects_granularity_and_balance() {
        let hp = hp_27b();
        // Q is split at 3072 granularity (2*n_embd_q = 2*6*256): 12288 halves
        // cleanly into 6144 each.
        let q = split_state("blk.3.attn_q.weight", &hp, 12288, 32, 2);
        assert_eq!(q.ne[0], 6144);
        assert_eq!(q.ne[1], 6144);
        // KV granularity is 256: 1024 -> 512/512
        let k = split_state("blk.3.attn_k.weight", &hp, 1024, 32, 2);
        assert_eq!(k.ne[0] + k.ne[1], 1024);
        assert_eq!(k.ne[0] % 256, 0);
        // GDN fused qkv: 3 segment kinds via repeats (2 key + 3 value at
        // key scale), each segment split at 128 granularity
        let qkv = split_state("blk.0.attn_qkv.weight", &hp, 10240, 32, 2);
        assert_eq!(qkv.n_segments, 1);
        assert_eq!(qkv.nr[0], 5); // 2 + head_ratio(3)
        assert_eq!(qkv.ne[0] + qkv.ne[1], 2048); // one key-sized segment
        assert_eq!(qkv.ne[0] % 128, 0);
        // FFN splits at 128
        let up = split_state("blk.0.ffn_up.weight", &hp, 17408, 32, 2);
        assert_eq!(up.ne[0] % 128, 0);
        assert_eq!(up.ne[0] + up.ne[1], 17408);
        // mirrored tensors carry no per-device extents
        let n = split_state("blk.0.attn_norm.weight", &hp, 5120, 1, 2);
        assert_eq!(n.axis, Axis::Mirrored);
    }

    #[test]
    fn rotation_alternates_within_layer_kind() {
        let hp = hp_27b();
        // GDN layers 0,1,2 are the 1st/2nd/3rd recurrent layers → 0,1,0
        assert_eq!(rotation("blk.0.ssm_out.weight", &hp, 2), 0);
        assert_eq!(rotation("blk.1.ssm_out.weight", &hp, 2), 1);
        assert_eq!(rotation("blk.2.ssm_out.weight", &hp, 2), 0);
        // attention layers 3,7 are the 1st/2nd attention layers → 0,1
        assert_eq!(rotation("blk.3.attn_output.weight", &hp, 2), 0);
        assert_eq!(rotation("blk.7.attn_output.weight", &hp, 2), 1);
        // caches follow their layer
        assert_eq!(rotation("cache_k_l7", &hp, 2), 1);
    }
}
