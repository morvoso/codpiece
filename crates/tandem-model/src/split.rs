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
}

impl Split {
    fn mirrored() -> Split {
        Split { axis: Axis::Mirrored, segments: vec![] }
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
    let head_v = hp.gdn_head_v();

    let seg = |axis: Axis, segments: Vec<(i64, u32)>| Split { axis, segments };

    // cache tensors first (they carry a layer suffix but not a blk. prefix)
    if let Some(rest) = name.strip_prefix("cache_") {
        let kind = rest.split("_l").next().unwrap_or("");
        return match kind {
            // KV cache rows are per-head: split like attn_output's input side
            "k" | "v" => seg(Axis::Axis0, vec![]),
            // conv state: [q|k|v] channels, same segmentation as ssm_conv1d
            "conv" => seg(
                Axis::Axis0,
                vec![(key_dim * (hp.d_conv - 1), 2 + head_ratio)],
            ),
            // GDN recurrent state: one [S,S] matrix per V head
            "gdn" => seg(Axis::Axis0, vec![(hp.n_k_heads * head_v * head_v, head_ratio)]),
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
        "attn_qkv.weight" => seg(Axis::Axis1, vec![(key_dim, 2 + head_ratio)]),
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" => seg(Axis::Axis1, vec![]),
        "attn_q.bias" | "attn_k.bias" | "attn_v.bias" | "attn_qkv.bias" => {
            seg(Axis::Axis0, vec![])
        }
        // per-head q/k norms are vectors of head_dim → replicated
        "attn_q_norm.weight" | "attn_k_norm.weight" => Split::mirrored(),
        "attn_output.weight" => seg(Axis::Axis0, vec![]),
        "attn_output.bias" => Split::mirrored(),
        "attn_gate.weight" => seg(Axis::Axis1, vec![(key_dim, head_ratio)]),

        // ---- gated delta net ----
        "ssm_dt.bias" | "ssm_a" => seg(Axis::Axis0, vec![(hp.n_k_heads, head_ratio)]),
        "ssm_alpha.weight" | "ssm_beta.weight" => {
            seg(Axis::Axis1, vec![(hp.n_k_heads, head_ratio)])
        }
        "ssm_conv1d.weight" => seg(Axis::Axis1, vec![(key_dim, 2 + head_ratio)]),
        "ssm_out.weight" => seg(Axis::Axis0, vec![(key_dim, head_ratio)]),
        "ssm_norm.weight" => Split::mirrored(),

        // ---- ffn ----
        "ffn_up.weight" | "ffn_gate.weight" => seg(Axis::Axis1, vec![]),
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
        let conv = classify("cache_conv_l0", &hp);
        assert_eq!(conv.extent(), hp.conv_dim() * (hp.d_conv - 1));
        let gdn = classify("cache_gdn_l0", &hp);
        // one head_v x head_v matrix per V head
        assert_eq!(gdn.extent(), hp.n_v_heads * hp.gdn_head_v() * hp.gdn_head_v());
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
