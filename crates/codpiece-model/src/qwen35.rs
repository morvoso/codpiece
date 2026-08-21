//! qwen35 forward graph (trunk only: GDN + full-attention layers).
//!
//! Faithful port of llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp`
//! (snapshots in notes/reference/). One graph builder serves three modes:
//!
//! - **Stateless** (M1 reference rig): whole prefix recomputed, zero initial
//!   recurrent states, no KV cache. O(n²); exists to be diffed against
//!   llama.cpp. All M1 gates passed on this path.
//! - **Session general step** (prefill / odd shapes): KV caches + carried
//!   conv/GDN states; cache writes via views at n_past offsets.
//! - **Session cached decode step** (the hot path): a T=1 graph built ONCE
//!   per KV bucket whose topology is completely independent of n_past — KV
//!   writes go through ggml_set_rows (write position is input DATA), reads
//!   cover a fixed padded bucket, and the same ggml_cgraph object is
//!   recomputed every step. That stability is what lets ggml-cuda's CUDA
//!   graph capture engage (it fast-paths on cgraph uid, exactly like
//!   llama.cpp's graph reuse).
//!
//! Numerics policy: KV caches are f16 (prod parity; the all-f32 cache variant
//! hit undertested CUDA kernels — see git history). Flash attention is the
//! default session path; parity is checked path-matched (codpiece FA ↔ oracle
//! -fa on, codpiece non-FA ↔ oracle -fa off).

use codpiece_ggml_sys as ffi;
use codpiece_gguf::Value;

use crate::{ModelError, Weights};

/// KV window bucket: decode graphs keep identical shapes while n_past + T
/// stays within the same bucket multiple. Padded cells are zero-initialized
/// and masked to -inf.
const KV_BUCKET_DEFAULT: i64 = 256;

/// Bucket granularity: bigger = fewer graph rebuilds but more masked-out
/// attention work per step; smaller = tighter attention but more rebuilds.
/// CODPIECE_KV_BUCKET overrides for measurement.
fn kv_bucket() -> i64 {
    std::env::var("CODPIECE_KV_BUCKET")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(KV_BUCKET_DEFAULT)
}

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
    pub fn from_gguf(g: &codpiece_gguf::GgufFile) -> Result<Hparams, ModelError> {
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
    /// Layers whose INPUT hidden states every graph exports (the DFlash2
    /// drafter consumes them). Empty = no taps. Set once before serving.
    pub tap_layers: Vec<usize>,
}

/// Layer tensor handles, resolved per graph build.
pub(crate) struct Layer {
    pub(crate) attn_norm: *mut ffi::ggml_tensor,
    pub(crate) post_attn_norm: *mut ffi::ggml_tensor,
    // full attention
    pub(crate) wq: *mut ffi::ggml_tensor,
    pub(crate) wk: *mut ffi::ggml_tensor,
    pub(crate) wv: *mut ffi::ggml_tensor,
    pub(crate) wo: *mut ffi::ggml_tensor,
    pub(crate) q_norm: *mut ffi::ggml_tensor,
    pub(crate) k_norm: *mut ffi::ggml_tensor,
    // gdn
    pub(crate) wqkv: *mut ffi::ggml_tensor,
    pub(crate) wqkv_gate: *mut ffi::ggml_tensor,
    pub(crate) conv1d: *mut ffi::ggml_tensor,
    pub(crate) dt_bias: *mut ffi::ggml_tensor,
    pub(crate) ssm_a: *mut ffi::ggml_tensor,
    pub(crate) ssm_beta: *mut ffi::ggml_tensor,
    pub(crate) ssm_alpha: *mut ffi::ggml_tensor,
    pub(crate) ssm_norm: *mut ffi::ggml_tensor,
    pub(crate) ssm_out: *mut ffi::ggml_tensor,
    // ffn
    pub(crate) ffn_gate: *mut ffi::ggml_tensor,
    pub(crate) ffn_up: *mut ffi::ggml_tensor,
    pub(crate) ffn_down: *mut ffi::ggml_tensor,
}

/// Raw pointers into a Session's persistent tensors — plain data, so graph
/// building never holds a Rust borrow of the Session itself.
#[derive(Clone)]
struct SessView {
    k_slots: i64,
    k_cache: Vec<*mut ffi::ggml_tensor>,
    v_cache: Vec<*mut ffi::ggml_tensor>,
    conv_state: Vec<*mut ffi::ggml_tensor>,
    gdn_state: Vec<*mut ffi::ggml_tensor>,
    n_ctx_max: usize,
    fa: bool,
}

/// What the fused draft tail needs from the session.
#[derive(Clone, Copy)]
pub(crate) struct MtpTail {
    pub k_cache: *mut ffi::ggml_tensor,
    pub v_cache: *mut ffi::ggml_tensor,
    pub n_ctx_max: usize,
    pub n_past: usize,
    pub n_kv: i64,
    /// how many drafts to chain inside this one graph
    pub depth: usize,
    /// Size of the candidate vocabulary the DRAFT head projects onto, or 0
    /// for the full vocabulary.
    ///
    /// A draft only needs its argmax, so projecting onto a shortlist instead
    /// of all 248k rows turns a 1.27 GiB read into a few tens of MB. It stays
    /// lossless because verification always uses the FULL vocabulary: if the
    /// true token is outside the shortlist the draft is simply wrong and gets
    /// rejected, exactly like any other bad draft.
    pub n_cand: i64,
}

/// Kernel-path bisect switches, kept from the CUDA debugging campaign.
#[derive(Clone, Copy, Default)]
pub(crate) struct DebugToggles {
    attn_batch: bool,
    gdn_zero: bool,
    no_writes: bool,
    k_batch: bool,
    v_batch: bool,
}

impl DebugToggles {
    fn from_env() -> DebugToggles {
        let on = |k: &str| std::env::var(k).is_ok();
        DebugToggles {
            attn_batch: on("CODPIECE_DBG_ATTN_BATCH"),
            gdn_zero: on("CODPIECE_DBG_GDN_ZERO"),
            no_writes: on("CODPIECE_DBG_NO_WRITES"),
            k_batch: on("CODPIECE_DBG_K_BATCH"),
            v_batch: on("CODPIECE_DBG_V_BATCH"),
        }
    }
}

/// Either raw logits or an in-graph-sampled token id.
enum StepOut {
    Logits(Vec<f32>),
    Token(u32),
    /// in-graph argmax at several positions
    Tokens(Vec<u32>),
}

/// How a graph's tokens map onto sequences.
///
/// The attention layers, FFN and sampling are already per-token — position, mask row
/// and KV write row are per-token inputs — so serving several sequences at once only
/// changes the recurrent layers, whose state carries one slice per sequence.
#[derive(Clone, Copy, PartialEq)]
enum SeqMode {
    /// All tokens belong to one sequence using recurrent-state slot 0 and the
    /// K-snapshot rollback machinery. Everything before batching.
    Single,
    /// All tokens belong to one sequence, but its recurrent state lives in slot `.0`
    /// of a shared batch session, and exactly one snapshot (the live state) is kept.
    /// Used to prefill one sequence of a batch.
    Slot(i64),
    /// `t_len` tokens, one per sequence: token i is sequence i's next token, the
    /// state's slot dimension is the sequence dimension, and the GDN op runs in its
    /// `n_seqs` form with K = 1.
    Batched,
}

enum StateSrc {
    /// Zero states, no KV cache, positions start at 0 (reference rig).
    Stateless,
    Session(SessView),
}

/// A fully built (not yet allocated) forward graph plus its input handles.
struct Built {
    ctx: *mut ffi::ggml_context,
    gf: *mut ffi::ggml_cgraph,
    inp_tokens: *mut ffi::ggml_tensor,
    /// raw [n_embd, t_len] input in place of the token lookup: image chunks
    /// feed the vision tower's output here (null on token graphs)
    inp_embd: *mut ffi::ggml_tensor,
    /// T-scaled gumbel noise [n_vocab, n_out] added to the trunk logits
    /// before the in-graph argmax, turning it into an exact temperature
    /// sample the chain can condition on (null unless the round samples)
    inp_gumbel: *mut ffi::ggml_tensor,
    inp_pos: *mut ffi::ggml_tensor,
    /// layer-input hiddens at tap_layers, [n_embd, t_len] each
    taps: Vec<*mut ffi::ggml_tensor>,
    kq_mask: *mut ffi::ggml_tensor,
    /// set_rows write positions (cached decode graphs only)
    row_ids: *mut ffi::ggml_tensor,
    conv_zero: *mut ffi::ggml_tensor,
    state_zero: *mut ffi::ggml_tensor,
    out_ids: *mut ffi::ggml_tensor,
    out: *mut ffi::ggml_tensor,
    /// The trunk's logits before sampling. Kept readable even when the graph samples
    /// in-place: verifying a draft above temperature 0 asks how likely the target was
    /// to produce the drafted token, which needs the distribution, not the winner.
    logits: *mut ffi::ggml_tensor,
    /// pre-LM-head hidden states at the requested positions (MTP input)
    h_out: *mut ffi::ggml_tensor,
    /// fused draft head outputs, when the MTP tail is attached
    draft_out: *mut ffi::ggml_tensor,
    /// chained drafts beyond the first, in order
    draft_chain: Vec<*mut ffi::ggml_tensor>,
    /// per-link softmax peak at the drafted token, aligned with draft_out+draft_chain
    draft_conf: Vec<*mut ffi::ggml_tensor>,
    /// candidate token ids the draft head projects onto (empty = full vocab)
    cand_ids: *mut ffi::ggml_tensor,
    mtp_pos: *mut ffi::ggml_tensor,
    mtp_mask: *mut ffi::ggml_tensor,
    mtp_rows: *mut ffi::ggml_tensor,
    mtp_n_kv: i64,
    n_kv: i64,
    fa_mask: bool,
    /// out is an I32 token id (in-graph argmax) rather than f32 logits
    greedy: bool,
}

/// Cached T=1 decode graph with its dedicated allocator: the same cgraph
/// object is recomputed every step (stable uid → CUDA graph capture engages).
struct CachedStep {
    built: Built,
    galloc: ffi::ggml_gallocr_t,
    bucket: i64,
    greedy: bool,
    /// tokens per step this graph was built for (1 = decode, n_spec+1 = verify)
    t_len: i64,
    /// Host staging buffers, owned for the graph's lifetime: async uploads
    /// must not reference stack temporaries.
    host: HostStaging,
}

/// Per-step input staging. Reused every decode step (no per-token allocation)
/// and stable in memory so ggml_backend_tensor_set_async is sound.
struct HostStaging {
    tokens: Vec<i32>,
    pos: Vec<i32>,
    mask_f16: Vec<u16>,
    row_ids: Vec<i64>,
    /// how many leading mask cells are visible (0.0); the rest stay -inf
    mask_visible: usize,
    out_ids: Vec<i32>,
}

impl Drop for CachedStep {
    fn drop(&mut self) {
        unsafe {
            ffi::ggml_gallocr_free(self.galloc);
            ffi::ggml_free(self.built.ctx);
        }
    }
}

/// Persistent per-sequence state: KV caches for attention layers, conv + GDN
/// recurrent states for GDN layers. Lives in its own backend buffer, distinct
/// from any compute buffer, so graph allocation can never reuse it.

/// Report a graph's compute-buffer size when `CODPIECE_TRACE_MEM=1`.
///
/// At long context the question "what actually does not fit" is otherwise guesswork:
/// the weights and KV cache are easy to compute by hand, and everything left over is
/// this.
/// Window the draft head's KV cache reaches back over. `CODPIECE_MTP_CTX` overrides.
fn mtp_ctx_cap() -> usize {
    std::env::var("CODPIECE_MTP_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(16384)
}

pub(crate) unsafe fn trace_mem(what: &str, galloc: ffi::ggml_gallocr_t, n_kv: i64, t_len: i64) {
    use std::sync::atomic::{AtomicI8, Ordering};
    static ON: AtomicI8 = AtomicI8::new(-1);
    let mut on = ON.load(Ordering::Relaxed);
    if on < 0 {
        on = i8::from(std::env::var("CODPIECE_TRACE_MEM").as_deref() == Ok("1"));
        ON.store(on, Ordering::Relaxed);
    }
    if on == 1 {
        let bytes = ffi::ggml_gallocr_get_buffer_size(galloc, 0);
        eprintln!(
            "[mem] {what}: compute buffer {:.1} MiB (n_kv {n_kv}, t_len {t_len})",
            bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

pub struct Session {
    pub n_ctx_max: usize,
    pub n_past: usize,
    /// Tap features of the most recent graph run, `[n_taps * n_embd]` per
    /// token in batch order — filled only when the model has tap_layers.
    pub last_taps: Vec<f32>,
    /// RoPE position minus physical row position. Zero for text-only
    /// conversations. An image chunk of nx x ny merged patches occupies
    /// nx*ny physical rows but advances the RoPE clock by only max(nx, ny)
    /// (the Qwen-VL rule), so every image makes this more negative by
    /// nx*ny - max(nx, ny). All TEXT position fills add this offset; the
    /// physical n_past keeps indexing KV rows and masks.
    pub rope_off: i64,
    /// Recurrent-state snapshot slots: K = n_spec + 1. Slot 0 is live; slot s
    /// holds the state as of s tokens earlier. Speculative decoding advances
    /// the recurrent layers over draft tokens that may be rejected, and those
    /// layers cannot be run backwards — the snapshots ARE the undo.
    pub k_slots: i64,
    /// flash-attention path (untransposed V cache); CODPIECE_NO_FA=1 disables
    pub fa: bool,
    /// whether the caches live on the tensor-parallel meta device
    tp: bool,
    ctx: *mut ffi::ggml_context,
    buffer: ffi::ggml_backend_buffer_t,
    /// scratch allocator for general (prefill / odd-shaped) steps
    galloc: ffi::ggml_gallocr_t,
    k_cache: Vec<*mut ffi::ggml_tensor>,
    v_cache: Vec<*mut ffi::ggml_tensor>,
    conv_state: Vec<*mut ffi::ggml_tensor>,
    gdn_state: Vec<*mut ffi::ggml_tensor>,
    /// KV cache for the single MTP draft block (blk.n_layer)
    mtp_k: *mut ffi::ggml_tensor,
    mtp_v: *mut ffi::ggml_tensor,
    /// How far back the draft head's own KV cache reaches. Bounded independently of the
    /// trunk's context, because it is an optimisation and not a correctness
    /// requirement: every draft is verified against the full model, so a shorter window
    /// can only cost acceptance. At 196K the unbounded version is 0.38 GiB per card,
    /// which is most of the headroom that decides whether speculation fits at all.
    pub mtp_ctx_max: usize,
    /// how many tokens the MTP block has consumed
    pub mtp_past: usize,
    /// MTP draft graph, rebuilt only when the KV bucket changes. Building it
    /// per draft cost ~6 ms on the 27B — several times the head's actual
    /// compute — and prevented CUDA-graph reuse.
    mtp_cached: Option<(crate::mtp_graph::MtpGraph, ffi::ggml_gallocr_t, i64)>,
    /// Fused verify+draft graphs, keyed by their shape. Adaptive depth moves between
    /// depths round to round, and a switch also produces a transient shape (the batch
    /// carries the old depth's drafts while the chain runs at the new depth), so this
    /// is a small LRU rather than a single slot. Each entry owns a compute buffer, so
    /// the cap is a VRAM decision as much as a hit-rate one.
    fused: Vec<FusedStep>,
    fused_clock: u64,
    /// the one replayed graph of batched decoding, keyed by sequence count
    batch_step: Option<BatchStep>,
    cached: Option<CachedStep>,
    /// One rollback graph per rollback distance, kept for the life of the session.
    /// Distance is a view offset, so it cannot be an input — but there are only
    /// k_slots of them. Keeping the contexts alive matters for more than the build
    /// cost: the meta backend keys its per-device tensor map on raw tensor pointers,
    /// so a context that is freed and re-allocated every round hands the same
    /// addresses to different tensors and silently aliases another graph's entries.
    rollback: Vec<Option<RollbackGraph>>,
    /// Contexts of superseded cached graphs, held until the session ends.
    ///
    /// Freeing one is not safe while the session lives: the meta backend keys its
    /// per-device tensor map on raw tensor pointers, and a freed metadata context
    /// hands the same addresses back to the next graph, which then aliases the dead
    /// graph's entries. These are metadata-only (no_alloc) contexts of a few MB, and
    /// a session supersedes a graph only when the KV bucket grows, so there are a
    /// handful at most.
    graveyard: Vec<*mut ffi::ggml_context>,
}

struct RollbackGraph {
    ctx: *mut ffi::ggml_context,
    gf: *mut ffi::ggml_cgraph,
    galloc: ffi::ggml_gallocr_t,
}

impl Session {
    pub fn new(model: &Qwen35, n_ctx_max: usize) -> Result<Session, ModelError> {
        Self::new_spec(model, n_ctx_max, 0)
    }

    /// `n_spec` = how many draft tokens a verify step may need to undo.
    pub fn new_spec(
        model: &Qwen35,
        n_ctx_max: usize,
        n_spec: usize,
    ) -> Result<Session, ModelError> {
        let hp = &model.hp;
        let fa = std::env::var("CODPIECE_NO_FA").is_err();
        let k_slots = (n_spec + 1) as i64;
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
            let f16t = ffi::ggml_type_GGML_TYPE_F16;
            let f32t = ffi::ggml_type_GGML_TYPE_F32;
            let mut k_cache = vec![std::ptr::null_mut(); hp.n_layer];
            let mut v_cache = vec![std::ptr::null_mut(); hp.n_layer];
            let mut conv_state = vec![std::ptr::null_mut(); hp.n_layer];
            let mut gdn_state = vec![std::ptr::null_mut(); hp.n_layer];
            // Names matter: under tensor parallelism the meta device asks the
            // split callback how to divide each allocated tensor, and the
            // callback classifies by name (see split.rs).
            let name_it = |t: *mut ffi::ggml_tensor, n: String| {
                if let Ok(c) = std::ffi::CString::new(n) {
                    ffi::ggml_set_name(t, c.as_ptr());
                }
            };
            for il in 0..hp.n_layer {
                if hp.is_recurrent(il) {
                    conv_state[il] = ffi::ggml_new_tensor_3d(
                        ctx, f32t, hp.d_conv - 1, hp.conv_dim(), k_slots,
                    );
                    gdn_state[il] = ffi::ggml_new_tensor_4d(
                        ctx, f32t, hp.gdn_head_v(), hp.gdn_head_v(), hp.n_v_heads, k_slots,
                    );
                    name_it(conv_state[il], format!("cache_conv_l{il}"));
                    name_it(gdn_state[il], format!("cache_gdn_l{il}"));
                } else {
                    // f16 KV, like production llama.cpp.
                    k_cache[il] = ffi::ggml_new_tensor_2d(
                        ctx, f16t, hp.head_k * hp.n_head_kv, n_ctx_max as i64,
                    );
                    // FA: V stored column-major like K; non-FA: transposed.
                    v_cache[il] = if fa {
                        ffi::ggml_new_tensor_2d(
                            ctx, f16t, hp.head_v * hp.n_head_kv, n_ctx_max as i64,
                        )
                    } else {
                        ffi::ggml_new_tensor_2d(
                            ctx, f16t, n_ctx_max as i64, hp.head_v * hp.n_head_kv,
                        )
                    };
                    name_it(k_cache[il], format!("cache_k_l{il}"));
                    name_it(v_cache[il], format!("cache_v_l{il}"));
                }
            }
            // NOTE: session KV/state tensors all live on device 0 for now.
            // With a layer split the scheduler will copy them to the layer's
            // device each step — correct, but it doubles bus traffic for the
            // second half's layers. Placing them per-layer is the next step
            // (tracked in ROADMAP M3).
            // MTP draft block: one attention layer with its own cache
            let mtp_ctx_max = n_ctx_max.min(mtp_ctx_cap());
            let mtp_k = ffi::ggml_new_tensor_2d(
                ctx, f16t, hp.head_k * hp.n_head_kv, mtp_ctx_max as i64,
            );
            let mtp_v = if fa {
                ffi::ggml_new_tensor_2d(ctx, f16t, hp.head_v * hp.n_head_kv, mtp_ctx_max as i64)
            } else {
                ffi::ggml_new_tensor_2d(ctx, f16t, mtp_ctx_max as i64, hp.head_v * hp.n_head_kv)
            };
            name_it(mtp_k, format!("cache_k_l{}", hp.n_layer));
            name_it(mtp_v, format!("cache_v_l{}", hp.n_layer));

            let buffer = ffi::ggml_backend_alloc_ctx_tensors(ctx, model.weights.backend());
            if buffer.is_null() {
                ffi::ggml_free(ctx);
                return Err(ModelError::Load("session buffer alloc".into()));
            }
            ffi::ggml_backend_buffer_clear(buffer, 0);
            let galloc = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(
                model.weights.backend(),
            ));
            if galloc.is_null() {
                ffi::ggml_backend_buffer_free(buffer);
                ffi::ggml_free(ctx);
                return Err(ModelError::Load("session gallocr".into()));
            }
            Ok(Session {
                n_ctx_max,
                n_past: 0,
                last_taps: Vec::new(),
                rope_off: 0,
                k_slots,
                fa,
                tp: model.weights.is_tensor_parallel(),
                ctx,
                buffer,
                galloc,
                k_cache,
                v_cache,
                conv_state,
                gdn_state,
                mtp_k,
                mtp_v,
                mtp_ctx_max,
                mtp_past: 0,
                mtp_cached: None,
                fused: Vec::new(),
                fused_clock: 0,
                batch_step: None,
                rollback: (0..=k_slots).map(|_| None).collect(),
                graveyard: Vec::new(),
                cached: None,
            })
        }
    }

    fn view(&self) -> SessView {
        SessView {
            k_slots: self.k_slots,
            k_cache: self.k_cache.clone(),
            v_cache: self.v_cache.clone(),
            conv_state: self.conv_state.clone(),
            gdn_state: self.gdn_state.clone(),
            n_ctx_max: self.n_ctx_max,
            fa: self.fa,
        }
    }

    /// Drop every cached fused graph shape, freeing their compute buffers.
    ///
    /// Prefill builds shapes that decoding never uses again, and each one owns a
    /// compute buffer holding a mask of `n_kv x chunk` and logits of `n_vocab x chunk`.
    /// At short context that is noise; at 137K it is gigabytes, and it is what made a
    /// speculative run fail to allocate 317 MiB while plain decoding at the same
    /// context still had room.
    pub fn clear_fused_cache(&mut self) {
        for c in self.fused.drain(..) {
            unsafe { ffi::ggml_gallocr_free(c.galloc) }
            self.graveyard.push(c.built.ctx);
        }
    }

    pub fn reset(&mut self) {
        unsafe {
            ffi::ggml_backend_buffer_clear(self.buffer, 0);
        }
        for r in self.rollback.iter_mut() {
            if let Some(r) = r.take() {
                unsafe {
                    ffi::ggml_gallocr_free(r.galloc);
                    ffi::ggml_free(r.ctx);
                }
            }
        }
        self.cached = None;
        self.mtp_cached = None;
        // Retire the cached graphs. Keeping them across a reset looks like the obvious
        // optimisation — the shapes recur and the session tensors survive — but it
        // crashes the process on the next speculative request, so something in the
        // backend's per-graph state does not survive a reset the way the tensors do.
        // Not chased further; the cost is a graph rebuild per request, and the note is
        // here so the next attempt starts from "this was tried".
        for c in self.fused.drain(..) {
            unsafe { ffi::ggml_gallocr_free(c.galloc) }
            self.graveyard.push(c.built.ctx);
        }
        if let Some(c) = self.batch_step.take() {
            unsafe { ffi::ggml_gallocr_free(c.galloc) }
            self.graveyard.push(c.built.ctx);
        }
        self.n_past = 0;
        self.rope_off = 0;
        self.mtp_past = 0;
    }

    /// The RoPE position of the NEXT row: n_past shifted by the image debt.
    pub fn rope_base(&self) -> usize {
        (self.n_past as i64 + self.rope_off) as usize
    }
}

struct BatchStep {
    built: Built,
    galloc: ffi::ggml_gallocr_t,
    t_len: i64,
    want_logits: bool,
}

struct FusedStep {
    /// whether this shape carries the gumbel-noise input
    gumbel: bool,
    built: Built,
    galloc: ffi::ggml_gallocr_t,
    bucket: i64,
    mtp_bucket: i64,
    t_len: i64,
    /// how many positions the graph emits predictions for: `t_len`, or 1 for a prefill
    n_out: i64,
    n_cand: i64,
    /// Chain length baked into the graph. Independent of `t_len` — the batch carries
    /// however many drafts the PREVIOUS round produced, while this is how many the
    /// current one will produce — so it has to be part of the key in its own right.
    depth: usize,
    used: u64,
}

/// One fused verify+draft round's outputs.
pub struct FusedOut {
    /// the trunk's argmax at each requested position
    pub preds: Vec<u32>,
    /// draft chains: chain[link][position]
    pub chain: Vec<Vec<u32>>,
    /// per-link softmax peak at the drafted token, aligned with `chain`
    pub conf: Vec<Vec<f32>>,
    /// full-vocabulary logits per position when requested, else empty
    pub logits: Vec<f32>,
    /// trunk hidden per position when logits were requested, else empty
    pub hidden: Vec<f32>,
}

/// How many fused graph shapes to keep. Adaptive depth settles on one depth but probes
/// its neighbours, and each probe passes through a transitional shape, so a handful
/// covers the churn. Each entry owns its own compute buffer, so this is a VRAM decision
/// before it is a hit-rate one. Six is enough that a session bounded to three depths
/// never evicts, which matters for more than hit rate: eviction frees a compute buffer
/// the backend may still hold registrations against.
const FUSED_CACHE: usize = 6;


/// A session's state, lifted to host RAM.
///
/// This is the other half of fast conversations: in-session prefix reuse makes the
/// *next* turn of the current conversation cheap, and a snapshot makes *returning* to a
/// different conversation cheap — copy ~2 GiB back over PCIe (~0.1 s) instead of
/// re-prefilling 32K tokens (~24 s). Only the used prefix of each cache is copied:
/// the K/V tensors are allocated at full context, but their first `n_past` columns are
/// contiguous, which is what makes the copy a single `tensor_get` per tensor.
pub struct SessionSnapshot {
    n_past: usize,
    rope_off: i64,
    mtp_past: usize,
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
    conv: Vec<Vec<u8>>,
    gdn: Vec<Vec<u8>>,
    mtp_k: Vec<u8>,
    mtp_v: Vec<u8>,
}

impl SessionSnapshot {
    pub fn nbytes(&self) -> usize {
        self.k.iter().map(Vec::len).sum::<usize>()
            + self.v.iter().map(Vec::len).sum::<usize>()
            + self.conv.iter().map(Vec::len).sum::<usize>()
            + self.gdn.iter().map(Vec::len).sum::<usize>()
            + self.mtp_k.len()
            + self.mtp_v.len()
    }

    pub fn n_past(&self) -> usize {
        self.n_past
    }
}

unsafe fn get_prefix(t: *mut ffi::ggml_tensor, cols: usize) -> Vec<u8> {
    // The per-layer cache vectors are indexed by layer and null for the other layer
    // kind — attention layers have no conv/gdn state, recurrent layers no KV.
    if t.is_null() {
        return Vec::new();
    }
    let bytes = cols * (*t).nb[1];
    let mut buf = vec![0u8; bytes];
    if bytes > 0 {
        ffi::ggml_backend_tensor_get(t, buf.as_mut_ptr().cast(), 0, bytes);
    }
    buf
}

unsafe fn get_all(t: *mut ffi::ggml_tensor) -> Vec<u8> {
    if t.is_null() {
        return Vec::new();
    }
    let bytes = ffi::ggml_nbytes(t);
    let mut buf = vec![0u8; bytes];
    ffi::ggml_backend_tensor_get(t, buf.as_mut_ptr().cast(), 0, bytes);
    buf
}

impl Session {
    /// Copy this session's state to host RAM. `None` when the layout does not support
    /// it (the non-flash-attention V cache stores token columns transposed, so its used
    /// prefix is not contiguous — not worth supporting for a debug path).
    pub fn snapshot(&self) -> Option<SessionSnapshot> {
        if !self.fa {
            return None;
        }
        // Not supported under tensor parallelism: the meta backend's split-tensor
        // get/set handles axes 0 and 1 in three dimensions, and the GDN state is
        // four-dimensional and split on axis 2 (GGML_ASSERT(tensor->ne[3] == 1) at
        // ggml-backend-meta.cpp:1441). The TP path gets conversation switching from
        // the session pool instead — whole sessions resident in VRAM, switched by
        // pointer, which is faster than any copy could be.
        if self.tp {
            return None;
        }
        let whole = false;
        unsafe {
            Some(SessionSnapshot {
                n_past: self.n_past,
                rope_off: self.rope_off,
                mtp_past: self.mtp_past,
                k: self
                    .k_cache
                    .iter()
                    .map(|t| if whole { get_all(*t) } else { get_prefix(*t, self.n_past) })
                    .collect(),
                v: self
                    .v_cache
                    .iter()
                    .map(|t| if whole { get_all(*t) } else { get_prefix(*t, self.n_past) })
                    .collect(),
                conv: self.conv_state.iter().map(|t| get_all(*t)).collect(),
                gdn: self.gdn_state.iter().map(|t| get_all(*t)).collect(),
                mtp_k: if whole {
                    get_all(self.mtp_k)
                } else {
                    get_prefix(self.mtp_k, self.mtp_past)
                },
                mtp_v: if whole {
                    get_all(self.mtp_v)
                } else {
                    get_prefix(self.mtp_v, self.mtp_past)
                },
            })
        }
    }

    /// Load a snapshot back into this session, replacing whatever it held.
    pub fn restore(&mut self, snap: &SessionSnapshot) {
        self.reset();
        unsafe {
            for (t, buf) in self.k_cache.iter().zip(&snap.k) {
                if !t.is_null() && !buf.is_empty() {
                    ffi::ggml_backend_tensor_set(*t, buf.as_ptr().cast(), 0, buf.len());
                }
            }
            for (t, buf) in self.v_cache.iter().zip(&snap.v) {
                if !t.is_null() && !buf.is_empty() {
                    ffi::ggml_backend_tensor_set(*t, buf.as_ptr().cast(), 0, buf.len());
                }
            }
            for (t, buf) in self.conv_state.iter().zip(&snap.conv) {
                if !t.is_null() && !buf.is_empty() {
                    ffi::ggml_backend_tensor_set(*t, buf.as_ptr().cast(), 0, buf.len());
                }
            }
            for (t, buf) in self.gdn_state.iter().zip(&snap.gdn) {
                if !t.is_null() && !buf.is_empty() {
                    ffi::ggml_backend_tensor_set(*t, buf.as_ptr().cast(), 0, buf.len());
                }
            }
            if !snap.mtp_k.is_empty() {
                ffi::ggml_backend_tensor_set(self.mtp_k, snap.mtp_k.as_ptr().cast(), 0, snap.mtp_k.len());
            }
            if !snap.mtp_v.is_empty() {
                ffi::ggml_backend_tensor_set(self.mtp_v, snap.mtp_v.as_ptr().cast(), 0, snap.mtp_v.len());
            }
        }
        self.n_past = snap.n_past;
        self.rope_off = snap.rope_off;
        self.mtp_past = snap.mtp_past;
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        for r in self.rollback.iter_mut() {
            if let Some(r) = r.take() {
                unsafe {
                    ffi::ggml_gallocr_free(r.galloc);
                    ffi::ggml_free(r.ctx);
                }
            }
        }
        for c in self.fused.drain(..) {
            unsafe {
                ffi::ggml_gallocr_free(c.galloc);
                ffi::ggml_free(c.built.ctx);
            }
        }
        if let Some(c) = self.batch_step.take() {
            unsafe {
                ffi::ggml_gallocr_free(c.galloc);
                ffi::ggml_free(c.built.ctx);
            }
        }
        if let Some((_, ga, _)) = self.mtp_cached.take() {
            unsafe { ffi::ggml_gallocr_free(ga) }
        }
        self.cached = None;
        for ctx in self.graveyard.drain(..) {
            unsafe { ffi::ggml_free(ctx) }
        }
        unsafe {
            ffi::ggml_gallocr_free(self.galloc);
            ffi::ggml_backend_buffer_free(self.buffer);
            ffi::ggml_free(self.ctx);
        }
    }
}

impl Qwen35 {
    pub fn load(path: &std::path::Path) -> Result<Qwen35, ModelError> {
        Self::load_on(path, crate::Device::Cpu)
    }

    pub fn load_on(path: &std::path::Path, device: crate::Device) -> Result<Qwen35, ModelError> {
        let weights = Weights::load(path, device)?;
        let hp = Hparams::from_gguf(&weights.gguf)?;
        Ok(Qwen35 {
            tap_layers: Vec::new(), weights, hp })
    }

    fn t(&self, name: &str) -> Result<*mut ffi::ggml_tensor, ModelError> {
        self.weights
            .tensor(name)
            .ok_or_else(|| ModelError::Load(format!("missing tensor {name}")))
    }

    pub(crate) fn t_pub(&self, name: &str) -> Result<*mut ffi::ggml_tensor, ModelError> {
        self.t(name)
    }

    pub(crate) fn layer_pub(&self, il: usize) -> Result<Layer, ModelError> {
        self.layer(il)
    }

    fn layer(&self, il: usize) -> Result<Layer, ModelError> {
        let n = |suffix: &str| format!("blk.{il}.{suffix}");
        let opt = |name: &str| self.weights.tensor(name).unwrap_or(std::ptr::null_mut());
        // The MTP block (il == n_layer) is a dense attention block regardless
        // of where it lands in the recurrent/attention cadence.
        let recurrent = il < self.hp.n_layer && self.hp.is_recurrent(il);
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
        match self.run_general(
            tokens,
            StateSrc::Stateless,
            None,
            0,
            0,
            out_positions,
            n_threads,
            false,
            None,
        )? {
            StepOut::Logits(l) => Ok(l),
            _ => unreachable!("stateless forward returns logits"),
        }
    }

    /// Stateful step: consume `tokens` at positions [session.n_past, ..),
    /// update caches/states, return logits at `out_positions` (ubatch-
    /// relative). Advances session.n_past on success. Single-token FA steps
    /// with out_positions == [0] take the cached-graph fast path.
    pub fn step(
        &self,
        session: &mut Session,
        tokens: &[u32],
        out_positions: &[i32],
        n_threads: i32,
    ) -> Result<Vec<f32>, ModelError> {
        match self.step_impl(session, tokens, out_positions, n_threads, false)? {
            StepOut::Logits(l) => Ok(l),
            _ => unreachable!("logits requested"),
        }
    }

    /// Greedy step: argmax runs IN the graph, so only the chosen token id
    /// crosses the bus. Numerically identical to argmax over step()'s logits.
    pub fn step_greedy(
        &self,
        session: &mut Session,
        tokens: &[u32],
        n_threads: i32,
    ) -> Result<u32, ModelError> {
        let last = [(tokens.len() - 1) as i32];
        let in_graph = self.weights.can_sample_in_graph();
        match self.step_impl(session, tokens, &last, n_threads, in_graph)? {
            StepOut::Token(t) => Ok(t),
            StepOut::Tokens(v) => Ok(*v.last().unwrap()),
            StepOut::Logits(l) => Ok(argmax(&l)),
        }
    }

    fn step_impl(
        &self,
        session: &mut Session,
        tokens: &[u32],
        out_positions: &[i32],
        n_threads: i32,
        greedy: bool,
    ) -> Result<StepOut, ModelError> {
        if session.n_past + tokens.len() > session.n_ctx_max {
            return Err(ModelError::Load(format!(
                "context overflow: {} + {} > {}",
                session.n_past,
                tokens.len(),
                session.n_ctx_max
            )));
        }
        // The cached-graph fast path assumes one backend and a frozen
        // allocation; split models go through the scheduler every step.
        let cacheable = self.weights.sched().is_none();
        let wants_all = out_positions.len() == tokens.len()
            && out_positions.iter().enumerate().all(|(i, p)| *p == i as i32);
        let out = if cacheable && session.fa && (out_positions == [0] || wants_all) {
            let (o, _h, tv) = self.step_cached(session, tokens, n_threads, greedy)?;
            if !tv.is_empty() {
                session.last_taps = tv;
            }
            o
        } else {
            let view = session.view();
            let galloc = session.galloc;
            // odd-shaped step invalidates the frozen decode allocation
            session.cached = None;
            self.run_general(
                tokens,
                StateSrc::Session(view),
                Some(galloc),
                session.n_past,
                session.rope_base(),
                out_positions,
                n_threads,
                greedy,
                Some(&mut session.last_taps),
            )?
        };
        session.n_past += tokens.len();
        Ok(out)
    }

    /// Inject an image chunk: `nx * ny` precomputed trunk embeddings advance
    /// the session exactly like that many prefill tokens, but with Qwen-VL
    /// vision M-RoPE positions — t = pos0 for the whole image, y/x walk the
    /// merged grid — and an intra-image block-visible mask. llama.cpp keeps
    /// causal masking on for these tokens, but it masks by comparing
    /// POSITIONS and every image token shares t = pos0, so the image block is
    /// mutually visible there; this row-indexed mask replicates that
    /// explicitly. The RoPE clock advances by max(nx, ny) while n_past
    /// advances by nx*ny — the growing gap lives in Session::rope_off and
    /// every text-position fill adds it.
    pub fn step_embd(
        &self,
        session: &mut Session,
        embd: &[f32],
        nx: usize,
        ny: usize,
        n_threads: i32,
    ) -> Result<(), ModelError> {
        let t = nx * ny;
        let n_embd = self.hp.n_embd as usize;
        if t == 0 || embd.len() != t * n_embd {
            return Err(ModelError::Load(format!(
                "embd chunk: {} floats for a {nx}x{ny} grid",
                embd.len()
            )));
        }
        if session.n_past + t > session.n_ctx_max {
            return Err(ModelError::Load("context overflow".into()));
        }
        let pos0 = session.rope_base();
        unsafe {
            let n_kv_exact = session.n_past as i64 + t as i64;
            let kvb = kv_bucket();
            let n_kv = (((n_kv_exact + kvb - 1) / kvb) * kvb).min(session.n_ctx_max as i64);
            let built = self.build_inner(
                t as i64,
                n_kv,
                &StateSrc::Session(session.view()),
                1,
                /* use_set_rows */ false,
                session.n_past,
                /* greedy */ false,
                None,
                SeqMode::Single,
                /* embd_input */ true,
                false,
            )?;
            // odd-shaped step: the frozen decode allocation no longer holds
            session.cached = None;
            let galloc = session.galloc;
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _guard = G(built.ctx);
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, built.gf) {
                    return Err(ModelError::Load("embd sched alloc".into()));
                }
            } else if !ffi::ggml_gallocr_alloc_graph(galloc, built.gf) {
                return Err(ModelError::Load("embd graph alloc".into()));
            }

            ffi::ggml_backend_tensor_set(
                built.inp_embd,
                embd.as_ptr().cast(),
                0,
                embd.len() * 4,
            );
            let mut pos = vec![0i32; t * 4];
            for i in 0..t {
                pos[i] = pos0 as i32;
                pos[t + i] = (pos0 + i / nx) as i32;
                pos[2 * t + i] = (pos0 + i % nx) as i32;
            }
            ffi::ggml_backend_tensor_set(built.inp_pos, pos.as_ptr().cast(), 0, pos.len() * 4);

            let nkv = built.n_kv as usize;
            let vis = (session.n_past + t).min(nkv);
            if built.fa_mask {
                let mut mask = vec![0xFC00u16; nkv * t];
                for q in 0..t {
                    for kv in 0..vis {
                        mask[q * nkv + kv] = 0;
                    }
                }
                ffi::ggml_backend_tensor_set(built.kq_mask, mask.as_ptr().cast(), 0, mask.len() * 2);
            } else {
                let mut mask = vec![f32::NEG_INFINITY; nkv * t];
                for q in 0..t {
                    for kv in 0..vis {
                        mask[q * nkv + kv] = 0.0;
                    }
                }
                ffi::ggml_backend_tensor_set(built.kq_mask, mask.as_ptr().cast(), 0, mask.len() * 4);
            }
            let out_ids = [(t - 1) as i32];
            ffi::ggml_backend_tensor_set(built.out_ids, out_ids.as_ptr().cast(), 0, 4);
            self.compute(built.gf, n_threads)?;
            let tv = self.read_taps(&built, t);
            if !tv.is_empty() {
                session.last_taps = tv;
            }
        }
        session.n_past += t;
        session.rope_off += nx.max(ny) as i64 - t as i64;
        Ok(())
    }

    /// One-shot path: build graph, allocate (scratch or throwaway), fill,
    /// compute, read.
    fn run_general(
        &self,
        tokens: &[u32],
        state: StateSrc,
        galloc: Option<ffi::ggml_gallocr_t>,
        n_past: usize,
        rope_base: usize,
        out_positions: &[i32],
        n_threads: i32,
        greedy: bool,
        taps_out: Option<&mut Vec<f32>>,
    ) -> Result<StepOut, ModelError> {
        assert!(!tokens.is_empty());
        assert!(!out_positions.is_empty());
        unsafe {
            let built = self.build(
                tokens.len() as i64,
                n_past,
                &state,
                out_positions.len() as i64,
                false,
                greedy,
                None,
            )?;
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
            let mut guard = CtxGuard(built.ctx, std::ptr::null_mut());
            if let Some(sched) = self.weights.sched() {
                // The scheduler allocates split graphs itself; a single
                // gallocr cannot, because the nodes span several buffers.
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, built.gf) {
                    return Err(ModelError::Load("sched graph alloc failed".into()));
                }
            } else {
                let galloc = match galloc {
                    Some(g) => g,
                    None => {
                        let g = ffi::ggml_gallocr_new(
                            ffi::ggml_backend_get_default_buffer_type(self.weights.backend()),
                        );
                        guard.1 = g;
                        g
                    }
                };
                if !ffi::ggml_gallocr_alloc_graph(galloc, built.gf) {
                    return Err(ModelError::Load("graph alloc failed".into()));
                }
            }
            self.fill_inputs(&built, tokens, n_past, rope_base, out_positions);
            self.compute(built.gf, n_threads)?;
            if let Some(o) = taps_out {
                *o = self.read_taps(&built, tokens.len());
            }
            Ok(self.read_out(&built, out_positions.len()))
        }
    }

    unsafe fn read_out(&self, b: &Built, n_out: usize) -> StepOut {
        if b.greedy {
            let mut ids = vec![0i32; n_out];
            ffi::ggml_backend_tensor_get(b.out, ids.as_mut_ptr().cast(), 0, n_out * 4);
            if n_out > 1 {
                return StepOut::Tokens(ids.into_iter().map(|v| v as u32).collect());
            }
            StepOut::Token(ids[n_out - 1] as u32)
        } else {
            let mut logits = vec![0f32; self.hp.n_vocab as usize * n_out];
            ffi::ggml_backend_tensor_get(b.out, logits.as_mut_ptr().cast(), 0, logits.len() * 4);
            StepOut::Logits(logits)
        }
    }

    /// Undo `n` tokens of recurrent-state advance by promoting snapshot slot
    /// `n` to slot 0. KV/attention needs no undo — rewinding `n_past` is
    /// enough, since stale entries are simply overwritten.
    pub fn rollback_recurrent(
        &self,
        session: &mut Session,
        n: usize,
        n_threads: i32,
    ) -> Result<(), ModelError> {
        if n == 0 {
            return Ok(());
        }
        if n as i64 >= session.k_slots {
            return Err(ModelError::Load(format!(
                "rollback of {n} exceeds {} snapshot slots",
                session.k_slots
            )));
        }
        let hp = &self.hp;
        unsafe {
            if let Some(r) = self.rollback_slot(session, n) {
                return self.compute(r, n_threads);
            }

            // Per recurrent layer: 2 copies plus 4 views — and a view is a
            // NODE in ggml, not a leaf. The 27B needs 48*6 = 288; sizing this
            // by hand is what overflowed the graph before.
            let n_recr = (0..hp.n_layer).filter(|&il| hp.is_recurrent(il)).count();
            let n_nodes = n_recr * 8 + 32;
            let params = ffi::ggml_init_params {
                mem_size: (n_recr * 8 + 32) * ffi::ggml_tensor_overhead()
                    + ffi::ggml_graph_overhead_custom(n_nodes, false),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(ModelError::Load("rollback ctx".into()));
            }
            let gf = ffi::ggml_new_graph_custom(ctx, n_nodes, false);
            for il in 0..hp.n_layer {
                if !hp.is_recurrent(il) {
                    continue;
                }
                // Views must keep the tensor's own dimensions: under tensor
                // parallelism these states are split (conv on ne[1], gdn on
                // ne[2]), and a flattened 1-D view erases the axis the meta
                // backend needs to follow. Only the slot offset differs.
                let cs = session.conv_state[il];
                let gs = session.gdn_state[il];
                let conv_src = ffi::ggml_view_3d(
                    ctx, cs, (*cs).ne[0], (*cs).ne[1], 1,
                    (*cs).nb[1], (*cs).nb[2], n * (*cs).nb[2],
                );
                let conv_dst = ffi::ggml_view_3d(
                    ctx, cs, (*cs).ne[0], (*cs).ne[1], 1,
                    (*cs).nb[1], (*cs).nb[2], 0,
                );
                ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, conv_src, conv_dst));

                let gdn_src = ffi::ggml_view_4d(
                    ctx, gs, (*gs).ne[0], (*gs).ne[1], (*gs).ne[2], 1,
                    (*gs).nb[1], (*gs).nb[2], (*gs).nb[3], n * (*gs).nb[3],
                );
                let gdn_dst = ffi::ggml_view_4d(
                    ctx, gs, (*gs).ne[0], (*gs).ne[1], (*gs).ne[2], 1,
                    (*gs).nb[1], (*gs).nb[2], (*gs).nb[3], 0,
                );
                ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, gdn_src, gdn_dst));
            }
            // A dedicated allocator: sharing the session scratch allocator with the
            // fused graph would hand both the same arena and invalidate whichever was
            // allocated first.
            let galloc = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(
                self.weights.backend(),
            ));
            if galloc.is_null() {
                ffi::ggml_free(ctx);
                return Err(ModelError::Load("rollback galloc".into()));
            }
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, gf) {
                    ffi::ggml_gallocr_free(galloc);
                    ffi::ggml_free(ctx);
                    return Err(ModelError::Load("rollback sched alloc".into()));
                }
            } else if !ffi::ggml_gallocr_alloc_graph(galloc, gf) {
                ffi::ggml_gallocr_free(galloc);
                ffi::ggml_free(ctx);
                return Err(ModelError::Load("rollback alloc".into()));
            }
            if self.weights.sched().is_none() {
                ffi::ggml_graph_set_new_uid(gf);
            }
            self.compute(gf, n_threads)?;
            // Only the allocator path may be replayed: a scheduler-allocated graph is
            // invalidated by the next ggml_backend_sched_reset.
            if self.weights.sched().is_none() && session.rollback.get(n).is_some() {
                session.rollback[n] = Some(RollbackGraph { ctx, gf, galloc });
            } else {
                ffi::ggml_gallocr_free(galloc);
                ffi::ggml_free(ctx);
            }
        }
        Ok(())
    }

    /// The cached rollback graph for distance `n`, if it has been built already.
    fn rollback_slot(
        &self,
        session: &Session,
        n: usize,
    ) -> Option<*mut ffi::ggml_cgraph> {
        session.rollback.get(n).and_then(|s| s.as_ref()).map(|r| r.gf)
    }

    /// Trunk step that also returns the pre-LM-head hidden state, which the
    /// MTP draft head consumes. Same graph as `step` — the hidden was always
    /// computed, it just was not readable before.
    pub fn step_with_hidden(
        &self,
        session: &mut Session,
        tokens: &[u32],
        out_positions: &[i32],
        n_threads: i32,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        if session.n_past + tokens.len() > session.n_ctx_max {
            return Err(ModelError::Load("context overflow".into()));
        }
        let view = session.view();
        let galloc = session.galloc;
        session.cached = None;
        let out = unsafe {
            let built = self.build(
                tokens.len() as i64,
                session.n_past,
                &StateSrc::Session(view),
                out_positions.len() as i64,
                false,
                false,
                None,
            )?;
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _g = G(built.ctx);
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, built.gf) {
                    return Err(ModelError::Load("sched alloc".into()));
                }
            } else if !ffi::ggml_gallocr_alloc_graph(galloc, built.gf) {
                return Err(ModelError::Load("graph alloc".into()));
            }
            self.fill_inputs(&built, tokens, session.n_past, session.rope_base(), out_positions);
            self.compute(built.gf, n_threads)?;
            let n_out = out_positions.len();
            let mut logits = vec![0f32; self.hp.n_vocab as usize * n_out];
            ffi::ggml_backend_tensor_get(built.out, logits.as_mut_ptr().cast(), 0, logits.len() * 4);
            let mut hidden = vec![0f32; self.hp.n_embd as usize * n_out];
            ffi::ggml_backend_tensor_get(
                built.h_out,
                hidden.as_mut_ptr().cast(),
                0,
                hidden.len() * 4,
            );
            (logits, hidden)
        };
        session.n_past += tokens.len();
        Ok(out)
    }

    /// Verify step: consume `tokens` and return the token the model itself
    /// predicts at every position, plus the hidden states.
    ///
    /// When sampling can run in the graph, the readback is 4 bytes per
    /// position instead of n_vocab floats — 3.9 MB each on the 27B. That is
    /// what makes it cheap to extend a speculative round with extra
    /// candidate tokens, since the verify pass itself is bandwidth-bound and
    /// nearly indifferent to how many tokens ride along.
    pub fn step_verify(
        &self,
        session: &mut Session,
        tokens: &[u32],
        n_threads: i32,
    ) -> Result<(Vec<u32>, Vec<f32>), ModelError> {
        if session.n_past + tokens.len() > session.n_ctx_max {
            return Err(ModelError::Load("context overflow".into()));
        }
        let out_positions: Vec<i32> = (0..tokens.len() as i32).collect();
        let greedy = self.weights.can_sample_in_graph();

        // A speculative round has a FIXED shape, so its graph can be built
        // once per KV bucket and replayed like the T=1 decode graph. That
        // matters more than it sounds: measured on the 27B, a round that
        // rebuilds the 64-layer trunk graph costs ~38 ms against ~26 ms for
        // a replayed one — the rebuild, not the drafting, was the overhead.
        // Not under tensor parallelism: `spec` interleaves this verify with
        // per-draft graph executions, and the meta backend aborts when a
        // cached graph is replayed with other graphs in between. Rebuilding
        // is the correct choice there until that is resolved.
        // Every failing TP experiment so far mixed a CACHED graph with a
        // REBUILT one. Two cached graphs alternating has not been tried, and
        // it is the configuration that would actually pay: CODPIECE_BOTH_CACHED
        // turns it on for both the verify and the draft.
        let both_cached = std::env::var("CODPIECE_BOTH_CACHED").is_ok();
        if session.fa && (!self.weights.is_tensor_parallel() || both_cached) {
            let (out, hidden, tv) = self.step_cached(session, tokens, n_threads, greedy)?;
            if !tv.is_empty() {
                session.last_taps = tv;
            }
            session.n_past += tokens.len();
            let preds = match out {
                StepOut::Tokens(ids) => ids,
                StepOut::Token(t) => vec![t],
                StepOut::Logits(l) => {
                    let nv = self.hp.n_vocab as usize;
                    (0..tokens.len())
                        .map(|i| argmax(&l[i * nv..(i + 1) * nv]))
                        .collect()
                }
            };
            return Ok((preds, hidden));
        }

        let view = session.view();
        let galloc = session.galloc;
        session.cached = None;
        let out = unsafe {
            let built = self.build(
                tokens.len() as i64,
                session.n_past,
                &StateSrc::Session(view),
                out_positions.len() as i64,
                false,
                greedy,
                None,
            )?;
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _g = G(built.ctx);
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, built.gf) {
                    return Err(ModelError::Load("sched alloc".into()));
                }
            } else if !ffi::ggml_gallocr_alloc_graph(galloc, built.gf) {
                return Err(ModelError::Load("graph alloc".into()));
            }
            self.fill_inputs(&built, tokens, session.n_past, session.rope_base(), &out_positions);
            self.compute(built.gf, n_threads)?;

            let n_out = out_positions.len();
            let preds: Vec<u32> = if greedy {
                let mut ids = vec![0i32; n_out];
                ffi::ggml_backend_tensor_get(built.out, ids.as_mut_ptr().cast(), 0, n_out * 4);
                ids.into_iter().map(|v| v as u32).collect()
            } else {
                let mut logits = vec![0f32; self.hp.n_vocab as usize * n_out];
                ffi::ggml_backend_tensor_get(
                    built.out,
                    logits.as_mut_ptr().cast(),
                    0,
                    logits.len() * 4,
                );
                (0..n_out)
                    .map(|i| {
                        argmax(&logits[i * self.hp.n_vocab as usize..(i + 1) * self.hp.n_vocab as usize])
                    })
                    .collect()
            };
            let mut hidden = vec![0f32; self.hp.n_embd as usize * n_out];
            ffi::ggml_backend_tensor_get(
                built.h_out,
                hidden.as_mut_ptr().cast(),
                0,
                hidden.len() * 4,
            );
            (preds, hidden)
        };
        session.n_past += tokens.len();
        Ok(out)
    }

    /// Cached fused round: ONE graph, built once per (KV bucket, MTP bucket)
    /// and replayed.
    ///
    /// This shape exists because of a hard constraint measured on this
    /// machine: under tensor parallelism the meta backend cannot replay a
    /// cached graph when other graphs are computed in between — interleaving
    /// a cached verify with per-draft graphs aborts. Folding the draft head
    /// into the verify graph leaves exactly one graph per round, which is
    /// both cacheable and free of that interleaving.
    /// One fused verify+draft round.
    ///
    /// `last_only` asks for predictions at just the final position instead of every
    /// one. A decode round needs them all — that is what verification compares against
    /// — but a prefill has nothing to verify, and asking for all of them materialises a
    /// logits tensor of `n_vocab x n_prompt`: 26 GiB on a 27K-token prompt, which is
    /// simply an allocation failure. It is the reason speculative decoding worked to
    /// about 9K tokens of context and not beyond.
    pub fn step_fused_cached(
        &self,
        session: &mut Session,
        tokens: &[u32],
        depth: usize,
        cands: Option<&[i32]>,
        last_only: bool,
        want_logits: bool,
        // T-scaled gumbel noise, [n_vocab * n_out] row-major per position:
        // turns the in-graph argmax into an exact temperature sample the
        // chain conditions on. None keeps plain argmax (greedy / prefill).
        noise_t: Option<&[f32]>,
        n_threads: i32,
    ) -> Result<FusedOut, ModelError> {
        let n_cand = cands.map(|c| c.len() as i64).unwrap_or(0);
        let gumbel = noise_t.is_some();
        if let Some(nz) = noise_t {
            let want = self.hp.n_vocab as usize * if last_only { 1 } else { tokens.len() };
            if nz.len() != want {
                return Err(ModelError::Load(format!(
                    "gumbel noise: {} floats, expected {want}",
                    nz.len()
                )));
            }
        }
        if session.n_past + tokens.len() > session.n_ctx_max {
            return Err(ModelError::Load("context overflow".into()));
        }
        if !self.weights.can_sample_in_graph() {
            return Err(ModelError::Load("fused round needs in-graph sampling".into()));
        }
        let t_len = tokens.len() as i64;
        let n_out = if last_only { 1 } else { t_len };
        let kvb = kv_bucket();
        let bucket = ((((session.n_past as i64 + t_len) + kvb - 1) / kvb) * kvb)
            .min(session.n_ctx_max as i64);
        // The draft head's window is bounded, so when it fills, drop its history and
        // start again. Only draft quality is affected — the verifier is unchanged — and
        // acceptance recovers within a few rounds.
        if session.mtp_past + tokens.len() > session.mtp_ctx_max {
            session.mtp_past = 0;
        }
        let mtp_bucket = ((((session.mtp_past as i64 + t_len) + kvb - 1) / kvb) * kvb)
            .min(session.mtp_ctx_max as i64);

        unsafe {
            session.fused_clock += 1;
            let hit = session.fused.iter().position(|c| {
                c.bucket == bucket
                    && c.mtp_bucket == mtp_bucket
                    && c.t_len == t_len
                    && c.n_out == n_out
                    && c.n_cand == n_cand
                    && c.depth == depth
                    && c.gumbel == gumbel
            });
            let idx = match hit {
                Some(i) => {
                    session.fused[i].used = session.fused_clock;
                    i
                }
                None => {
                    if session.fused.len() >= FUSED_CACHE {
                        let victim = session
                            .fused
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, c)| c.used)
                            .map(|(i, _)| i)
                            .unwrap();
                        let c = session.fused.remove(victim);
                        ffi::ggml_gallocr_free(c.galloc);
                        session.graveyard.push(c.built.ctx);
                    }
                    let tail = MtpTail {
                        k_cache: session.mtp_k,
                        v_cache: session.mtp_v,
                        n_ctx_max: session.n_ctx_max,
                        n_past: 0,
                        n_kv: mtp_bucket,
                        depth,
                        n_cand,
                    };
                    let built = self.build_inner(
                        t_len,
                        bucket,
                        &StateSrc::Session(session.view()),
                        n_out,
                        /* use_set_rows */ true,
                        0,
                        /* greedy */ true,
                        Some(tail),
                        SeqMode::Single,
                        false,
                        gumbel,
                    )?;
                    let ga = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(
                        self.weights.backend(),
                    ));
                    if ga.is_null() || !ffi::ggml_gallocr_alloc_graph(ga, built.gf) {
                        return Err(ModelError::Load(format!(
                            "fused graph alloc failed (t_len {t_len}, depth {depth}, \
                             {} shapes cached) — likely out of device memory",
                            session.fused.len()
                        )));
                    }
                    trace_mem("fused", ga, bucket, t_len);
                    // Frozen from here on: give it an identity so the backend can tell
                    // a replay from a new graph and skip rebuilding its per-device map.
                    ffi::ggml_graph_set_new_uid(built.gf);
                    session.fused.push(FusedStep {
                        gumbel,
                        built,
                        galloc: ga,
                        bucket,
                        mtp_bucket,
                        t_len,
                        n_out,
                        n_cand,
                        depth,
                        used: session.fused_clock,
                    });
                    session.fused.len() - 1
                }
            };
            let c = &session.fused[idx];
            let b = &c.built;
            let t = tokens.len();
            let no = n_out as usize;
            // The draft head runs on the positions the trunk emits, so with `last_only`
            // it sees one position and writes one KV entry — its own cache stays
            // compacted while RoPE below still uses the true sequence positions.
            let first_out = t - no;

            let out_positions: Vec<i32> = (first_out..t).map(|i| i as i32).collect();
            self.fill_inputs(b, tokens, session.n_past, session.rope_base(), &out_positions);
            if let Some(nz) = noise_t {
                ffi::ggml_backend_tensor_set(b.inp_gumbel, nz.as_ptr().cast(), 0, nz.len() * 4);
            }
            let rows: Vec<i64> = (0..t).map(|i| (session.n_past + i) as i64).collect();
            ffi::ggml_backend_tensor_set(b.row_ids, rows.as_ptr().cast(), 0, t * 8);

            let mut mpos = vec![0i32; no * 4];
            for i in 0..no {
                let p = (session.rope_base() + first_out + i + 1) as i32;
                mpos[i] = p;
                mpos[no + i] = p;
                mpos[2 * no + i] = p;
            }
            ffi::ggml_backend_tensor_set(b.mtp_pos, mpos.as_ptr().cast(), 0, mpos.len() * 4);
            let mrows: Vec<i64> = (0..no).map(|i| (session.mtp_past + i) as i64).collect();
            ffi::ggml_backend_tensor_set(b.mtp_rows, mrows.as_ptr().cast(), 0, no * 8);

            let nkv = b.mtp_n_kv as usize;
            let mut mask = vec![0xFC00u16; nkv * no];
            for q in 0..no {
                for kv in 0..=(session.mtp_past + q).min(nkv - 1) {
                    mask[q * nkv + kv] = 0;
                }
            }
            ffi::ggml_backend_tensor_set(b.mtp_mask, mask.as_ptr().cast(), 0, mask.len() * 2);
            if let (Some(c), false) = (cands, b.cand_ids.is_null()) {
                ffi::ggml_backend_tensor_set(b.cand_ids, c.as_ptr().cast(), 0, c.len() * 4);
            }

            self.compute(b.gf, n_threads)?;

            let mut preds = vec![0i32; no];
            ffi::ggml_backend_tensor_get(b.out, preds.as_mut_ptr().cast(), 0, no * 4);
            let taps_v = self.read_taps(b, t);
            let mut conf: Vec<Vec<f32>> = Vec::new();
            for tnsr in &b.draft_conf {
                let mut c = vec![0f32; no];
                ffi::ggml_backend_tensor_get(*tnsr, c.as_mut_ptr().cast(), 0, no * 4);
                conf.push(c);
            }
            let mut chain: Vec<Vec<u32>> = Vec::new();
            for tnsr in std::iter::once(b.draft_out).chain(b.draft_chain.iter().copied()) {
                let mut ids = vec![0i32; no];
                ffi::ggml_backend_tensor_get(tnsr, ids.as_mut_ptr().cast(), 0, no * 4);
                // with a shortlist the argmax indexes the shortlist, not the vocab
                chain.push(match cands {
                    Some(c) => ids
                        .into_iter()
                        .map(|v| *c.get(v as usize).unwrap_or(&0) as u32)
                        .collect(),
                    None => ids.into_iter().map(|v| v as u32).collect(),
                });
            }
            // Only read the vocabulary back when a caller needs the distribution: it is
            // n_vocab x n_out floats, ~4 MB at four positions, and greedy rounds have no
            // use for it. The trunk hidden rides along for the same callers — it is what
            // a post-commit re-draft feeds the MTP head when a sampled token diverges
            // from the argmax the in-graph chain assumed.
            let mut logits = Vec::new();
            let mut hidden = Vec::new();
            if want_logits {
                logits = vec![0f32; self.hp.n_vocab as usize * no];
                ffi::ggml_backend_tensor_get(
                    b.logits,
                    logits.as_mut_ptr().cast(),
                    0,
                    logits.len() * 4,
                );
                hidden = vec![0f32; self.hp.n_embd as usize * no];
                ffi::ggml_backend_tensor_get(
                    b.h_out,
                    hidden.as_mut_ptr().cast(),
                    0,
                    hidden.len() * 4,
                );
            }

            session.n_past += t;
            if !taps_v.is_empty() {
                session.last_taps = taps_v;
            }
            Ok(FusedOut {
                preds: preds.into_iter().map(|v| v as u32).collect(),
                chain,
                conf,
                logits,
                hidden,
            })
        }
    }

    /// Verify `tokens` AND produce the next round's draft candidates in a
    /// single graph execution.
    ///
    /// Returns (predictions, drafts): `predictions[i]` is what the model
    /// itself says follows position i, and `drafts[i]` is the draft head's
    /// guess for the token after that — i.e. the draft to use if exactly the
    /// first `i` proposals were accepted. The caller picks the one matching
    /// the accept count it computes, so a whole graph execution per round
    /// disappears.
    pub fn step_verify_drafting(
        &self,
        session: &mut Session,
        tokens: &[u32],
        depth: usize,
        cands: Option<&[i32]>,
        n_threads: i32,
    ) -> Result<(Vec<u32>, Vec<Vec<u32>>), ModelError> {
        let n_cand = cands.map(|c| c.len() as i64).unwrap_or(0);
        if session.n_past + tokens.len() > session.n_ctx_max {
            return Err(ModelError::Load("context overflow".into()));
        }
        if !self.weights.can_sample_in_graph() {
            return Err(ModelError::Load(
                "fused drafting needs in-graph sampling (see split.rs)".into(),
            ));
        }
        let n_out = tokens.len();
        let out_positions: Vec<i32> = (0..n_out as i32).collect();
        let kvb = kv_bucket();
        if session.mtp_past + n_out > session.mtp_ctx_max {
            session.mtp_past = 0;
        }
        let mtp_kv_exact = (session.mtp_past + n_out) as i64;
        let mtp_n_kv =
            (((mtp_kv_exact + kvb - 1) / kvb) * kvb).min(session.mtp_ctx_max as i64);

        let view = session.view();
        let galloc = session.galloc;
        session.cached = None;
        let out = unsafe {
            let tail = MtpTail {
                k_cache: session.mtp_k,
                v_cache: session.mtp_v,
                n_ctx_max: session.n_ctx_max,
                n_past: session.mtp_past,
                n_kv: mtp_n_kv,
                depth,
                n_cand,
            };
            let built = self.build(
                n_out as i64,
                session.n_past,
                &StateSrc::Session(view),
                n_out as i64,
                false,
                true,
                Some(tail),
            )?;
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _g = G(built.ctx);
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, built.gf) {
                    return Err(ModelError::Load("sched alloc".into()));
                }
            } else if !ffi::ggml_gallocr_alloc_graph(galloc, built.gf) {
                return Err(ModelError::Load("graph alloc".into()));
            }
            self.fill_inputs(&built, tokens, session.n_past, session.rope_base(), &out_positions);

            // the draft head's own positions and causal window
            let mut mpos = vec![0i32; n_out * 4];
            for i in 0..n_out {
                let p = (session.rope_base() + i + 1) as i32;
                mpos[i] = p;
                mpos[n_out + i] = p;
                mpos[2 * n_out + i] = p;
            }
            ffi::ggml_backend_tensor_set(built.mtp_pos, mpos.as_ptr().cast(), 0, mpos.len() * 4);

            let nkv = built.mtp_n_kv as usize;
            let mut mask16 = vec![0xFC00u16; nkv * n_out];
            let mut mask32 = vec![f32::NEG_INFINITY; nkv * n_out];
            let f16 = (*built.mtp_mask).type_ == ffi::ggml_type_GGML_TYPE_F16;
            for q in 0..n_out {
                for kv in 0..=(session.mtp_past + q).min(nkv - 1) {
                    if f16 {
                        mask16[q * nkv + kv] = 0;
                    } else {
                        mask32[q * nkv + kv] = 0.0;
                    }
                }
            }
            if f16 {
                ffi::ggml_backend_tensor_set(
                    built.mtp_mask, mask16.as_ptr().cast(), 0, mask16.len() * 2,
                );
            } else {
                ffi::ggml_backend_tensor_set(
                    built.mtp_mask, mask32.as_ptr().cast(), 0, mask32.len() * 4,
                );
            }

            if let (Some(c), false) = (cands, built.cand_ids.is_null()) {
                ffi::ggml_backend_tensor_set(built.cand_ids, c.as_ptr().cast(), 0, c.len() * 4);
            }

            self.compute(built.gf, n_threads)?;

            let mut preds = vec![0i32; n_out];
            ffi::ggml_backend_tensor_get(built.out, preds.as_mut_ptr().cast(), 0, n_out * 4);
            let mut chain: Vec<Vec<u32>> = Vec::new();
            for tnsr in std::iter::once(built.draft_out)
                .chain(built.draft_chain.iter().copied())
            {
                let mut ids = vec![0i32; n_out];
                ffi::ggml_backend_tensor_get(tnsr, ids.as_mut_ptr().cast(), 0, n_out * 4);
                chain.push(match cands {
                    Some(c) => ids
                        .into_iter()
                        .map(|v| *c.get(v as usize).unwrap_or(&0) as u32)
                        .collect(),
                    None => ids.into_iter().map(|v| v as u32).collect(),
                });
            }
            (preds.into_iter().map(|v| v as u32).collect::<Vec<u32>>(), chain)
        };
        session.n_past += tokens.len();
        Ok(out)
    }

    /// One MTP draft: given the trunk hidden for position p and the token at
    /// p+1, predict the token at p+2. Advances the MTP block's own cache.
    pub fn mtp_draft(
        &self,
        session: &mut Session,
        hidden: &[f32],
        token: u32,
        pos: usize,
        n_threads: i32,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        let n_kv_exact = (session.mtp_past + 1) as i64;
        let kvb = kv_bucket();
        let n_kv = (((n_kv_exact + kvb - 1) / kvb) * kvb).min(session.n_ctx_max as i64);
        // The cached draft graph works everywhere now. Under tensor parallelism it was
        // blocked by the meta backend's per-graph container aliasing
        // (ggml-backend-meta.cpp:1838); the uid-keyed containers fixed that, and the
        // cached path measures lossless and faster (48.8 -> 50.0 tok/s on the spec
        // path). CODPIECE_MTP_UNCACHED=1 restores the rebuild-per-draft behaviour as a
        // bisect lever.
        let cacheable = std::env::var("CODPIECE_MTP_UNCACHED").is_err();
        unsafe {
            if !cacheable {
                return self.mtp_draft_uncached(session, hidden, token, pos, n_threads);
            }
            let stale = session
                .mtp_cached
                .as_ref()
                .map(|(_, _, b)| *b != n_kv)
                .unwrap_or(true);
            if stale {
                if let Some((_, ga, _)) = session.mtp_cached.take() {
                    ffi::ggml_gallocr_free(ga);
                }
                let g = self.build_mtp(
                    1,
                    n_kv,
                    0, // set_rows-free: writes go through row offsets below
                    session.mtp_k,
                    session.mtp_v,
                    session.n_ctx_max,
                    session.fa,
                )?;
                let ga = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(
                    self.weights.backend(),
                ));
                if ga.is_null() || !ffi::ggml_gallocr_alloc_graph(ga, g.gf) {
                    return Err(ModelError::Load("mtp graph alloc".into()));
                }
                // frozen shape: let the backend recognise replays (uid 0 would force a
                // per-device rebuild every call)
                ffi::ggml_graph_set_new_uid(g.gf);
                session.mtp_cached = Some((g, ga, n_kv));
            }
            let (g, _, _) = session.mtp_cached.as_ref().unwrap();

            let tok = [token as i32];
            ffi::ggml_backend_tensor_set(g.inp_tokens, tok.as_ptr().cast(), 0, 4);
            ffi::ggml_backend_tensor_set(
                g.inp_h,
                hidden.as_ptr().cast(),
                0,
                self.hp.n_embd as usize * 4,
            );
            let p = pos as i32;
            let posv = [p, p, p, 0];
            ffi::ggml_backend_tensor_set(g.inp_pos, posv.as_ptr().cast(), 0, 16);
            let nkv = n_kv as usize;
            if g.fa_mask {
                let mut mask = vec![0xFC00u16; nkv];
                for c in mask.iter_mut().take((session.mtp_past + 1).min(nkv)) {
                    *c = 0;
                }
                ffi::ggml_backend_tensor_set(g.kq_mask, mask.as_ptr().cast(), 0, nkv * 2);
            } else {
                let mut mask = vec![f32::NEG_INFINITY; nkv];
                for c in mask.iter_mut().take((session.mtp_past + 1).min(nkv)) {
                    *c = 0.0;
                }
                ffi::ggml_backend_tensor_set(g.kq_mask, mask.as_ptr().cast(), 0, nkv * 4);
            }
            let zero = [0i32];
            ffi::ggml_backend_tensor_set(g.out_ids, zero.as_ptr().cast(), 0, 4);
            ffi::ggml_backend_tensor_set(
                g.row_ids,
                [session.mtp_past as i64].as_ptr().cast(),
                0,
                8,
            );

            self.compute(g.gf, n_threads)?;
            let mut logits = vec![0f32; self.hp.n_vocab as usize];
            ffi::ggml_backend_tensor_get(g.out, logits.as_mut_ptr().cast(), 0, logits.len() * 4);
            let mut h = vec![0f32; self.hp.n_embd as usize];
            ffi::ggml_backend_tensor_get(g.h_out, h.as_mut_ptr().cast(), 0, h.len() * 4);
            session.mtp_past += 1;
            Ok((logits, h))
        }
    }

    /// Draft with a freshly built graph. Slower, but the only path the
    /// tensor-parallel backend accepts today.
    fn mtp_draft_uncached(
        &self,
        session: &mut Session,
        hidden: &[f32],
        token: u32,
        pos: usize,
        n_threads: i32,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        let n_kv_exact = (session.mtp_past + 1) as i64;
        let kvb = kv_bucket();
        let n_kv = (((n_kv_exact + kvb - 1) / kvb) * kvb).min(session.n_ctx_max as i64);
        unsafe {
            let g = self.build_mtp_at(
                1,
                n_kv,
                session.mtp_past,
                session.mtp_k,
                session.mtp_v,
                session.n_ctx_max,
                session.fa,
            )?;
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, g.gf) {
                    return Err(ModelError::Load("mtp sched alloc".into()));
                }
            } else if !ffi::ggml_gallocr_alloc_graph(session.galloc, g.gf) {
                return Err(ModelError::Load("mtp graph alloc".into()));
            }
            self.fill_mtp_inputs(&g, hidden, token, pos, session.mtp_past, n_kv);
            self.compute(g.gf, n_threads)?;
            let mut logits = vec![0f32; self.hp.n_vocab as usize];
            ffi::ggml_backend_tensor_get(g.out, logits.as_mut_ptr().cast(), 0, logits.len() * 4);
            let mut h = vec![0f32; self.hp.n_embd as usize];
            ffi::ggml_backend_tensor_get(g.h_out, h.as_mut_ptr().cast(), 0, h.len() * 4);
            session.mtp_past += 1;
            Ok((logits, h))
        }
    }

    unsafe fn fill_mtp_inputs(
        &self,
        g: &crate::mtp_graph::MtpGraph,
        hidden: &[f32],
        token: u32,
        pos: usize,
        mtp_past: usize,
        n_kv: i64,
    ) {
        let tok = [token as i32];
        ffi::ggml_backend_tensor_set(g.inp_tokens, tok.as_ptr().cast(), 0, 4);
        ffi::ggml_backend_tensor_set(
            g.inp_h,
            hidden.as_ptr().cast(),
            0,
            self.hp.n_embd as usize * 4,
        );
        let p = pos as i32;
        let posv = [p, p, p, 0];
        ffi::ggml_backend_tensor_set(g.inp_pos, posv.as_ptr().cast(), 0, 16);
        let nkv = n_kv as usize;
        if g.fa_mask {
            let mut mask = vec![0xFC00u16; nkv];
            for c in mask.iter_mut().take((mtp_past + 1).min(nkv)) {
                *c = 0;
            }
            ffi::ggml_backend_tensor_set(g.kq_mask, mask.as_ptr().cast(), 0, nkv * 2);
        } else {
            let mut mask = vec![f32::NEG_INFINITY; nkv];
            for c in mask.iter_mut().take((mtp_past + 1).min(nkv)) {
                *c = 0.0;
            }
            ffi::ggml_backend_tensor_set(g.kq_mask, mask.as_ptr().cast(), 0, nkv * 4);
        }
        let zero = [0i32];
        ffi::ggml_backend_tensor_set(g.out_ids, zero.as_ptr().cast(), 0, 4);
    }

    /// Cached T=1 decode: reuse the per-bucket graph + frozen allocation.
    fn step_cached(
        &self,
        session: &mut Session,
        tokens: &[u32],
        n_threads: i32,
        greedy: bool,
    ) -> Result<(StepOut, Vec<f32>, Vec<f32>), ModelError> {
        let t_len = tokens.len() as i64;
        let n_kv_exact = (session.n_past + tokens.len()) as i64;
        let kvb = kv_bucket();
        let bucket = (((n_kv_exact + kvb - 1) / kvb) * kvb).min(session.n_ctx_max as i64);
        let stale = session
            .cached
            .as_ref()
            .map(|c| c.bucket != bucket || c.greedy != greedy || c.t_len != t_len)
            .unwrap_or(true);
        if stale {
            unsafe {
                let built = self.build_inner(
                    t_len,
                    bucket,
                    &StateSrc::Session(session.view()),
                    t_len,
                    true,
                    0,
                    greedy,
                    None,
                    SeqMode::Single,
                    false,
                    false,
                )?;
                let galloc = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(
                    self.weights.backend(),
                ));
                if galloc.is_null() {
                    ffi::ggml_free(built.ctx);
                    return Err(ModelError::Load("decode gallocr".into()));
                }
                if !ffi::ggml_gallocr_alloc_graph(galloc, built.gf) {
                    ffi::ggml_gallocr_free(galloc);
                    ffi::ggml_free(built.ctx);
                    return Err(ModelError::Load("decode graph alloc".into()));
                }
                trace_mem("decode", galloc, bucket, t_len);
                ffi::ggml_graph_set_new_uid(built.gf);
                let host = HostStaging {
                    tokens: vec![0i32; t_len as usize],
                    pos: vec![0i32; t_len as usize * 4],
                    mask_f16: vec![0xFC00u16; bucket as usize * t_len as usize],
                    row_ids: vec![0i64; t_len as usize],
                    mask_visible: 0,
                    out_ids: (0..t_len as i32).collect(),
                };
                session.cached =
                    Some(CachedStep { built, galloc, bucket, greedy, t_len, host });
            }
        }
        let n_past = session.n_past;
        let rope_off = session.rope_off;
        let cached = session.cached.as_mut().unwrap();
        unsafe {
            let backend = self.weights.backend();
            let b = &cached.built;
            let h = &mut cached.host;

            // Inputs are uploaded ASYNC into the compute stream and never read
            // back before the graph runs, so no per-upload device sync: the
            // stream ordering is the synchronization. Only the final output
            // read (4 bytes for greedy) synchronizes.
            let t = tokens.len();
            for (i, tk) in tokens.iter().enumerate() {
                h.tokens[i] = *tk as i32;
            }
            ffi::ggml_backend_tensor_set_async(
                backend, b.inp_tokens, h.tokens.as_ptr().cast(), 0, t * 4,
            );

            for i in 0..t {
                let p = (n_past as i64 + i as i64 + rope_off) as i32;
                h.pos[i] = p;
                h.pos[t + i] = p;
                h.pos[2 * t + i] = p;
                h.pos[3 * t + i] = 0;
            }
            ffi::ggml_backend_tensor_set_async(
                backend, b.inp_pos, h.pos.as_ptr().cast(), 0, t * 4 * 4,
            );

            for i in 0..t {
                h.row_ids[i] = (n_past + i) as i64;
            }
            ffi::ggml_backend_tensor_set_async(
                backend, b.row_ids, h.row_ids.as_ptr().cast(), 0, t * 8,
            );

            // EVERY input is re-uploaded before EVERY compute. ggml's graph
            // allocator may place intermediates over an input's storage once
            // that input's last consumer has run, so device-side values do
            // NOT survive between computes of the same graph. Skipping a
            // re-upload (out_ids "is constant"; the mask "only changed by one
            // cell") clobbered the out_ids index and aborted get_rows at
            // n_past 2333 — silent until the allocation happened to overlap.
            // causal window per query row; rows are contiguous in the mask
            let nkv = b.n_kv as usize;
            for q in 0..t {
                let vis = (n_past + q + 1).min(nkv);
                for c in 0..nkv {
                    h.mask_f16[q * nkv + c] = if c < vis { 0 } else { 0xFC00 };
                }
            }
            h.mask_visible = n_past + t;
            ffi::ggml_backend_tensor_set_async(
                backend,
                b.kq_mask,
                h.mask_f16.as_ptr().cast(),
                0,
                h.mask_f16.len() * 2,
            );

            ffi::ggml_backend_tensor_set_async(
                backend,
                b.out_ids,
                h.out_ids.as_ptr().cast(),
                0,
                h.out_ids.len() * 4,
            );

            self.compute(b.gf, n_threads)?;
            // the draft head consumes these, and they are already computed
            let mut hidden = vec![0f32; self.hp.n_embd as usize * t];
            ffi::ggml_backend_tensor_get(
                b.h_out, hidden.as_mut_ptr().cast(), 0, hidden.len() * 4,
            );
            let tv = self.read_taps(b, t);
            Ok((self.read_out(b, t), hidden, tv))
        }
    }

    unsafe fn build(
        &self,
        t_len: i64,
        n_past: usize,
        state: &StateSrc,
        n_out: i64,
        use_set_rows: bool,
        greedy: bool,
        mtp_tail: Option<MtpTail>,
    ) -> Result<Built, ModelError> {
        let n_kv_exact = n_past as i64 + t_len;
        let n_kv = match state {
            StateSrc::Stateless => n_kv_exact,
            StateSrc::Session(s) => {
                let kvb = kv_bucket();
                (((n_kv_exact + kvb - 1) / kvb) * kvb).min(s.n_ctx_max as i64)
            }
        };
        self.build_inner(
            t_len, n_kv, state, n_out, use_set_rows, n_past, greedy, mtp_tail, SeqMode::Single,
            false, false,
        )
    }

    /// Build the forward graph. `n_past_views` is used ONLY by the
    /// view-offset cache writes of the general path; set_rows graphs pass 0
    /// and stay topology-independent of n_past.
    unsafe fn build_inner(
        &self,
        t_len: i64,
        n_kv: i64,
        state: &StateSrc,
        n_out: i64,
        use_set_rows: bool,
        n_past_views: usize,
        greedy: bool,
        mtp_tail: Option<MtpTail>,
        seq: SeqMode,
        embd_input: bool,
        gumbel: bool,
    ) -> Result<Built, ModelError> {
        let hp = &self.hp;

        // --- CUDA-bisect toggles (kept for kernel-path debugging) ---
        let dbg = DebugToggles::from_env();
        let dbg_attn_batch = dbg.attn_batch;
        let dbg_gdn_zero = dbg.gdn_zero;
        let dbg_no_writes = dbg.no_writes;

        // ~60 nodes per layer at the widest (GDN with K snapshot writes),
        // plus head and inputs. Sized from the model, not a fixed guess: the
        // 27B with speculative verify overflowed a hardcoded 8192.
        let graph_nodes = hp.n_layer * 96 + 512;
        let params = ffi::ggml_init_params {
            mem_size: (graph_nodes * 2) * ffi::ggml_tensor_overhead()
                + ffi::ggml_graph_overhead_custom(graph_nodes, false)
                + (16 << 20),
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return Err(ModelError::Load("graph ctx init".into()));
        }

        let gf = ffi::ggml_new_graph_custom(ctx, graph_nodes, false);
        let f32t = ffi::ggml_type_GGML_TYPE_F32;

        // ---- inputs ----
        // Exactly one of inp_tokens / inp_embd exists: a tensor created but
        // absent from the graph would never be allocated, and writing to it
        // would fault, so the unused one stays null.
        let (inp_tokens, inp_embd) = if embd_input {
            let e = ffi::ggml_new_tensor_2d(ctx, f32t, hp.n_embd, t_len);
            ffi::ggml_set_input(e);
            (std::ptr::null_mut::<ffi::ggml_tensor>(), e)
        } else {
            let tk = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len);
            ffi::ggml_set_input(tk);
            (tk, std::ptr::null_mut::<ffi::ggml_tensor>())
        };
        let inp_pos = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len * 4);
        ffi::ggml_set_input(inp_pos);
        let use_fa = matches!(state, StateSrc::Session(s) if s.fa) && !dbg_attn_batch;
        let mask_t = if use_fa { ffi::ggml_type_GGML_TYPE_F16 } else { f32t };
        let kq_mask = ffi::ggml_new_tensor_2d(ctx, mask_t, n_kv, t_len);
        ffi::ggml_set_input(kq_mask);
        let row_ids = if use_set_rows {
            let r = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I64, t_len);
            ffi::ggml_set_input(r);
            r
        } else {
            std::ptr::null_mut()
        };
        let need_zeros = matches!(state, StateSrc::Stateless) || dbg_gdn_zero;
        let (conv_zero, state_zero) = if need_zeros {
            let c = ffi::ggml_new_tensor_3d(ctx, f32t, hp.d_conv - 1, hp.conv_dim(), 1);
            ffi::ggml_set_input(c);
            let s = ffi::ggml_new_tensor_4d(
                ctx, f32t, hp.gdn_head_v(), hp.gdn_head_v(), hp.n_v_heads, 1,
            );
            ffi::ggml_set_input(s);
            (c, s)
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut())
        };
        let out_ids = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, n_out);
        ffi::ggml_set_input(out_ids);

        let as_f32 = |ctx: *mut ffi::ggml_context, t: *mut ffi::ggml_tensor| {
            if (*t).type_ == f32t {
                t
            } else {
                ffi::ggml_cast(ctx, t, f32t)
            }
        };
        let rms = |ctx: *mut ffi::ggml_context, x: *mut ffi::ggml_tensor, w: *mut ffi::ggml_tensor| {
            let n = ffi::ggml_rms_norm(ctx, x, hp.rms_eps);
            ffi::ggml_mul(ctx, n, as_f32(ctx, w))
        };

        // ---- trunk ----
        let tok_embd = self.t("token_embd.weight")?;
        let mut cur;
        let mut inp_l = if embd_input {
            inp_embd
        } else {
            ffi::ggml_get_rows(ctx, tok_embd, inp_tokens)
        };

        let elt = ffi::ggml_type_size(f32t);
        let row = |n: i64| ffi::ggml_row_size(f32t, n);

        let mut taps: Vec<*mut ffi::ggml_tensor> = Vec::new();
        let collect_taps =
            !self.tap_layers.is_empty() && !matches!(seq, SeqMode::Batched);
        for il in 0..hp.n_layer {
            if collect_taps && self.tap_layers.contains(&il) {
                ffi::ggml_set_output(inp_l);
                taps.push(inp_l);
            }
            let l = self.layer(il)?;
            let inp_sa = inp_l;

            cur = rms(ctx, inp_l, l.attn_norm);

            if hp.is_recurrent(il) {
                // ---- gated delta net ----
                let key_dim = hp.key_dim();
                let value_dim = hp.value_dim();
                let head_v = hp.gdn_head_v();

                // In Batched mode the graph's t_len tokens are one token from each of
                // t_len sequences, and the GDN op wants the sequence axis in ne[3]:
                // q,k,v,g,beta [S, H, n_tokens, n_seqs] with n_tokens = 1. The buffers
                // are identical either way — these are contiguous reshapes.
                let (rt, rs) = if seq == SeqMode::Batched { (1, t_len) } else { (t_len, 1) };
                let qkv_mixed = ffi::ggml_mul_mat(ctx, l.wqkv, cur);
                let qkv_mixed = ffi::ggml_reshape_4d(ctx, qkv_mixed, hp.conv_dim(), rt, rs, 1);
                let z = ffi::ggml_mul_mat(ctx, l.wqkv_gate, cur);

                let beta = ffi::ggml_mul_mat(ctx, l.ssm_beta, cur);
                let beta = ffi::ggml_reshape_4d(ctx, beta, 1, hp.n_v_heads, rt, rs);
                let beta = ffi::ggml_sigmoid(ctx, beta);

                let alpha = ffi::ggml_mul_mat(ctx, l.ssm_alpha, cur);
                let alpha = ffi::ggml_reshape_3d(ctx, alpha, hp.n_v_heads, t_len, 1);
                let alpha = ffi::ggml_add(ctx, alpha, as_f32(ctx, l.dt_bias));
                let alpha = ffi::ggml_softplus(ctx, alpha);
                let g = ffi::ggml_mul(ctx, alpha, as_f32(ctx, l.ssm_a));
                let g = ffi::ggml_reshape_4d(ctx, g, 1, hp.n_v_heads, rt, rs);

                let (conv_in_state, gdn_in_state) = match state {
                    StateSrc::Stateless => (conv_zero, state_zero),
                    StateSrc::Session(_) if dbg_gdn_zero => (conv_zero, state_zero),
                    StateSrc::Session(s) => {
                        // The slot dimension serves two masters: in Single mode slot 0
                        // is the live state and the rest are rollback snapshots; in
                        // Slot/Batched modes each slot is a different SEQUENCE's live
                        // state.
                        let (slot0, nslots) = match seq {
                            SeqMode::Single => (0i64, 1i64),
                            SeqMode::Slot(i) => (i, 1),
                            SeqMode::Batched => (0, t_len),
                        };
                        let c3 = ffi::ggml_view_3d(
                            ctx, s.conv_state[il], hp.d_conv - 1, hp.conv_dim(), nslots,
                            (*s.conv_state[il]).nb[1], (*s.conv_state[il]).nb[2],
                            slot0 as usize * (*s.conv_state[il]).nb[2],
                        );
                        let s4 = ffi::ggml_view_4d(
                            ctx, s.gdn_state[il], head_v, head_v, hp.n_v_heads, nslots,
                            (*s.gdn_state[il]).nb[1], (*s.gdn_state[il]).nb[2],
                            (*s.gdn_state[il]).nb[3],
                            slot0 as usize * (*s.gdn_state[il]).nb[3],
                        );
                        (c3, s4)
                    }
                };

                let qkv_t = ffi::ggml_transpose(ctx, qkv_mixed);
                let conv_input = ffi::ggml_concat(ctx, conv_in_state, qkv_t, 0);

                if let (StateSrc::Session(s), false) = (state, dbg_no_writes || dbg_gdn_zero) {
                    match seq {
                        SeqMode::Single => {
                            // Snapshot slot s = the conv window ending s tokens back.
                            let n_written = s.k_slots.min(t_len);
                            for slot in 0..n_written {
                                let idx = (*conv_input).ne[0] - (hp.d_conv - 1) - slot;
                                if idx < 0 {
                                    break;
                                }
                                let tail = ffi::ggml_view_3d(
                                    ctx, conv_input,
                                    hp.d_conv - 1, hp.conv_dim(), 1,
                                    (*conv_input).nb[1], (*conv_input).nb[2],
                                    row(idx),
                                );
                                let dst = ffi::ggml_view_3d(
                                    ctx, s.conv_state[il], hp.d_conv - 1, hp.conv_dim(), 1,
                                    (*s.conv_state[il]).nb[1], (*s.conv_state[il]).nb[2],
                                    slot as usize * (*s.conv_state[il]).nb[2],
                                );
                                ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, tail, dst));
                            }
                        }
                        SeqMode::Slot(slot) => {
                            // one sequence, one live window, written to its own slot
                            let idx = (*conv_input).ne[0] - (hp.d_conv - 1);
                            let tail = ffi::ggml_view_3d(
                                ctx, conv_input,
                                hp.d_conv - 1, hp.conv_dim(), 1,
                                (*conv_input).nb[1], (*conv_input).nb[2],
                                row(idx),
                            );
                            let dst = ffi::ggml_view_3d(
                                ctx, s.conv_state[il], hp.d_conv - 1, hp.conv_dim(), 1,
                                (*s.conv_state[il]).nb[1], (*s.conv_state[il]).nb[2],
                                slot as usize * (*s.conv_state[il]).nb[2],
                            );
                            ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, tail, dst));
                        }
                        SeqMode::Batched => {
                            // every sequence advanced one token: its new window is the
                            // input shifted by one, all slots in one copy
                            let tail = ffi::ggml_view_3d(
                                ctx, conv_input,
                                hp.d_conv - 1, hp.conv_dim(), t_len,
                                (*conv_input).nb[1], (*conv_input).nb[2],
                                row(1),
                            );
                            let dst = ffi::ggml_view_3d(
                                ctx, s.conv_state[il], hp.d_conv - 1, hp.conv_dim(), t_len,
                                (*s.conv_state[il]).nb[1], (*s.conv_state[il]).nb[2], 0,
                            );
                            ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, tail, dst));
                        }
                    }
                }

                let conv_out = ffi::ggml_ssm_conv(ctx, conv_input, as_f32(ctx, l.conv1d));
                let conv_out = ffi::ggml_silu(ctx, conv_out);

                let nb1_qkv = row(hp.conv_dim());
                // ne[2]/ne[3] carry (tokens, sequences); per-token stride is nb1_qkv
                // either way, so only which axis it lands on changes
                let (nb2_q, nb3_q) = if seq == SeqMode::Batched {
                    (nb1_qkv, nb1_qkv)
                } else {
                    (nb1_qkv, nb1_qkv * t_len as usize)
                };
                let q = ffi::ggml_view_4d(
                    ctx, conv_out,
                    hp.d_state, hp.n_k_heads, rt, rs,
                    row(hp.d_state), nb2_q, nb3_q, 0,
                );
                let k = ffi::ggml_view_4d(
                    ctx, conv_out,
                    hp.d_state, hp.n_k_heads, rt, rs,
                    row(hp.d_state), nb2_q, nb3_q,
                    key_dim as usize * elt,
                );
                let v = ffi::ggml_view_4d(
                    ctx, conv_out,
                    head_v, hp.n_v_heads, rt, rs,
                    row(head_v), nb2_q, nb3_q,
                    row(2 * key_dim),
                );

                let q = ffi::ggml_l2_norm(ctx, q, hp.rms_eps);
                let k = ffi::ggml_l2_norm(ctx, k, hp.rms_eps);
                let v = ffi::ggml_cont(ctx, v);

                let k_snap = match (state, seq) {
                    // rollback snapshots exist only in Single mode; a batch slot's
                    // "snapshot 0" is simply its live state
                    (StateSrc::Session(s), SeqMode::Single) => s.k_slots,
                    _ => 1,
                };
                let gdn =
                    ffi::ggml_gated_delta_net(ctx, q, k, v, g, beta, gdn_in_state, k_snap);

                let out = ffi::ggml_view_4d(
                    ctx, gdn,
                    head_v, hp.n_v_heads, rt, rs,
                    row(head_v), row(head_v * hp.n_v_heads),
                    row(head_v * hp.n_v_heads * rt), 0,
                );

                if let (StateSrc::Session(s), false) = (state, dbg_no_writes || dbg_gdn_zero) {
                    // The op packs its state output after the attention rows. In Single
                    // mode that is K snapshots for slots 0..K; in Slot/Batched modes it
                    // is one live state per sequence, written to the matching slot(s).
                    let d = head_v * head_v * hp.n_v_heads;
                    let (n_states, dst_slot0) = match seq {
                        SeqMode::Single => (s.k_slots.min(t_len), 0i64),
                        SeqMode::Slot(i) => (1, i),
                        SeqMode::Batched => (t_len, 0),
                    };
                    // the states start after ALL attention rows; rt * rs == t_len in
                    // every mode, so the offset is the same expression in both
                    let src = ffi::ggml_view_3d(
                        ctx, gdn, d, 1, n_states,
                        row(d), row(d),
                        row(head_v * hp.n_v_heads * t_len),
                    );
                    let dst = ffi::ggml_view_3d(
                        ctx, s.gdn_state[il], d, 1, n_states,
                        row(d), row(d),
                        dst_slot0 as usize * (*s.gdn_state[il]).nb[3],
                    );
                    ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, src, dst));
                }

                let z4 = ffi::ggml_reshape_4d(ctx, z, head_v, hp.n_v_heads, rt, rs);
                let normed = rms(ctx, out, l.ssm_norm);
                let gated = ffi::ggml_mul(ctx, normed, ffi::ggml_silu(ctx, z4));
                let flat =
                    ffi::ggml_reshape_3d(ctx, ffi::ggml_cont(ctx, gated), value_dim, t_len, 1);
                cur = ffi::ggml_mul_mat(ctx, l.ssm_out, flat);
                cur = ffi::ggml_reshape_2d(ctx, cur, hp.n_embd, t_len);
            } else {
                // full attention: shared with the MTP draft head
                cur = self.build_attn_block(
                    ctx, gf, cur, &l, inp_pos, kq_mask, row_ids,
                    match state {
                        StateSrc::Stateless => None,
                        StateSrc::Session(s) => {
                            Some((s.k_cache[il], s.v_cache[il], s.n_ctx_max))
                        }
                    },
                    t_len, n_kv, n_past_views,
                    matches!(state, StateSrc::Session(s) if s.fa),
                    use_fa, use_set_rows, dbg,
                );
            }

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

        let output_norm = self.t("output_norm.weight")?;
        let output_w = self.weights.tensor("output.weight").unwrap_or(tok_embd);
        cur = rms(ctx, inp_l, output_norm);
        // h_nextn: the normed trunk hidden BEFORE the LM head. The MTP draft
        // head consumes exactly this (llama.cpp's res->t_h_nextn), so it must
        // be readable, not just an intermediate.
        let h_sel = ffi::ggml_get_rows(ctx, cur, out_ids);
        ffi::ggml_set_output(h_sel);
        ffi::ggml_build_forward_expand(gf, h_sel);
        cur = ffi::ggml_mul_mat(ctx, output_w, h_sel);
        let logits_t = cur;
        ffi::ggml_set_output(logits_t);
        ffi::ggml_build_forward_expand(gf, logits_t);
        let mut inp_gumbel: *mut ffi::ggml_tensor = std::ptr::null_mut();
        if greedy {
            // Sample in the graph: the readback becomes 4 bytes instead of
            // n_vocab floats (993 KB for this model), removing a full
            // device sync + PCIe transfer from every decode step. Exact for
            // temp-0: ggml_argmax selects the same element CPU argmax would.
            // With gumbel noise added first, the same argmax becomes an exact
            // temperature sample — argmax(l + g*T) = argmax(l/T + g), the
            // gumbel-max draw from softmax(l/T) — so the draft chain below
            // conditions on the token the host will actually commit.
            if gumbel {
                let gz = ffi::ggml_new_tensor_2d(ctx, f32t, hp.n_vocab, n_out);
                ffi::ggml_set_input(gz);
                inp_gumbel = gz;
                cur = ffi::ggml_argmax(ctx, ffi::ggml_add(ctx, logits_t, gz));
            } else {
                cur = ffi::ggml_argmax(ctx, cur);
            }
        }
        ffi::ggml_set_output(cur);
        ffi::ggml_build_forward_expand(gf, cur);

        // ---- fused MTP draft tail ----
        //
        // The draft head normally runs as its own graph execution, and on the
        // 27B that costs ~8 ms against ~1.4 ms of actual bandwidth: almost all
        // of it is building and allocating a graph, not computing one. So the
        // draft is built into the verify graph instead, where it is just a few
        // dozen more nodes on a launch we are already paying for.
        //
        // It works because the token the draft head needs — the model's own
        // prediction at each verified position — is available *inside* the
        // graph: ggml_argmax emits I32 and ggml_get_rows consumes I32, so
        // `embed(argmax(logits))` is expressible with no host round-trip.
        //
        // Entry i assumes positions before it were accepted, which is exactly
        // the condition under which its draft is the one we use, so the
        // batched drafts stay consistent with whatever prefix verification
        // ends up keeping.
        let mut draft_chain: Vec<*mut ffi::ggml_tensor> = Vec::new();
        let mut draft_conf: Vec<*mut ffi::ggml_tensor> = Vec::new();
        let mut cand_ids: *mut ffi::ggml_tensor = std::ptr::null_mut();
        let (mut draft_out, mut mtp_pos, mut mtp_mask, mut mtp_rows, mut mtp_n_kv) = (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0i64,
        );
        if let (Some(tail), true) = (mtp_tail, greedy) {
            let (ml, mx) = self
                .mtp_layer()
                .ok_or_else(|| ModelError::Load("model has no MTP head".into()))?;
            mtp_n_kv = tail.n_kv;

            let mp = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, n_out * 4);
            ffi::ggml_set_input(mp);
            mtp_pos = mp;
            let mm = ffi::ggml_new_tensor_2d(
                ctx,
                if use_fa { ffi::ggml_type_GGML_TYPE_F16 } else { f32t },
                tail.n_kv,
                n_out,
            );
            ffi::ggml_set_input(mm);
            mtp_mask = mm;
            let mr = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I64, n_out);
            ffi::ggml_set_input(mr);
            mtp_rows = mr;
            if tail.n_cand > 0 {
                let ci = ffi::ggml_new_tensor_1d(
                    ctx, ffi::ggml_type_GGML_TYPE_I32, tail.n_cand,
                );
                ffi::ggml_set_input(ci);
                cand_ids = ci;
            }

            // Chain the drafts inside this one graph. Each step feeds on the
            // previous step's own prediction and hidden state, so a depth-3
            // chain is ~120 extra nodes on a launch we already pay for rather
            // than three separate executions at ~7 ms each (see the cost
            // curve in docs/OPTIMIZATION-IDEAS.md).
            let mut tok_src = cur; // the trunk's own predictions
            let mut h_src = h_sel;
            for step in 0..tail.depth.max(1) {
                let next_emb = ffi::ggml_get_rows(
                    ctx,
                    mx.embed_tokens.unwrap_or(tok_embd),
                    tok_src,
                );
                let e_norm = rms(ctx, next_emb, mx.enorm);
                let h_norm = rms(ctx, h_src, mx.hnorm);
                let cat = ffi::ggml_concat(ctx, e_norm, h_norm, 0);
                let mut d = ffi::ggml_mul_mat(ctx, mx.eh_proj, cat);

                let d_sa = d;
                d = rms(ctx, d, ml.attn_norm);
                // Later chain steps must not write the draft cache: only the
                // first is on the path the round can actually commit.
                d = self.build_attn_block(
                    ctx, gf, d, &ml, mp, mm, mr,
                    Some((tail.k_cache, tail.v_cache, tail.n_ctx_max)),
                    n_out, tail.n_kv, tail.n_past, use_fa, use_fa,
                    use_set_rows,
                    DebugToggles { no_writes: step > 0, ..dbg },
                );
                d = ffi::ggml_add(ctx, d, d_sa);

                let d_res = d;
                let dn = rms(ctx, d, ml.post_attn_norm);
                let du = ffi::ggml_mul_mat(ctx, ml.ffn_up, dn);
                let dgt = ffi::ggml_mul_mat(ctx, ml.ffn_gate, dn);
                let da = ffi::ggml_mul(ctx, ffi::ggml_silu(ctx, dgt), du);
                d = ffi::ggml_mul_mat(ctx, ml.ffn_down, da);
                d = ffi::ggml_add(ctx, d, d_res);

                let hn = mx.shared_head_norm.unwrap_or(output_norm);
                let h_next = rms(ctx, d, hn);
                let head_w = mx.shared_head_head.unwrap_or(output_w);
                // Project the draft onto the shortlist when one is supplied:
                // gathering n_cand rows costs a few tens of MB against the
                // 1.27 GiB the full head reads, and the draft only needs its
                // argmax. Verification below still uses the full head, so a
                // token outside the shortlist just loses that draft.
                let (logits_d, ids) = if cand_ids.is_null() {
                    let l = ffi::ggml_mul_mat(ctx, head_w, h_next);
                    (l, ffi::ggml_argmax(ctx, l))
                } else {
                    let sub = ffi::ggml_get_rows(ctx, head_w, cand_ids);
                    let l = ffi::ggml_mul_mat(ctx, sub, h_next);
                    (l, ffi::ggml_argmax(ctx, l))
                };
                ffi::ggml_set_output(ids);
                ffi::ggml_build_forward_expand(gf, ids);
                // The link's confidence: the softmax peak at its own argmax. This is
                // what lets the host truncate a carried chain at the first link the
                // draft head does not believe in — production's p-min — and it is why
                // acceptance can sit above 0.9 without giving up depth. The gather is
                // batched get_rows: the softmax viewed as [1, vocab, n_out] with the
                // argmax ids as [1, n_out] selects element (ids[j], j) per column.
                // Everything here inherits the head's MIRRORED split state; source
                // ops like arange have no split axis and abort the TP meta backend.
                let conf = {
                    let vocab_rows = (*logits_d).ne[0];
                    let pm = ffi::ggml_soft_max(ctx, logits_d);
                    let table = ffi::ggml_reshape_3d(ctx, pm, 1, vocab_rows, n_out);
                    let cols = ffi::ggml_reshape_2d(ctx, ids, 1, n_out);
                    let peak = ffi::ggml_get_rows(ctx, table, cols);
                    ffi::ggml_set_output(peak);
                    ffi::ggml_build_forward_expand(gf, peak);
                    peak
                };
                draft_conf.push(conf);
                if step == 0 {
                    draft_out = ids;
                } else {
                    draft_chain.push(ids);
                }
                tok_src = ids;
                h_src = h_next;
            }
        }

        Ok(Built {
            ctx,
            gf,
            inp_tokens,
            inp_embd,
            inp_gumbel,
            inp_pos,
            taps,
            kq_mask,
            row_ids,
            conv_zero,
            state_zero,
            out_ids,
            out: cur,
            logits: logits_t,
            h_out: h_sel,
            draft_out,
            draft_chain,
            draft_conf,
            cand_ids,
            mtp_pos,
            mtp_mask,
            mtp_rows,
            mtp_n_kv,
            n_kv,
            fa_mask: use_fa,
            greedy,
        })
    }


/// One full-attention block: packed Q+gate projection, per-head q/k RMS
/// norms, IMROPE, GQA attention against `cache` (or in-batch K/V when
/// stateless), output gating, and the wo projection. Shared by the trunk's
/// attention layers and the MTP draft head so the two cannot drift apart.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn build_attn_block(
    &self,
    ctx: *mut ffi::ggml_context,
    gf: *mut ffi::ggml_cgraph,
    cur: *mut ffi::ggml_tensor,
    l: &Layer,
    inp_pos: *mut ffi::ggml_tensor,
    kq_mask: *mut ffi::ggml_tensor,
    row_ids: *mut ffi::ggml_tensor,
    // cache: (k, v, n_ctx_max); None runs stateless over the batch
    cache: Option<(*mut ffi::ggml_tensor, *mut ffi::ggml_tensor, usize)>,
    t_len: i64,
    n_kv: i64,
    n_past_views: usize,
    fa: bool,
    use_fa: bool,
    use_set_rows: bool,
    dbg: DebugToggles,
) -> *mut ffi::ggml_tensor {
        // Q and a per-head output gate are packed in wq (2 x head_dim per
        // head); q/k are RMS-normed per head; IMROPE covers n_rot of head_dim.
        let hp = &self.hp;
        let hd = hp.head_k;
        let elt = ffi::ggml_type_size(ffi::ggml_type_GGML_TYPE_F32);
        let mut sections = hp.rope_sections;
        let rms = |ctx: *mut ffi::ggml_context, x: *mut ffi::ggml_tensor, w: *mut ffi::ggml_tensor| {
            let n = ffi::ggml_rms_norm(ctx, x, hp.rms_eps);
            ffi::ggml_mul(ctx, n, if (*w).type_ == ffi::ggml_type_GGML_TYPE_F32 { w } else { ffi::ggml_cast(ctx, w, ffi::ggml_type_GGML_TYPE_F32) })
        };
        let q_full = ffi::ggml_mul_mat(ctx, l.wq, cur);

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

        let kcur = ffi::ggml_mul_mat(ctx, l.wk, cur);
        let kcur = ffi::ggml_reshape_3d(ctx, kcur, hd, hp.n_head_kv, t_len);
        let kcur = rms(ctx, kcur, l.k_norm);

        let vcur = ffi::ggml_mul_mat(ctx, l.wv, cur);

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

        let batch_kv = |ctx: *mut ffi::ggml_context| {
            let k = ffi::ggml_permute(ctx, kcur, 0, 2, 1, 3);
            let v3 = ffi::ggml_reshape_3d(ctx, vcur, hp.head_v, hp.n_head_kv, t_len);
            let v = ffi::ggml_permute(ctx, v3, 0, 2, 1, 3);
            let v_t = ffi::ggml_cont(ctx, ffi::ggml_transpose(ctx, v));
            (k, v_t)
        };
        let (k_all, v_all) = match cache {
            None => batch_kv(ctx),
            Some((kc, vc, n_ctx_max)) => {
                let elt_kv = ffi::ggml_type_size((*kc).type_);

                let k2 = ffi::ggml_reshape_2d(
                    ctx,
                    ffi::ggml_cont(ctx, kcur),
                    hd * hp.n_head_kv,
                    t_len,
                );
                if use_set_rows {
                    debug_assert!(fa, "set_rows path requires FA V layout");
                    if !dbg.no_writes {
                        ffi::ggml_build_forward_expand(
                            gf,
                            ffi::ggml_set_rows(ctx, kc, k2, row_ids),
                        );
                        ffi::ggml_build_forward_expand(
                            gf,
                            ffi::ggml_set_rows(ctx, vc, vcur, row_ids),
                        );
                    }
                } else {
                    let k_dst = ffi::ggml_view_2d(
                        ctx, kc, hd * hp.n_head_kv, t_len,
                        (*kc).nb[1], n_past_views * (*kc).nb[1],
                    );
                    if !dbg.no_writes {
                        ffi::ggml_build_forward_expand(
                            gf, ffi::ggml_cpy(ctx, k2, k_dst),
                        );
                    }
                    if fa {
                        let v_dst = ffi::ggml_view_2d(
                            ctx, vc, hp.head_v * hp.n_head_kv, t_len,
                            (*vc).nb[1], n_past_views * (*vc).nb[1],
                        );
                        if !dbg.no_writes {
                            ffi::ggml_build_forward_expand(
                                gf, ffi::ggml_cpy(ctx, vcur, v_dst),
                            );
                        }
                    } else {
                        let v_t_new = ffi::ggml_transpose(ctx, vcur);
                        let v_dst = ffi::ggml_view_2d(
                            ctx, vc, t_len, hp.head_v * hp.n_head_kv,
                            (*vc).nb[1], n_past_views * elt_kv,
                        );
                        if !dbg.no_writes {
                            ffi::ggml_build_forward_expand(
                                gf, ffi::ggml_cpy(ctx, v_t_new, v_dst),
                            );
                        }
                    }
                }

                // Cache reads are built as [head_dim, n_head_kv, n_kv]
                // — strides increase monotonically, so ggml does not
                // consider the view "permuted" — and then permuted
                // into the [head_dim, n_kv, n_head_kv] that attention
                // wants. Viewing tokens-before-heads directly would
                // invert nb[1]/nb[2], which the tensor-parallel meta
                // backend refuses ("view of permuted tensor not
                // implemented"): it cannot map a split axis through
                // a stride-reordered view.
                let k_all = {
                    let v4 = ffi::ggml_view_4d(
                        ctx, kc, hd, hp.n_head_kv, n_kv, 1,
                        hd as usize * elt_kv,
                        (*kc).nb[1],
                        (*kc).nb[1] * n_ctx_max,
                        0,
                    );
                    ffi::ggml_permute(ctx, v4, 0, 2, 1, 3)
                };
                let v_all = if fa {
                    let v4 = ffi::ggml_view_4d(
                        ctx, vc, hp.head_v, hp.n_head_kv, n_kv, 1,
                        hp.head_v as usize * elt_kv,
                        (*vc).nb[1],
                        (*vc).nb[1] * n_ctx_max,
                        0,
                    );
                    ffi::ggml_permute(ctx, v4, 0, 2, 1, 3)
                } else {
                    ffi::ggml_view_3d(
                        ctx, vc, n_kv, hp.head_v, hp.n_head_kv,
                        (*vc).nb[1],
                        n_ctx_max * hp.head_v as usize * elt_kv,
                        0,
                    )
                };
                if dbg.attn_batch {
                    batch_kv(ctx)
                } else {
                    let (kb, vb) = if dbg.k_batch || dbg.v_batch {
                        batch_kv(ctx)
                    } else {
                        (k_all, v_all)
                    };
                    (
                        if dbg.k_batch { kb } else { k_all },
                        if dbg.v_batch { vb } else { v_all },
                    )
                }
            }
        };

        let q = ffi::ggml_permute(ctx, qcur, 0, 2, 1, 3);
        let kq_scale = 1.0f32 / (hd as f32).sqrt();
        let merged = if use_fa {
            let fa_out = ffi::ggml_flash_attn_ext(
                ctx, q, k_all, v_all, kq_mask, kq_scale, 0.0, 0.0,
            );
            ffi::ggml_flash_attn_ext_set_prec(fa_out, ffi::ggml_prec_GGML_PREC_F32);
            ffi::ggml_reshape_2d(ctx, fa_out, hp.head_v * hp.n_head, t_len)
        } else {
            let kq = ffi::ggml_mul_mat(ctx, k_all, q);
            let p = ffi::ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0);
            let kqv = ffi::ggml_mul_mat(ctx, v_all, p);
            let m = ffi::ggml_permute(ctx, kqv, 0, 2, 1, 3);
            ffi::ggml_cont_2d(ctx, m, hp.head_v * hp.n_head, t_len)
        };

        let gated = ffi::ggml_mul(ctx, merged, ffi::ggml_sigmoid(ctx, gate));
        ffi::ggml_mul_mat(ctx, l.wo, gated)
}



    /// Zero one slot's recurrent state, making it a fresh sequence.
    ///
    /// Done through a graph rather than a host write because under tensor parallelism
    /// the states are split across devices and the meta backend cannot write a slice of
    /// a split 4-D tensor from the host (limit #4). `scale(view, 0)` copied back into
    /// the view zeroes exactly the slot, on whatever device each shard lives.
    pub fn zero_seq_slot(
        &self,
        session: &mut Session,
        slot: usize,
        n_threads: i32,
    ) -> Result<(), ModelError> {
        let hp = &self.hp;
        unsafe {
            let n_recr = (0..hp.n_layer).filter(|&il| hp.is_recurrent(il)).count();
            let n_nodes = n_recr * 8 + 32;
            let params = ffi::ggml_init_params {
                mem_size: n_nodes * ffi::ggml_tensor_overhead()
                    + ffi::ggml_graph_overhead_custom(n_nodes, false),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(ModelError::Load("zero-slot ctx".into()));
            }
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _g = G(ctx);
            let gf = ffi::ggml_new_graph_custom(ctx, n_nodes, false);
            for il in 0..hp.n_layer {
                if !hp.is_recurrent(il) {
                    continue;
                }
                let cs = session.conv_state[il];
                let gs = session.gdn_state[il];
                let cv = ffi::ggml_view_3d(
                    ctx, cs, (*cs).ne[0], (*cs).ne[1], 1,
                    (*cs).nb[1], (*cs).nb[2], slot * (*cs).nb[2],
                );
                ffi::ggml_build_forward_expand(
                    gf,
                    ffi::ggml_cpy(ctx, ffi::ggml_scale(ctx, cv, 0.0), cv),
                );
                let gv = ffi::ggml_view_4d(
                    ctx, gs, (*gs).ne[0], (*gs).ne[1], (*gs).ne[2], 1,
                    (*gs).nb[1], (*gs).nb[2], (*gs).nb[3], slot * (*gs).nb[3],
                );
                ffi::ggml_build_forward_expand(
                    gf,
                    ffi::ggml_cpy(ctx, ffi::ggml_scale(ctx, gv, 0.0), gv),
                );
            }
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, gf) {
                    return Err(ModelError::Load("zero-slot sched alloc".into()));
                }
            } else {
                session.cached = None;
                if !ffi::ggml_gallocr_alloc_graph(session.galloc, gf) {
                    return Err(ModelError::Load("zero-slot alloc".into()));
                }
            }
            self.compute(gf, n_threads)
        }
    }

    /// Prefill one sequence of a batch session.
    ///
    /// The session's caches are shared: sequence `slot` owns KV rows
    /// `[slot * seq_ctx, (slot+1) * seq_ctx)` and recurrent-state slot `slot`. Returns
    /// the argmax at the last position and, when `want_logits`, the full distribution.
    pub fn step_seq_prefill(
        &self,
        session: &mut Session,
        tokens: &[u32],
        slot: usize,
        seq_ctx: usize,
        seq_past: usize,
        want_logits: bool,
        n_threads: i32,
    ) -> Result<(u32, Vec<f32>), ModelError> {
        assert!(seq_past + tokens.len() <= seq_ctx, "sequence overflows its region");
        let base = slot * seq_ctx;
        let t_len = tokens.len() as i64;
        // every graph in batch mode attends over the whole combined cache, so the one
        // decode graph and all prefill graphs agree on n_kv and can coexist
        let n_kv = session.n_ctx_max as i64;
        unsafe {
            let built = self.build_inner(
                t_len,
                n_kv,
                &StateSrc::Session(session.view()),
                1,
                /* use_set_rows */ true,
                0,
                /* greedy */ !want_logits,
                None,
                SeqMode::Slot(slot as i64),
                false,
                false,
            )?;
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _g = G(built.ctx);
            let galloc = session.galloc;
            session.cached = None;
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, built.gf) {
                    return Err(ModelError::Load("seq prefill sched alloc".into()));
                }
            } else if !ffi::ggml_gallocr_alloc_graph(galloc, built.gf) {
                return Err(ModelError::Load("seq prefill alloc".into()));
            }
            self.fill_inputs_region(
                &built,
                tokens,
                seq_past,
                base,
                &[(t_len - 1) as i32],
            );
            self.compute(built.gf, n_threads)?;
            if want_logits {
                let mut logits = vec![0f32; self.hp.n_vocab as usize];
                ffi::ggml_backend_tensor_get(
                    built.logits,
                    logits.as_mut_ptr().cast(),
                    0,
                    logits.len() * 4,
                );
                Ok((argmax(&logits), logits))
            } else {
                let mut id = 0i32;
                ffi::ggml_backend_tensor_get(built.out, (&mut id as *mut i32).cast(), 0, 4);
                Ok((id as u32, Vec::new()))
            }
        }
    }

    /// One decode round for every active sequence of a batch session: token i advances
    /// sequence i. Returns per-sequence argmaxes and, when `want_logits`, the
    /// distributions. The graph is cached and replayed — its shape depends only on the
    /// sequence count.
    pub fn step_batch_decode(
        &self,
        session: &mut Session,
        tokens: &[u32],
        seq_pasts: &[usize],
        seq_ctx: usize,
        want_logits: bool,
        n_threads: i32,
    ) -> Result<(Vec<u32>, Vec<f32>), ModelError> {
        let n = tokens.len();
        assert_eq!(n, seq_pasts.len());
        let t_len = n as i64;
        let n_kv = session.n_ctx_max as i64;
        unsafe {
            let stale = session
                .batch_step
                .as_ref()
                .map(|c| c.t_len != t_len || c.want_logits != want_logits)
                .unwrap_or(true);
            if stale {
                if let Some(c) = session.batch_step.take() {
                    ffi::ggml_gallocr_free(c.galloc);
                    session.graveyard.push(c.built.ctx);
                }
                let built = self.build_inner(
                    t_len,
                    n_kv,
                    &StateSrc::Session(session.view()),
                    t_len,
                    /* use_set_rows */ true,
                    0,
                    /* greedy */ !want_logits,
                    None,
                    SeqMode::Batched,
                    false,
                    false,
                )?;
                let ga = ffi::ggml_gallocr_new(ffi::ggml_backend_get_default_buffer_type(
                    self.weights.backend(),
                ));
                if ga.is_null() || !ffi::ggml_gallocr_alloc_graph(ga, built.gf) {
                    return Err(ModelError::Load("batch decode alloc".into()));
                }
                ffi::ggml_graph_set_new_uid(built.gf);
                session.batch_step = Some(BatchStep { built, galloc: ga, t_len, want_logits });
            }
            let c = session.batch_step.as_ref().unwrap();
            let b = &c.built;

            let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            ffi::ggml_backend_tensor_set(b.inp_tokens, toks_i32.as_ptr().cast(), 0, n * 4);
            let mut pos = vec![0i32; n * 4];
            for i in 0..n {
                let p = seq_pasts[i] as i32;
                pos[i] = p;
                pos[n + i] = p;
                pos[2 * n + i] = p;
            }
            ffi::ggml_backend_tensor_set(b.inp_pos, pos.as_ptr().cast(), 0, pos.len() * 4);
            let rows: Vec<i64> = (0..n).map(|i| (i * seq_ctx + seq_pasts[i]) as i64).collect();
            ffi::ggml_backend_tensor_set(b.row_ids, rows.as_ptr().cast(), 0, n * 8);
            let nkv = n_kv as usize;
            let mut mask = vec![0xFC00u16; nkv * n];
            for (q, past) in seq_pasts.iter().enumerate() {
                let base = q * seq_ctx;
                for kv in base..=(base + past).min(nkv - 1) {
                    mask[q * nkv + kv] = 0;
                }
            }
            ffi::ggml_backend_tensor_set(b.kq_mask, mask.as_ptr().cast(), 0, mask.len() * 2);
            let outs: Vec<i32> = (0..n as i32).collect();
            ffi::ggml_backend_tensor_set(b.out_ids, outs.as_ptr().cast(), 0, n * 4);

            self.compute(b.gf, n_threads)?;

            let mut logits = Vec::new();
            let preds = if want_logits {
                logits = vec![0f32; self.hp.n_vocab as usize * n];
                ffi::ggml_backend_tensor_get(
                    b.logits,
                    logits.as_mut_ptr().cast(),
                    0,
                    logits.len() * 4,
                );
                (0..n)
                    .map(|i| {
                        argmax(&logits[i * self.hp.n_vocab as usize
                            ..(i + 1) * self.hp.n_vocab as usize])
                    })
                    .collect()
            } else {
                let mut ids = vec![0i32; n];
                ffi::ggml_backend_tensor_get(b.out, ids.as_mut_ptr().cast(), 0, n * 4);
                ids.into_iter().map(|v| v as u32).collect()
            };
            Ok((preds, logits))
        }
    }

    /// `fill_inputs` for a sequence living in a region of a shared cache: positions are
    /// sequence-relative, KV rows and the causal mask are region-offset.
    unsafe fn fill_inputs_region(
        &self,
        b: &Built,
        tokens: &[u32],
        seq_past: usize,
        base: usize,
        out_positions: &[i32],
    ) {
        let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        ffi::ggml_backend_tensor_set(b.inp_tokens, toks_i32.as_ptr().cast(), 0, tokens.len() * 4);
        let mut pos = vec![0i32; tokens.len() * 4];
        for i in 0..tokens.len() {
            let p = (seq_past + i) as i32;
            pos[i] = p;
            pos[tokens.len() + i] = p;
            pos[2 * tokens.len() + i] = p;
        }
        ffi::ggml_backend_tensor_set(b.inp_pos, pos.as_ptr().cast(), 0, pos.len() * 4);
        let rows: Vec<i64> = (0..tokens.len()).map(|i| (base + seq_past + i) as i64).collect();
        ffi::ggml_backend_tensor_set(b.row_ids, rows.as_ptr().cast(), 0, rows.len() * 8);
        let nkv = b.n_kv as usize;
        let mut mask = vec![0xFC00u16; nkv * tokens.len()];
        for q in 0..tokens.len() {
            for kv in base..=(base + seq_past + q).min(nkv - 1) {
                mask[q * nkv + kv] = 0;
            }
        }
        ffi::ggml_backend_tensor_set(b.kq_mask, mask.as_ptr().cast(), 0, mask.len() * 2);
        ffi::ggml_backend_tensor_set(
            b.out_ids,
            out_positions.as_ptr().cast(),
            0,
            out_positions.len() * 4,
        );
    }

    /// `n_past` indexes physical KV rows (mask visibility); `rope_base` is
    /// the RoPE position of the first token — they differ once an image has
    /// been injected (see Session::rope_off).
    unsafe fn fill_inputs(
        &self,
        b: &Built,
        tokens: &[u32],
        n_past: usize,
        rope_base: usize,
        out_positions: &[i32],
    ) {
        let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        ffi::ggml_backend_tensor_set(b.inp_tokens, toks_i32.as_ptr().cast(), 0, tokens.len() * 4);

        let mut pos = vec![0i32; tokens.len() * 4];
        for i in 0..tokens.len() {
            let p = (rope_base + i) as i32;
            pos[i] = p;
            pos[tokens.len() + i] = p;
            pos[2 * tokens.len() + i] = p;
        }
        ffi::ggml_backend_tensor_set(b.inp_pos, pos.as_ptr().cast(), 0, pos.len() * 4);

        let nkv = b.n_kv as usize;
        if b.fa_mask {
            let mut mask = vec![0xFC00u16; nkv * tokens.len()];
            for q in 0..tokens.len() {
                for kv in 0..=(n_past + q).min(nkv - 1) {
                    mask[q * nkv + kv] = 0;
                }
            }
            ffi::ggml_backend_tensor_set(b.kq_mask, mask.as_ptr().cast(), 0, mask.len() * 2);
        } else {
            let mut mask = vec![f32::NEG_INFINITY; nkv * tokens.len()];
            for q in 0..tokens.len() {
                for kv in 0..=(n_past + q).min(nkv - 1) {
                    mask[q * nkv + kv] = 0.0;
                }
            }
            ffi::ggml_backend_tensor_set(b.kq_mask, mask.as_ptr().cast(), 0, mask.len() * 4);
        }

        if !b.conv_zero.is_null() {
            let hp = &self.hp;
            let zc = vec![0f32; ((hp.d_conv - 1) * hp.conv_dim()) as usize];
            ffi::ggml_backend_tensor_set(b.conv_zero, zc.as_ptr().cast(), 0, zc.len() * 4);
            let zs = vec![0f32; (hp.gdn_head_v() * hp.gdn_head_v() * hp.n_v_heads) as usize];
            ffi::ggml_backend_tensor_set(b.state_zero, zs.as_ptr().cast(), 0, zs.len() * 4);
        }

        ffi::ggml_backend_tensor_set(
            b.out_ids,
            out_positions.as_ptr().cast(),
            0,
            out_positions.len() * 4,
        );
    }

    /// Assemble the tap features of a just-computed graph: per token, the
    /// tapped layers' input hiddens concatenated in tap-layer order.
    unsafe fn read_taps(&self, b: &Built, t: usize) -> Vec<f32> {
        if b.taps.is_empty() || t == 0 {
            return Vec::new();
        }
        let ne = self.hp.n_embd as usize;
        let nl = b.taps.len();
        let mut out = vec![0f32; t * nl * ne];
        let mut tmp = vec![0f32; t * ne];
        for (li, tap) in b.taps.iter().enumerate() {
            ffi::ggml_backend_tensor_get(*tap, tmp.as_mut_ptr().cast(), 0, t * ne * 4);
            for tok in 0..t {
                out[tok * nl * ne + li * ne..tok * nl * ne + (li + 1) * ne]
                    .copy_from_slice(&tmp[tok * ne..(tok + 1) * ne]);
            }
        }
        out
    }

    unsafe fn compute(&self, gf: *mut ffi::ggml_cgraph, n_threads: i32) -> Result<(), ModelError> {
        let backend = self.weights.backend();
        if self.weights.is_cpu() {
            ffi::ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
        let st = match self.weights.sched() {
            // multi-device: the scheduler owns placement and cross-device copies
            Some(sched) => ffi::ggml_backend_sched_graph_compute(sched, gf),
            None => ffi::ggml_backend_graph_compute(backend, gf),
        };
        if st != ffi::ggml_status_GGML_STATUS_SUCCESS {
            return Err(ModelError::Load(format!("graph compute status {st}")));
        }
        Ok(())
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
