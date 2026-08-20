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
//! default session path; parity is checked path-matched (tandem FA ↔ oracle
//! -fa on, tandem non-FA ↔ oracle -fa off).

use tandem_ggml_sys as ffi;
use tandem_gguf::Value;

use crate::{ModelError, Weights};

/// KV window bucket: decode graphs keep identical shapes while n_past + T
/// stays within the same bucket multiple. Padded cells are zero-initialized
/// and masked to -inf.
const KV_BUCKET_DEFAULT: i64 = 256;

/// Bucket granularity: bigger = fewer graph rebuilds but more masked-out
/// attention work per step; smaller = tighter attention but more rebuilds.
/// TANDEM_KV_BUCKET overrides for measurement.
fn kv_bucket() -> i64 {
    std::env::var("TANDEM_KV_BUCKET")
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
            attn_batch: on("TANDEM_DBG_ATTN_BATCH"),
            gdn_zero: on("TANDEM_DBG_GDN_ZERO"),
            no_writes: on("TANDEM_DBG_NO_WRITES"),
            k_batch: on("TANDEM_DBG_K_BATCH"),
            v_batch: on("TANDEM_DBG_V_BATCH"),
        }
    }
}

/// Either raw logits or an in-graph-sampled token id.
enum StepOut {
    Logits(Vec<f32>),
    Token(u32),
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
    inp_pos: *mut ffi::ggml_tensor,
    kq_mask: *mut ffi::ggml_tensor,
    /// set_rows write positions (cached decode graphs only)
    row_ids: *mut ffi::ggml_tensor,
    conv_zero: *mut ffi::ggml_tensor,
    state_zero: *mut ffi::ggml_tensor,
    out_ids: *mut ffi::ggml_tensor,
    out: *mut ffi::ggml_tensor,
    /// pre-LM-head hidden states at the requested positions (MTP input)
    h_out: *mut ffi::ggml_tensor,
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
pub struct Session {
    pub n_ctx_max: usize,
    pub n_past: usize,
    /// Recurrent-state snapshot slots: K = n_spec + 1. Slot 0 is live; slot s
    /// holds the state as of s tokens earlier. Speculative decoding advances
    /// the recurrent layers over draft tokens that may be rejected, and those
    /// layers cannot be run backwards — the snapshots ARE the undo.
    pub k_slots: i64,
    /// flash-attention path (untransposed V cache); TANDEM_NO_FA=1 disables
    pub fa: bool,
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
    /// how many tokens the MTP block has consumed
    pub mtp_past: usize,
    cached: Option<CachedStep>,
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
        let fa = std::env::var("TANDEM_NO_FA").is_err();
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
            let mtp_k = ffi::ggml_new_tensor_2d(
                ctx, f16t, hp.head_k * hp.n_head_kv, n_ctx_max as i64,
            );
            let mtp_v = if fa {
                ffi::ggml_new_tensor_2d(ctx, f16t, hp.head_v * hp.n_head_kv, n_ctx_max as i64)
            } else {
                ffi::ggml_new_tensor_2d(ctx, f16t, n_ctx_max as i64, hp.head_v * hp.n_head_kv)
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
                k_slots,
                fa,
                ctx,
                buffer,
                galloc,
                k_cache,
                v_cache,
                conv_state,
                gdn_state,
                mtp_k,
                mtp_v,
                mtp_past: 0,
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

    pub fn reset(&mut self) {
        unsafe {
            ffi::ggml_backend_buffer_clear(self.buffer, 0);
        }
        self.cached = None;
        self.n_past = 0;
        self.mtp_past = 0;
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.cached = None;
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
        Ok(Qwen35 { weights, hp })
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
            out_positions,
            n_threads,
            false,
        )? {
            StepOut::Logits(l) => Ok(l),
            StepOut::Token(_) => unreachable!(),
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
            StepOut::Token(_) => unreachable!("logits requested"),
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
        // Tensor parallelism splits the vocabulary across devices, so the
        // argmax has to happen after the logits are gathered host-side.
        let in_graph = !self.weights.is_tensor_parallel();
        match self.step_impl(session, tokens, &last, n_threads, in_graph)? {
            StepOut::Token(t) => Ok(t),
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
        let out = if cacheable && tokens.len() == 1 && out_positions == [0] && session.fa {
            self.step_cached(session, tokens[0], n_threads, greedy)?
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
                out_positions,
                n_threads,
                greedy,
            )?
        };
        session.n_past += tokens.len();
        Ok(out)
    }

    /// One-shot path: build graph, allocate (scratch or throwaway), fill,
    /// compute, read.
    fn run_general(
        &self,
        tokens: &[u32],
        state: StateSrc,
        galloc: Option<ffi::ggml_gallocr_t>,
        n_past: usize,
        out_positions: &[i32],
        n_threads: i32,
        greedy: bool,
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
            self.fill_inputs(&built, tokens, n_past, out_positions);
            self.compute(built.gf, n_threads)?;
            Ok(self.read_out(&built, out_positions.len()))
        }
    }

    unsafe fn read_out(&self, b: &Built, n_out: usize) -> StepOut {
        if b.greedy {
            let mut ids = vec![0i32; n_out];
            ffi::ggml_backend_tensor_get(b.out, ids.as_mut_ptr().cast(), 0, n_out * 4);
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
            let params = ffi::ggml_init_params {
                mem_size: (hp.n_layer * 6 + 16) * ffi::ggml_tensor_overhead()
                    + ffi::ggml_graph_overhead(),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(ModelError::Load("rollback ctx".into()));
            }
            struct G(*mut ffi::ggml_context);
            impl Drop for G {
                fn drop(&mut self) {
                    unsafe { ffi::ggml_free(self.0) }
                }
            }
            let _g = G(ctx);
            let gf = ffi::ggml_new_graph_custom(ctx, (hp.n_layer * 4 + 16) as usize, false);
            for il in 0..hp.n_layer {
                if !hp.is_recurrent(il) {
                    continue;
                }
                for t in [session.conv_state[il], session.gdn_state[il]] {
                    let slot_nb = if (*t).ne[3] > 1 { (*t).nb[3] } else { (*t).nb[2] };
                    let elems = ffi::ggml_nelements(t) / session.k_slots;
                    let src = ffi::ggml_view_1d(ctx, t, elems, n * slot_nb);
                    let dst = ffi::ggml_view_1d(ctx, t, elems, 0);
                    ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, src, dst));
                }
            }
            if let Some(sched) = self.weights.sched() {
                ffi::ggml_backend_sched_reset(sched);
                if !ffi::ggml_backend_sched_alloc_graph(sched, gf) {
                    return Err(ModelError::Load("rollback sched alloc".into()));
                }
            } else if !ffi::ggml_gallocr_alloc_graph(session.galloc, gf) {
                return Err(ModelError::Load("rollback alloc".into()));
            }
            self.compute(gf, n_threads)?;
        }
        Ok(())
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
            self.fill_inputs(&built, tokens, session.n_past, out_positions);
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
        unsafe {
            let g = self.build_mtp(
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

            self.compute(g.gf, n_threads)?;
            let mut logits = vec![0f32; self.hp.n_vocab as usize];
            ffi::ggml_backend_tensor_get(g.out, logits.as_mut_ptr().cast(), 0, logits.len() * 4);
            // the draft head's own hidden: chained drafts must consume this,
            // not the trunk's, or every draft past the first is predicted
            // from a stale state (measured: acceptance 0.40 vs 0.72 at depth 3)
            let mut h = vec![0f32; self.hp.n_embd as usize];
            ffi::ggml_backend_tensor_get(g.h_out, h.as_mut_ptr().cast(), 0, h.len() * 4);
            session.mtp_past += 1;
            Ok((logits, h))
        }
    }

    /// Cached T=1 decode: reuse the per-bucket graph + frozen allocation.
    fn step_cached(
        &self,
        session: &mut Session,
        token: u32,
        n_threads: i32,
        greedy: bool,
    ) -> Result<StepOut, ModelError> {
        let n_kv_exact = (session.n_past + 1) as i64;
        let kvb = kv_bucket();
        let bucket = (((n_kv_exact + kvb - 1) / kvb) * kvb).min(session.n_ctx_max as i64);
        let stale = session
            .cached
            .as_ref()
            .map(|c| c.bucket != bucket || c.greedy != greedy)
            .unwrap_or(true);
        if stale {
            unsafe {
                let built = self.build_inner(
                    1,
                    bucket,
                    &StateSrc::Session(session.view()),
                    1,
                    true,
                    0,
                    greedy,
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
                let host = HostStaging {
                    tokens: vec![0i32; 1],
                    pos: vec![0i32; 4],
                    mask_f16: vec![0xFC00u16; bucket as usize],
                    row_ids: vec![0i64; 1],
                    mask_visible: 0,
                    out_ids: vec![0i32; 1],
                };
                session.cached = Some(CachedStep { built, galloc, bucket, greedy, host });
            }
        }
        let n_past = session.n_past;
        let cached = session.cached.as_mut().unwrap();
        unsafe {
            let backend = self.weights.backend();
            let b = &cached.built;
            let h = &mut cached.host;

            // Inputs are uploaded ASYNC into the compute stream and never read
            // back before the graph runs, so no per-upload device sync: the
            // stream ordering is the synchronization. Only the final output
            // read (4 bytes for greedy) synchronizes.
            h.tokens[0] = token as i32;
            ffi::ggml_backend_tensor_set_async(backend, b.inp_tokens, h.tokens.as_ptr().cast(), 0, 4);

            let p = n_past as i32;
            h.pos[0] = p;
            h.pos[1] = p;
            h.pos[2] = p;
            h.pos[3] = 0;
            ffi::ggml_backend_tensor_set_async(backend, b.inp_pos, h.pos.as_ptr().cast(), 0, 16);

            h.row_ids[0] = n_past as i64;
            ffi::ggml_backend_tensor_set_async(backend, b.row_ids, h.row_ids.as_ptr().cast(), 0, 8);

            // EVERY input is re-uploaded before EVERY compute. ggml's graph
            // allocator may place intermediates over an input's storage once
            // that input's last consumer has run, so device-side values do
            // NOT survive between computes of the same graph. Skipping a
            // re-upload (out_ids "is constant"; the mask "only changed by one
            // cell") clobbered the out_ids index and aborted get_rows at
            // n_past 2333 — silent until the allocation happened to overlap.
            let want_visible = (n_past + 1).min(h.mask_f16.len());
            for c in h.mask_visible..want_visible {
                h.mask_f16[c] = 0;
            }
            h.mask_visible = want_visible;
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
                4,
            );

            self.compute(b.gf, n_threads)?;
            Ok(self.read_out(b, 1))
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
    ) -> Result<Built, ModelError> {
        let n_kv_exact = n_past as i64 + t_len;
        let n_kv = match state {
            StateSrc::Stateless => n_kv_exact,
            StateSrc::Session(s) => {
                let kvb = kv_bucket();
                (((n_kv_exact + kvb - 1) / kvb) * kvb).min(s.n_ctx_max as i64)
            }
        };
        self.build_inner(t_len, n_kv, state, n_out, use_set_rows, n_past, greedy)
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
    ) -> Result<Built, ModelError> {
        let hp = &self.hp;

        // --- CUDA-bisect toggles (kept for kernel-path debugging) ---
        let dbg = DebugToggles::from_env();
        let dbg_attn_batch = dbg.attn_batch;
        let dbg_gdn_zero = dbg.gdn_zero;
        let dbg_no_writes = dbg.no_writes;

        let params = ffi::ggml_init_params {
            mem_size: 64 << 20,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return Err(ModelError::Load("graph ctx init".into()));
        }

        let gf = ffi::ggml_new_graph_custom(ctx, 8192, false);
        let f32t = ffi::ggml_type_GGML_TYPE_F32;

        // ---- inputs ----
        let inp_tokens = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, t_len);
        ffi::ggml_set_input(inp_tokens);
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
        let mut inp_l = ffi::ggml_get_rows(ctx, tok_embd, inp_tokens);

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

                let qkv_mixed = ffi::ggml_mul_mat(ctx, l.wqkv, cur);
                let qkv_mixed = ffi::ggml_reshape_3d(ctx, qkv_mixed, hp.conv_dim(), t_len, 1);
                let z = ffi::ggml_mul_mat(ctx, l.wqkv_gate, cur);

                let beta = ffi::ggml_mul_mat(ctx, l.ssm_beta, cur);
                let beta = ffi::ggml_reshape_4d(ctx, beta, 1, hp.n_v_heads, t_len, 1);
                let beta = ffi::ggml_sigmoid(ctx, beta);

                let alpha = ffi::ggml_mul_mat(ctx, l.ssm_alpha, cur);
                let alpha = ffi::ggml_reshape_3d(ctx, alpha, hp.n_v_heads, t_len, 1);
                let alpha = ffi::ggml_add(ctx, alpha, as_f32(ctx, l.dt_bias));
                let alpha = ffi::ggml_softplus(ctx, alpha);
                let g = ffi::ggml_mul(ctx, alpha, as_f32(ctx, l.ssm_a));
                let g = ffi::ggml_reshape_4d(ctx, g, 1, hp.n_v_heads, t_len, 1);

                let (conv_in_state, gdn_in_state) = match state {
                    StateSrc::Stateless => (conv_zero, state_zero),
                    StateSrc::Session(_) if dbg_gdn_zero => (conv_zero, state_zero),
                    StateSrc::Session(s) => {
                        // slot 0 is the live state
                        let c3 = ffi::ggml_view_3d(
                            ctx, s.conv_state[il], hp.d_conv - 1, hp.conv_dim(), 1,
                            (*s.conv_state[il]).nb[1], (*s.conv_state[il]).nb[2], 0,
                        );
                        let s4 = ffi::ggml_view_4d(
                            ctx, s.gdn_state[il], head_v, head_v, hp.n_v_heads, 1,
                            (*s.gdn_state[il]).nb[1], (*s.gdn_state[il]).nb[2],
                            (*s.gdn_state[il]).nb[3], 0,
                        );
                        (c3, s4)
                    }
                };

                let qkv_t = ffi::ggml_transpose(ctx, qkv_mixed);
                let conv_input = ffi::ggml_concat(ctx, conv_in_state, qkv_t, 0);

                if let (StateSrc::Session(s), false) = (state, dbg_no_writes || dbg_gdn_zero) {
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

                let k_snap = match state {
                    StateSrc::Session(s) => s.k_slots,
                    StateSrc::Stateless => 1,
                };
                let gdn =
                    ffi::ggml_gated_delta_net(ctx, q, k, v, g, beta, gdn_in_state, k_snap);

                let out = ffi::ggml_view_4d(
                    ctx, gdn,
                    head_v, hp.n_v_heads, t_len, 1,
                    row(head_v), row(head_v * hp.n_v_heads),
                    row(head_v * hp.n_v_heads * t_len), 0,
                );

                if let (StateSrc::Session(s), false) = (state, dbg_no_writes || dbg_gdn_zero) {
                    // The op packs K state snapshots after the attention rows,
                    // most recent first; copy the written ones into the
                    // session's matching slots.
                    let d = head_v * head_v * hp.n_v_heads;
                    let n_written = s.k_slots.min(t_len);
                    let src = ffi::ggml_view_3d(
                        ctx, gdn, d, 1, n_written,
                        row(d), row(d),
                        row(head_v * hp.n_v_heads * t_len),
                    );
                    let dst = ffi::ggml_view_3d(
                        ctx, s.gdn_state[il], d, 1, n_written,
                        row(d), row(d), 0,
                    );
                    ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx, src, dst));
                }

                let z4 = ffi::ggml_reshape_4d(ctx, z, head_v, hp.n_v_heads, t_len, 1);
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
        if greedy {
            // Sample in the graph: the readback becomes 4 bytes instead of
            // n_vocab floats (993 KB for this model), removing a full
            // device sync + PCIe transfer from every decode step. Exact for
            // temp-0: ggml_argmax selects the same element CPU argmax would.
            cur = ffi::ggml_argmax(ctx, cur);
        }
        ffi::ggml_set_output(cur);
        ffi::ggml_build_forward_expand(gf, cur);

        Ok(Built {
            ctx,
            gf,
            inp_tokens,
            inp_pos,
            kq_mask,
            row_ids,
            conv_zero,
            state_zero,
            out_ids,
            out: cur,
            h_out: h_sel,
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

    unsafe fn fill_inputs(&self, b: &Built, tokens: &[u32], n_past: usize, out_positions: &[i32]) {
        let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        ffi::ggml_backend_tensor_set(b.inp_tokens, toks_i32.as_ptr().cast(), 0, tokens.len() * 4);

        let mut pos = vec![0i32; tokens.len() * 4];
        for i in 0..tokens.len() {
            let p = (n_past + i) as i32;
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
