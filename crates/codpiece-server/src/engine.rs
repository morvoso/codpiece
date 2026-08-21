//! The generation worker.
//!
//! The model and its session hold raw ggml pointers and are not `Send`, so the engine
//! owns a thread that loads the model itself and never lets it escape. Requests arrive
//! as jobs on a channel and are served one at a time; that is the honest shape of the
//! engine today, and it is the seam where continuous batching replaces the loop with a
//! scheduler over several sequences.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::sync::Arc;

use codpiece_model::dflash::{Dflash, DflashCache};
use codpiece_model::qwen35::{Qwen35, Session};
use codpiece_sample::{Sampler, SamplerParams};
use codpiece_vision::preprocess::{PreparedImage, Preprocessor};
use codpiece_vision::VisionModel;


/// How many prompt tokens to prefill per graph.
///
/// Prefill is compute-bound, so a bigger chunk keeps the GPU busier — at `-c 131072` a
/// 33.5K prompt prefills at 1331 tok/s with 512 against 849 with 64. But the attention
/// mask is `n_kv x chunk`, and at a context where the KV cache has already taken the
/// card there is nowhere to put it: at `-c 200704` the same comparison inverts, 815
/// tok/s with 64 against 685 with 334. So the chunk follows the headroom left after the
/// weights and the cache, not the context length.
///
/// A two-point fit, and honest about it: the two measurements above are what set the
/// threshold.
/// Headroom on the tightest card once everything resident is allocated,
/// in GiB. Recorded after startup; `None` until then.
static FREE_GIB: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

/// How many prompt tokens to push through the model at once.
///
/// A prefill graph's compute buffer scales with `n_kv * chunk`, so the chunk
/// is bounded by real headroom rather than a guess. It used to be derived
/// from a static estimate of weights and KV, which knew nothing about the
/// drafter (~1.2 GiB, mirrored on both cards) or the vision tower (~0.87
/// GiB) — at 98K context that estimate claimed 4 GiB free where the driver
/// reported 1.15, and picked the largest chunk on the strength of it.
pub fn prefill_chunk_for(n_ctx: usize) -> usize {
    if let Some(n) = std::env::var("CODPIECE_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        return n;
    }
    let free = FREE_GIB.get().copied().unwrap_or_else(|| {
        const WEIGHTS_GIB: f64 = 16.46; // per card, incl. mirrored head and embeddings
        const CARD_GIB: f64 = 23.5;
        CARD_GIB - WEIGHTS_GIB - n_ctx as f64 * 65536.0 / 2.0 / (1u64 << 30) as f64
    });
    match free {
        f if f > 2.5 => 512,
        f if f > 1.5 => 256,
        f if f > 0.9 => 128,
        _ => 64,
    }
}

/// Record real headroom once everything resident is allocated, and give the
/// graph cache a budget scaled to it.
///
/// The cache may hold 40% of what the tightest card has free, clamped to
/// [256 MiB, 1 GiB]. The rest is left for the transient buffers a request
/// allocates while it runs — a cache budget larger than the card can spare
/// does not prevent an out-of-memory failure, it only moves it from eviction
/// time to compute time.
fn record_free_vram() {
    let free_bytes = codpiece_model::device_memory()
        .iter()
        .map(|(_, used, total)| total.saturating_sub(*used))
        .min()
        .unwrap_or(0);
    let _ = FREE_GIB.set(free_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    let budget = ((free_bytes as f64 * 0.4) as usize)
        .clamp(256 * 1024 * 1024, 1024 * 1024 * 1024);
    codpiece_model::qwen35::set_graph_cache_budget(budget);
    eprintln!("serve: graph cache budget {} MiB", budget / (1024 * 1024));
}

/// How deep to draft next, given how confident the model just was.
fn gated_depth(depth: usize, gate: f32, peak: f32) -> usize {
    if gate <= 0.0 || peak >= gate {
        depth
    } else {
        1
    }
}


/// Host-RAM store of suspended conversations.
///
/// Bounded by bytes, evicted least-recently-used. A stored snapshot is ~64 KiB per
/// token of conversation plus ~160 MB of recurrent state, so a 32K conversation is
/// about 2.2 GiB — the default budget holds a handful, which matches how many long
/// conversations one person actually alternates between.
struct SessionStore {
    entries: Vec<StoredSession>,
    budget: usize,
    clock: u64,
}

struct StoredSession {
    history: Vec<u32>,
    snap: codpiece_model::qwen35::SessionSnapshot,
    last_used: u64,
}

impl SessionStore {
    fn new() -> Self {
        let gib = std::env::var("CODPIECE_SESSION_CACHE_GIB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(12);
        Self { entries: Vec::new(), budget: gib << 30, clock: 0 }
    }

    fn save(&mut self, history: &[u32], snap: codpiece_model::qwen35::SessionSnapshot) {
        self.clock += 1;
        // an older entry that this conversation extends is superseded by it
        self.entries
            .retain(|e| !(e.history.len() <= history.len() && history[..e.history.len()] == e.history[..]));
        self.entries.push(StoredSession {
            history: history.to_vec(),
            snap,
            last_used: self.clock,
        });
        let mut total: usize = self.entries.iter().map(|e| e.snap.nbytes()).sum();
        while total > self.budget && self.entries.len() > 1 {
            let (i, _) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .expect("non-empty");
            total -= self.entries[i].snap.nbytes();
            self.entries.remove(i);
        }
    }

    /// The stored conversation with the longest full-prefix match against `prompt`,
    /// leaving at least one token to prefill.
    fn best_prefix(&mut self, prompt: &[u32]) -> Option<&StoredSession> {
        self.clock += 1;
        let clock = self.clock;
        let best = self
            .entries
            .iter_mut()
            .filter(|e| {
                e.history.len() < prompt.len() && prompt[..e.history.len()] == e.history[..]
            })
            .max_by_key(|e| e.history.len())?;
        best.last_used = clock;
        Some(&*best)
    }
}

pub struct GenRequest {
    pub prompt: String,
    /// Pre-tokenized prompt, when the client sent token ids rather than text.
    /// Re-tokenizing text a harness already tokenized can shift token
    /// boundaries, which silently changes what a loglikelihood score means.
    pub prompt_ids: Option<Vec<u32>>,
    /// Preprocessed images, in the order their markers appear in the prompt.
    /// Each `<|image_pad|>` token in the prompt consumes the next entry.
    pub images: Vec<PreparedImage>,
    pub params: SamplerParams,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    /// Tokens the model may spend inside a `<think>` block before it is forced to
    /// stop and answer. 0 disables. This is a safety net, not a quality knob: without
    /// it the model can think until it exhausts `max_tokens` and return an empty
    /// answer — a pathology production caps at 4096. `think_close` is the token
    /// sequence that ends the block, provided by the caller who has the tokenizer.
    pub think_budget: usize,
    pub think_close: Vec<u32>,
    /// OpenAI `echo` + `logprobs`: score the prompt itself and return a
    /// logprob per token, plus this many alternatives per position. This is
    /// what loglikelihood benchmarks (MMLU, HellaSwag, ARC) run on — they
    /// never generate. Scoring runs instead of generation, not before it.
    pub echo_logprobs: Option<usize>,
}

#[derive(Debug)]
pub enum Event {
    /// prompt token count, once, before any text
    Prefilled { n_prompt: usize },
    Token(String),
    /// the scored prompt, when `echo_logprobs` was set
    Scored(Scored),
    Done(Finish),
    Failed(String),
}

/// Per-token scores for an echoed prompt, in OpenAI's `logprobs` shape. The
/// first token has no score — nothing precedes it to predict it.
#[derive(Debug, Default)]
pub struct Scored {
    /// the echoed prompt, decoded as a whole so it is byte-exact
    pub text: String,
    pub tokens: Vec<String>,
    pub logprobs: Vec<Option<f32>>,
    pub top: Vec<Option<Vec<(String, f32)>>>,
    /// byte offset of each token in the echoed text
    pub text_offset: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Finish {
    pub reason: &'static str,
    pub n_prompt: usize,
    pub n_generated: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
    /// accepted / proposed when the round was speculative
    pub acceptance: Option<f64>,
    /// drafts proposed, drafts kept, and speculative rounds run — the names match
    /// llama.cpp's `timings` so existing tooling reads them without changes
    pub draft_n: usize,
    pub draft_n_accepted: usize,
    pub n_draft_calls: usize,
}

struct Job {
    req: GenRequest,
    out: SyncSender<Event>,
}

#[derive(Clone)]
pub struct Engine {
    tx: Sender<Job>,
    pub stats: Arc<Stats>,
    pub model_name: String,
    pub n_ctx: usize,
    /// Set when a vision tower is loaded: the API layer preprocesses images
    /// with it (CPU work, off the engine thread) before submitting.
    pub vision_prep: Option<Preprocessor>,
}

#[derive(Default)]
pub struct Stats {
    pub queued: AtomicUsize,
    pub processing: AtomicUsize,
    pub served: AtomicU64,
    pub tokens_generated: AtomicU64,
    /// Resident conversation slots, published by the engine thread once the
    /// pool is built — the count depends on how much VRAM was left, so it is
    /// not knowable when the Engine handle is created.
    pub session_slots: AtomicUsize,
}

pub struct EngineConfig {
    pub model_path: String,
    /// DFlash2 draft model (GGUF); None keeps the MTP-head drafter.
    pub dflash: Option<String>,
    /// Vision tower (mmproj GGUF); None serves text only.
    pub mmproj: Option<String>,
    /// Device for the vision tower: a CUDA ordinal, or None for the CPU.
    /// The tower is ~0.9 GiB of weights plus a few hundred MB of compute
    /// buffer at encode time — VRAM that must fit beside the trunk.
    pub mmproj_gpu: Option<i32>,
    pub n_ctx: usize,
    pub threads: i32,
    pub tp: Option<Vec<i32>>,
    pub gpu: Option<i32>,
    pub depth: usize,
    /// Ceiling for the adaptive draft depth.
    pub max_depth: usize,
    /// Below this confidence, stop drafting deep for a round.
    ///
    /// Production runs llama.cpp with `--spec-draft-p-min 0.75`, which declines to
    /// draft when the draft head is unsure. This is the same idea measured off a signal
    /// codpiece already has: how peaked the *target* distribution was at the position it
    /// just sampled. A flat distribution means the text is unpredictable here, and
    /// under the rejection rule an unlikely draft is rejected in proportion — so the
    /// chain steps behind it are spent for nothing. 0 disables.
    pub draft_gate: f32,
}

/// The vision tower plus everything the engine needs to splice images into
/// a token stream.
struct VisionCtx {
    model: VisionModel,
    /// patch * spatial_merge: one trunk row covers align x align pixels
    align: u32,
    pad_id: u32,
}

/// Image spans in the expanded stream carry PSEUDO ids derived from the image
/// content hash — far above any real vocab id (vocab is 248320; these are all
/// >= 2^31). They never reach the model (image rows go through step_embd);
/// they exist so prefix reuse over `history` treats an image span as matched
/// only when the image bytes matched.
fn img_pseudo_id(hash: u64, j: usize) -> u32 {
    let mix = (hash ^ (hash >> 32)) as u32;
    0x8000_0000 | (mix.wrapping_mul(0x9E37_79B9).wrapping_add(j as u32) & 0x7FFF_FFFF)
}

/// One stretch of the expanded prompt: text token indices, or one image.
enum Seg {
    /// range of indices into the expanded id stream holding REAL token ids
    Text(std::ops::Range<usize>),
    /// image index into `GenRequest::images`; its span in the stream is
    /// `start..start + n_tokens`
    Image { idx: usize, start: usize },
}

/// Replace each `<|image_pad|>` in the tokenized prompt with the image's
/// pseudo-id span and record the segment layout for prefill.
fn expand_prompt(
    raw: &[u32],
    pad_id: u32,
    align: u32,
    images: &[PreparedImage],
) -> Result<(Vec<u32>, Vec<Seg>), String> {
    let mut ids = Vec::with_capacity(raw.len());
    let mut segs: Vec<Seg> = Vec::new();
    let mut text_start = 0usize;
    let mut next_img = 0usize;
    for &t in raw {
        if t == pad_id {
            let img = images.get(next_img).ok_or_else(|| {
                format!("prompt has more image markers than images ({})", images.len())
            })?;
            if ids.len() > text_start {
                segs.push(Seg::Text(text_start..ids.len()));
            }
            let n = img.n_tokens(align);
            if n == 0 {
                return Err("image resolved to zero tokens".into());
            }
            let start = ids.len();
            for j in 0..n {
                ids.push(img_pseudo_id(img.hash, j));
            }
            segs.push(Seg::Image { idx: next_img, start });
            next_img += 1;
            text_start = ids.len();
        } else {
            ids.push(t);
        }
    }
    if next_img != images.len() {
        return Err(format!(
            "{} images supplied but {} markers in the prompt",
            images.len(),
            next_img
        ));
    }
    if ids.len() > text_start {
        segs.push(Seg::Text(text_start..ids.len()));
    }
    Ok((ids, segs))
}

impl Engine {
    /// Load the model on a worker thread and return once it is ready to serve.
    pub fn start(cfg: EngineConfig) -> Result<(Self, String), String> {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        #[allow(clippy::type_complexity)]
        let (ready_tx, ready_rx) =
            std::sync::mpsc::channel::<Result<(String, Option<Preprocessor>), String>>();
        let stats = Arc::new(Stats::default());
        let stats_worker = stats.clone();
        let name = std::path::Path::new(&cfg.model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());
        let n_ctx = cfg.n_ctx;
        std::thread::Builder::new()
            .name("codpiece-engine".into())
            .spawn(move || worker(cfg, rx, ready_tx, stats_worker))
            .map_err(|e| format!("engine thread: {e}"))?;
        let (template, vision_prep) = ready_rx
            .recv()
            .map_err(|_| "engine thread died during load".to_string())??;
        Ok((Self { tx, stats, model_name: name, n_ctx, vision_prep }, template))
    }

    /// Queue a request. Events stream back on the returned receiver.
    pub fn submit(&self, req: GenRequest) -> Receiver<Event> {
        // A bounded channel applies backpressure: if a client stops reading, the worker
        // blocks on that client rather than generating into an unbounded buffer.
        let (out, rx) = std::sync::mpsc::sync_channel(64);
        self.stats.queued.fetch_add(1, Ordering::Relaxed);
        if self.tx.send(Job { req, out: out.clone() }).is_err() {
            self.stats.queued.fetch_sub(1, Ordering::Relaxed);
            let _ = out.send(Event::Failed("engine is not running".into()));
        }
        rx
    }
}

fn worker(
    cfg: EngineConfig,
    rx: Receiver<Job>,
    ready: Sender<Result<(String, Option<Preprocessor>), String>>,
    stats: Arc<Stats>,
) {
    let device = match (&cfg.tp, cfg.gpu) {
        (Some(ids), _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None) => codpiece_model::Device::Cpu,
    };
    let mut model = match Qwen35::load_on(std::path::Path::new(&cfg.model_path), device) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(format!("load: {e}")));
            return;
        }
    };
    // The drafter loads into the trunk's backend and turns on the trunk's
    // layer taps — BEFORE any session or graph exists, so every shape is
    // built with the taps attached.
    let dflash: Option<Dflash> = match &cfg.dflash {
        Some(p) => match Dflash::load_into(&mut model, std::path::Path::new(p)) {
            Ok(d) => {
                model.tap_layers = d.hp.target_layers.clone();
                eprintln!(
                    "serve: DFlash2 drafter up ({} layers, block {}, window {}, taps {:?})",
                    d.hp.n_layer, d.hp.block_size, d.hp.n_swa, d.hp.target_layers
                );
                Some(d)
            }
            Err(e) => {
                let _ = ready.send(Err(format!("dflash: {e}")));
                return;
            }
        },
        None => None,
    };
    let model = model;
    let tok = match codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf) {
        Ok(t) => t,
        Err(e) => {
            let _ = ready.send(Err(format!("tokenizer: {e}")));
            return;
        }
    };
    let template = model
        .weights
        .gguf
        .kv("tokenizer.chat_template")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    // The vision tower loads before ready so a broken mmproj fails startup
    // loudly instead of 500ing the first image request.
    let vision: Option<VisionCtx> = match &cfg.mmproj {
        Some(path) => {
            let dev = match cfg.mmproj_gpu {
                Some(i) => codpiece_model::Device::Cuda(i),
                None => codpiece_model::Device::Cpu,
            };
            let vm = match VisionModel::load(std::path::Path::new(path), dev) {
                Ok(v) => v,
                Err(e) => {
                    let _ = ready.send(Err(format!("mmproj: {e:?}")));
                    return;
                }
            };
            let pad = tok.encode("<|image_pad|>", true);
            if pad.len() != 1 {
                let _ = ready.send(Err(
                    "tokenizer has no <|image_pad|> token; cannot serve vision".into(),
                ));
                return;
            }
            let align = (vm.hp.patch * vm.hp.merge) as u32;
            eprintln!(
                "serve: vision tower up ({} layers, d={}, align {}px/token, {})",
                vm.hp.n_layer,
                vm.hp.n_embd,
                align,
                match cfg.mmproj_gpu {
                    Some(i) => format!("cuda:{i}"),
                    None => "cpu".into(),
                }
            );
            Some(VisionCtx { model: vm, align, pad_id: pad[0] })
        }
        None => None,
    };
    // One session for the life of the process, reset between requests. Building it per
    // request meant allocating and freeing the KV cache each time — 4+ GiB at a long
    // context — and `cudaFree` synchronises, so the teardown outlived the response and
    // left `/slots` reporting the slot busy after the client already had its answer.
    // Slots have to cover the deepest chain that can ever be chosen, not the configured
    // one: with adaptive depth `cfg.depth` is 0 and the picker may still ask for
    // `max_depth`, and a rollback deeper than the slots it has is an error, not a
    // degradation.
    // Not every model has a draft head; without one the fused round cannot be built and
    // decoding falls back to one token per step.
    // depth 0 means the caller asked for no speculation at all
    let can_speculate = dflash.is_some() || (model.has_mtp() && cfg.depth != 0);
    if !can_speculate {
        eprintln!("serve: model has no MTP head; decoding without speculation");
    }
    let slots = if cfg.depth == usize::MAX { cfg.max_depth } else { cfg.depth }
        .max(cfg.max_depth)
        .max(1)
        // a rejected DFlash block rolls back up to block_size-1 tokens
        .max(dflash.as_ref().map(|d| d.hp.block_size - 1).unwrap_or(0));
    let session = match Session::new_spec(&model, cfg.n_ctx, slots) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!("session: {e}")));
            return;
        }
    };
    // A pool of sessions, each holding one conversation's state in VRAM, plus the
    // history of tokens that state covers. Switching conversations is a pointer choice,
    // not a copy — which matters doubly under tensor parallelism, where the split GDN
    // state cannot be copied off-device at all. Pool size is bounded by VRAM: each
    // extra session costs its KV cache (~2 GiB per card at 64K context), so the count
    // shrinks as the context grows.
    let n_sessions = std::env::var("CODPIECE_SESSIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(if cfg.n_ctx <= 70_000 { 2 } else { 1 });
    let mut pool: Vec<(Session, Vec<u32>, u64)> = vec![(session, Vec::new(), 0)];
    for _ in 1..n_sessions {
        match Session::new_spec(&model, cfg.n_ctx, slots) {
            Ok(s) => pool.push((s, Vec::new(), 0)),
            Err(e) => {
                eprintln!("serve: session pool capped at {}: {e}", pool.len());
                break;
            }
        }
    }
    eprintln!("serve: {} session slot(s)", pool.len());
    stats.session_slots.store(pool.len(), Ordering::Relaxed);
    report_vram("after weights, sessions, drafter, vision");
    record_free_vram();
    eprintln!("serve: prefill chunk {} tokens", prefill_chunk_for(cfg.n_ctx));

    // The image size cap is decided here, not earlier, because it depends on
    // headroom that only exists once every session is allocated. Readiness is
    // published after it for the same reason: the port opening should mean the
    // server is done taking memory, not that it is about to.
    let batch_slots = std::env::var("CODPIECE_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    // Per-slot KV region. 8192 suits long jobs; concurrent short-request
    // fleets (the vLLM-style workload) fit far more slots at 2-4K, and slots
    // are what aggregate throughput scales with while rounds stay
    // memory-bound (measured 16.5 -> 16.3 tok/s/seq from 8-way to 12-way).
    let batch_seq_ctx = std::env::var("CODPIECE_BATCH_CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8192)
        .min(cfg.n_ctx);

    // What the batch session will claim when concurrency first happens. The
    // encoder has to be capped as though that memory were already gone,
    // because it will be by the time an image and a busy server coincide.
    let batch_reserve = if batch_slots > 1 {
        let devs = codpiece_model::device_memory().len().max(1);
        let need = codpiece_model::qwen35::session_bytes(
            &model,
            batch_slots * batch_seq_ctx,
            batch_slots - 1,
        ) / devs;
        // Reserve only if it could ever be created. Headroom only shrinks
        // from here, so a batch session that does not fit now never will —
        // and reserving for it would needlessly shrink every image.
        let free = FREE_GIB.get().copied().unwrap_or(0.0) * (1u64 << 30) as f64;
        if (need + need / 4) as f64 <= free { need } else { 0 }
    } else {
        0
    };

    let vision_prep = vision.as_ref().map(|v| {
        let max_t = std::env::var("CODPIECE_IMAGE_MAX_TOKENS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or_else(|| vision_token_cap(batch_reserve));
        // prod llama.cpp runs --image-min-tokens 1024 (Qwen-VL grounding
        // degrades below that), so that is the default here too — but a floor
        // above the ceiling is nonsense, so the floor yields.
        let min_t = std::env::var("CODPIECE_IMAGE_MIN_TOKENS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(1024)
            .min(max_t);
        eprintln!("serve: image tokens {min_t}..{max_t}");
        Preprocessor::new(&v.model.hp, Some(min_t), Some(max_t))
    });
    if ready.send(Ok((template, vision_prep))).is_err() {
        return;
    }
    // CODPIECE_TRACE_VRAM=1: a line per request, which is how cached-graph
    // growth becomes visible instead of showing up as a dead process.
    let trace_vram = std::env::var("CODPIECE_TRACE_VRAM").is_ok_and(|v| v == "1");
    let mut dcaches: Vec<Option<DflashCache>> = (0..pool.len())
        .map(|_| match &dflash {
            Some(d) => d.new_cache().ok(),
            None => None,
        })
        .collect();
    let mut store = SessionStore::new();
    let mut clock = 0u64;
    // The batch session is created the first time two requests overlap: it costs
    // ~2.6 GiB of VRAM, and a single-user box that never overlaps never pays it.
    let mut bsession: Option<Session> = None;
    // Once the batch session has been declined for want of VRAM, stop
    // re-pricing it on every overlapping request.
    let mut batch_refused = false;
    let mut pending: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
    'serve: while let Ok(mut job) = pending.pop_front().map(Ok).unwrap_or_else(|| rx.recv()) {
        // A request that arrives while another is queued means real concurrency:
        // serve the group through the batch path so neither waits for the other's
        // whole generation.
        while let Ok(next) = rx.try_recv() {
            pending.push_back(next);
        }
        if !pending.is_empty()
            && batch_slots > 1
            && job.req.images.is_empty()
            && fits_a_slot(&tok, &mut job.req, batch_seq_ctx)
        {
            if bsession.is_none() && !batch_refused && !fits_in_vram(
                &model,
                batch_slots * batch_seq_ctx,
                batch_slots - 1,
                "batch session",
            ) {
                // Serving serially is slower; taking the process down is worse.
                batch_refused = true;
            }
            if bsession.is_none() && !batch_refused {
                match Session::new_spec(&model, batch_slots * batch_seq_ctx, batch_slots - 1) {
                    Ok(s) => {
                        bsession = Some(s);
                        report_vram("after batch session");
                    }
                    Err(e) => {
                        eprintln!("serve: batch session unavailable ({e}); serving serially");
                    }
                }
            }
            if let Some(bs) = bsession.as_mut() {
                run_batch(
                    &model, &tok, &cfg, bs, batch_seq_ctx, batch_slots, job, &rx,
                    &mut pending, &stats,
                );
                continue 'serve;
            }
        }
        stats.queued.fetch_sub(1, Ordering::Relaxed);
        stats.processing.fetch_add(1, Ordering::Relaxed);
        let out = job.out.clone();
        // Longest full-prefix match wins the slot; with no match, evict the least
        // recently used conversation.
        let prompt_ids = {
            let raw = match &job.req.prompt_ids {
                Some(ids) => ids.clone(),
                None => tok.encode(&job.req.prompt, true),
            };
            match (&vision, job.req.images.is_empty()) {
                (Some(v), false) => {
                    expand_prompt(&raw, v.pad_id, v.align, &job.req.images)
                        .map(|(ids, _)| ids)
                        // run_job reports the error properly; match on nothing here
                        .unwrap_or_default()
                }
                _ => raw,
            }
        };
        clock += 1;
        let slot = pool
            .iter()
            .position(|(_, h, _)| {
                !h.is_empty() && h.len() < prompt_ids.len() && prompt_ids[..h.len()] == h[..]
            })
            .unwrap_or_else(|| {
                // A miss with a plausible-looking pool is worth explaining: prefix
                // reuse lives or dies on token-exact matches, and the usual killer is
                // a BPE merge across the boundary between a cached prompt and text
                // appended to it.
                if std::env::var("CODPIECE_TRACE_REUSE").as_deref() == Ok("1") {
                    for (i, (_, h, _)) in pool.iter().enumerate() {
                        if h.is_empty() {
                            continue;
                        }
                        let div = h
                            .iter()
                            .zip(&prompt_ids)
                            .position(|(a, b)| a != b)
                            .unwrap_or(h.len().min(prompt_ids.len()));
                        eprintln!(
                            "[reuse] miss slot {i}: history {} tok, prompt {} tok, diverge at {div}                              (history {:?} vs prompt {:?})",
                            h.len(),
                            prompt_ids.len(),
                            &h[div.saturating_sub(3)..(div + 3).min(h.len())],
                            &prompt_ids[div.saturating_sub(3)..(div + 3).min(prompt_ids.len())]
                        );
                    }
                }
                pool.iter()
                    .enumerate()
                    .min_by_key(|(_, (_, _, used))| *used)
                    .map(|(i, _)| i)
                    .expect("pool is never empty")
            });
        let (session, history, used) = &mut pool[slot];
        *used = clock;
        // Scoring replaces generation rather than preceding it: it needs
        // logits at every position, so it runs from a reset state.
        if let Some(top_k) = job.req.echo_logprobs {
            let t0 = std::time::Instant::now();
            let n = prompt_ids.len();
            let outcome = if n == 0 {
                Err("cannot score an empty prompt".to_string())
            } else if n > cfg.n_ctx {
                Err(format!("prompt of {n} tokens exceeds the {} token context", cfg.n_ctx))
            } else {
                run_score(&model, &tok, &cfg, session, history, &prompt_ids, top_k)
            };
            stats.processing.fetch_sub(1, Ordering::Relaxed);
            stats.served.fetch_add(1, Ordering::Relaxed);
            match outcome {
                Ok(s) => {
                    let _ = out.send(Event::Scored(s));
                    let _ = out.send(Event::Done(Finish {
                        reason: "length",
                        n_prompt: n,
                        n_generated: 0,
                        prefill_s: t0.elapsed().as_secs_f64(),
                        decode_s: 0.0,
                        acceptance: None,
                        draft_n: 0,
                        draft_n_accepted: 0,
                        n_draft_calls: 0,
                    }));
                }
                Err(e) => {
                    // the session state is now unknown to the history bookkeeping
                    history.clear();
                    let _ = out.send(Event::Failed(e));
                }
            }
            continue 'serve;
        }
        let dcache = &mut dcaches[slot];
        let outcome = run_job(
            &model, &tok, &cfg, vision.as_ref(), dflash.as_ref(), dcache, session, history,
            &mut store, can_speculate, job, &stats,
        );
        if trace_vram {
            report_vram("after request");
        }
        // Drop the busy flag *before* the client is told the request finished. A client
        // that polls /slots the instant it has its answer — which is exactly what the
        // box's own benchmark does — would otherwise see the slot still working.
        stats.processing.fetch_sub(1, Ordering::Relaxed);
        stats.served.fetch_add(1, Ordering::Relaxed);
        match outcome {
            Ok(Some(f)) => {
                let _ = out.send(Event::Done(f));
            }
            Ok(None) => {}
            Err(e) => {
                let _ = out.send(Event::Failed(e));
            }
        }
    }
}


/// One request being served from a batch slot.
struct BatchReq {
    out: SyncSender<Event>,
    sampler: Sampler,
    greedy: bool,
    prompt_ids: Vec<u32>,
    /// prompt tokens already prefilled into this slot's region
    fed: usize,
    seq_past: usize,
    last: u32,
    generated: Vec<u32>,
    /// raw decoded bytes of `generated`; `text` is its UTF-8-lossy prefix
    bytes: Vec<u8>,
    /// bytes[..valid] have been converted into `text`
    valid: usize,
    text: String,
    emitted: usize,
    stop: Vec<String>,
    /// longest stop string in bytes; bounds the re-scan window per round
    max_stop: usize,
    max_tokens: usize,
    ignore_eos: bool,
    t_start: std::time::Instant,
    prefill_s: f64,
    client_gone: bool,
}

/// Serve every queued request through the shared batch session until none remain.
///
/// The loop's shape is the scheduler: admit new arrivals into free slots, feed one
/// prefill chunk per pending request, then run one fixed-width decode round for
/// everyone who is ready. Fixed width — dead slots carry a dummy token — keeps it at
/// exactly one cached graph, and chunked prefill bounds how long any request can stall
/// the others (~a quarter second per chunk on the 27B).
#[allow(clippy::too_many_arguments)]
fn run_batch(
    model: &Qwen35,
    tok: &codpiece_tok::Tokenizer,
    cfg: &EngineConfig,
    bsession: &mut Session,
    seq_ctx: usize,
    n_slots: usize,
    first: Job,
    rx: &Receiver<Job>,
    pending: &mut std::collections::VecDeque<Job>,
    stats: &Stats,
) {
    const PREFILL_CHUNK: usize = 256;
    let n_vocab = model.hp.n_vocab as usize;
    let mut slots: Vec<Option<BatchReq>> = (0..n_slots).map(|_| None).collect();

    let admit = |slot: usize,
                 job: Job,
                 slots: &mut Vec<Option<BatchReq>>,
                 bsession: &mut Session,
                 stats: &Stats| {
        stats.queued.fetch_sub(1, Ordering::Relaxed);
        stats.processing.fetch_add(1, Ordering::Relaxed);
        let Job { mut req, out } = job;
        // The fit check already tokenized this prompt and stored the result;
        // re-tokenizing a 90K-token prompt here would repeat real work.
        let prompt_ids = req.prompt_ids.take().unwrap_or_else(|| tok.encode(&req.prompt, true));
        if prompt_ids.is_empty() || prompt_ids.len() + req.max_tokens > seq_ctx {
            let _ = out.send(Event::Failed(format!(
                "prompt of {} tokens plus {} to generate exceeds this batch slot's {} token region",
                prompt_ids.len(),
                req.max_tokens,
                seq_ctx
            )));
            stats.processing.fetch_sub(1, Ordering::Relaxed);
            stats.served.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if let Err(e) = model.zero_seq_slot(bsession, slot, cfg.threads) {
            let _ = out.send(Event::Failed(format!("slot reset: {e}")));
            stats.processing.fetch_sub(1, Ordering::Relaxed);
            stats.served.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let _ = out.send(Event::Prefilled { n_prompt: prompt_ids.len() });
        let mut sampler = Sampler::new(req.params.clone());
        sampler.accept_all(&prompt_ids);
        slots[slot] = Some(BatchReq {
            out,
            greedy: req.params.is_greedy(),
            sampler,
            prompt_ids,
            fed: 0,
            seq_past: 0,
            last: 0,
            generated: Vec::new(),
            bytes: Vec::new(),
            valid: 0,
            text: String::new(),
            emitted: 0,
            max_stop: req.stop.iter().map(|s| s.len()).max().unwrap_or(0),
            stop: req.stop,
            max_tokens: req.max_tokens,
            ignore_eos: req.ignore_eos,
            t_start: std::time::Instant::now(),
            prefill_s: 0.0,
            client_gone: false,
        });
    };

    // seed with the request that triggered batch mode
    admit(0, first, &mut slots, bsession, stats);

    // CODPIECE_BATCH_TRACE=1: decompose the round: prefill stalls, the
    // decode step (model prints its own fill/compute/readback split), and
    // per-round host work (detok, stops, channel sends).
    let trace = std::env::var("CODPIECE_BATCH_TRACE").is_ok_and(|v| v == "1");
    // Measured at 32 wide: readback 9.96ms -> 0.01ms, round 72.5ms -> 64.9ms,
    // aggregate 273.7 -> 285.7 tok/s. Set CODPIECE_BATCH_GREEDY=0 to go back
    // to reading full logits and taking the argmax on the host.
    let batch_greedy = std::env::var("CODPIECE_BATCH_GREEDY").map_or(true, |v| v != "0");
    let mut tr_rounds = 0u64;
    let (mut tr_prefill, mut tr_step, mut tr_post) = (0f64, 0f64, 0f64);
    let mut tr_tokens = 0u64;
    let mut tr_wall = std::time::Instant::now();

    loop {
        // fill free slots from the queue and any freshly arrived work
        while let Ok(j) = rx.try_recv() {
            pending.push_back(j);
        }
        for slot in 0..n_slots {
            if slots[slot].is_none() {
                // Two kinds of request only run on the single path: images
                // (the batch graph has no embd input and no vision M-RoPE),
                // and prompts too long for a slot's region — a batch slot is
                // a fraction of the full context, and failing a long prompt
                // just because something else happened to be running would
                // make large requests fail intermittently. Both stay queued
                // for the serve loop to pick up once this batch drains.
                let mut found = None;
                for i in 0..pending.len() {
                    if !pending[i].req.images.is_empty() {
                        continue;
                    }
                    // Tokenize once per job, not once per scan: this loop runs
                    // for every free slot on every round, and a long prompt
                    // costs real milliseconds to encode.
                    let j = &mut pending[i];
                    let n = j
                        .req
                        .prompt_ids
                        .get_or_insert_with(|| tok.encode(&j.req.prompt, true))
                        .len();
                    if n > 0 && n + j.req.max_tokens <= seq_ctx {
                        found = Some(i);
                        break;
                    }
                }
                if let Some(at) = found {
                    let j = pending.remove(at).unwrap();
                    admit(slot, j, &mut slots, bsession, stats);
                }
            }
        }
        if slots.iter().all(Option::is_none) {
            return;
        }

        // one prefill chunk for each request still feeding its prompt
        let t_prefill = std::time::Instant::now();
        for slot in 0..n_slots {
            let Some(r) = slots[slot].as_mut() else { continue };
            if r.fed >= r.prompt_ids.len() {
                continue;
            }
            let hi = (r.fed + PREFILL_CHUNK).min(r.prompt_ids.len());
            let is_last = hi == r.prompt_ids.len();
            match model.step_seq_prefill(
                bsession,
                &r.prompt_ids[r.fed..hi],
                slot,
                seq_ctx,
                r.seq_past,
                is_last && !r.greedy,
                cfg.threads,
            ) {
                Ok((id, logits)) => {
                    r.seq_past += hi - r.fed;
                    r.fed = hi;
                    if is_last {
                        r.last = if r.greedy { id } else { r.sampler.sample(&logits) };
                        r.sampler.accept(r.last);
                        r.prefill_s = r.t_start.elapsed().as_secs_f64();
                    }
                }
                Err(e) => {
                    let _ = r.out.send(Event::Failed(format!("prefill: {e}")));
                    finish_slot(&mut slots[slot], stats, None);
                }
            }
        }

        tr_prefill += t_prefill.elapsed().as_secs_f64();

        // one decode round, fixed width, for everyone whose prompt is in
        let ready: Vec<usize> = (0..n_slots)
            .filter(|&i| slots[i].as_ref().is_some_and(|r| r.fed >= r.prompt_ids.len()))
            .collect();
        if ready.is_empty() {
            continue;
        }
        let mut lasts = vec![0u32; n_slots];
        let mut pasts = vec![0usize; n_slots];
        for &i in &ready {
            let r = slots[i].as_ref().unwrap();
            lasts[i] = r.last;
            pasts[i] = r.seq_past;
        }
        // CODPIECE_BATCH_GREEDY=1: all-greedy rounds read back one argmax id
        // per lane (128 bytes) instead of full-vocab logits (~32 MB at width
        // 32). Opt-in until the batched in-graph argmax is verified; the
        // graph rebuilds when this flips, so it only costs when the
        // greedy/sampled composition of the pool changes.
        let want_logits =
            !batch_greedy || ready.iter().any(|&i| !slots[i].as_ref().unwrap().greedy);
        let t_step = std::time::Instant::now();
        let (preds, logits) = match model.step_batch_decode(
            bsession,
            &lasts,
            &pasts,
            seq_ctx,
            want_logits,
            cfg.threads,
        ) {
            Ok(v) => v,
            Err(e) => {
                for slot in slots.iter_mut() {
                    if let Some(r) = slot.as_ref() {
                        let _ = r.out.send(Event::Failed(format!("decode: {e}")));
                    }
                    finish_slot(slot, stats, None);
                }
                return;
            }
        };
        tr_step += t_step.elapsed().as_secs_f64();
        let t_post = std::time::Instant::now();
        for &i in &ready {
            let r = slots[i].as_mut().unwrap();
            r.seq_past += 1;
            // commit the token the round consumed, then choose the next one
            r.generated.push(r.last);
            tok.token_bytes_into(r.last, false, &mut r.bytes);
            // convert newly complete UTF-8 into `text` (exactly what
            // from_utf8_lossy over the whole buffer would have produced)
            let scan_from = r.text.len().saturating_sub(r.max_stop.saturating_sub(1));
            while r.valid < r.bytes.len() {
                match std::str::from_utf8(&r.bytes[r.valid..]) {
                    Ok(s) => {
                        r.text.push_str(s);
                        r.valid = r.bytes.len();
                    }
                    Err(e) => {
                        let good = e.valid_up_to();
                        r.text.push_str(unsafe {
                            std::str::from_utf8_unchecked(&r.bytes[r.valid..r.valid + good])
                        });
                        r.valid += good;
                        match e.error_len() {
                            Some(bad) => {
                                r.text.push('\u{FFFD}');
                                r.valid += bad;
                            }
                            None => break, // incomplete tail: wait for more bytes
                        }
                    }
                }
            }
            let mut reason: Option<&'static str> = None;
            // a new stop occurrence must end in text added this round, so
            // only re-scan a window reaching max_stop-1 bytes back
            if r.max_stop > 0 {
                let mut w = scan_from;
                while !r.text.is_char_boundary(w) {
                    w -= 1;
                }
                if let Some(hit) =
                    r.stop.iter().find_map(|s| r.text[w..].find(s.as_str()).map(|p| w + p))
                {
                    r.text.truncate(hit);
                    reason = Some("stop");
                }
            }
            if r.text.len() > r.emitted && !r.client_gone {
                let chunk = r.text[r.emitted..].to_string();
                r.emitted = r.text.len();
                if r.out.send(Event::Token(chunk)).is_err() {
                    r.client_gone = true;
                }
            }
            if reason.is_none() && r.generated.len() >= r.max_tokens {
                reason = Some("length");
            }
            let next = if r.greedy {
                preds[i]
            } else {
                let d = r.sampler.distribution(&logits[i * n_vocab..(i + 1) * n_vocab]);
                r.sampler.draw_from(&d)
            };
            if reason.is_none() && !r.ignore_eos && Some(next) == tok.eos {
                reason = Some("stop");
            }
            r.sampler.accept(next);
            r.last = next;
            if r.client_gone && reason.is_none() {
                reason = Some("stop");
            }
            if let Some(why) = reason {
                let f = Finish {
                    reason: why,
                    n_prompt: r.prompt_ids.len(),
                    n_generated: r.generated.len(),
                    prefill_s: r.prefill_s,
                    decode_s: r.t_start.elapsed().as_secs_f64() - r.prefill_s,
                    acceptance: None,
                    draft_n: 0,
                    draft_n_accepted: 0,
                    n_draft_calls: 0,
                };
                finish_slot(&mut slots[i], stats, Some(f));
            }
        }
        tr_post += t_post.elapsed().as_secs_f64();
        tr_tokens += ready.len() as u64;
        tr_rounds += 1;
        if trace && tr_rounds % 64 == 0 {
            let wall = tr_wall.elapsed().as_secs_f64();
            eprintln!(
                "[round-trace] {} rounds: wall {:.2}ms  prefill {:.2}ms  step {:.2}ms  post {:.2}ms  | {:.1} tok/s",
                tr_rounds,
                wall * 1000.0 / 64.0,
                tr_prefill * 1000.0 / 64.0,
                tr_step * 1000.0 / 64.0,
                tr_post * 1000.0 / 64.0,
                tr_tokens as f64 / wall,
            );
            (tr_prefill, tr_step, tr_post, tr_tokens) = (0.0, 0.0, 0.0, 0);
            tr_wall = std::time::Instant::now();
        }
    }
}

/// Score a prompt: the logprob the model assigns to each of its own tokens.
///
/// Position j's logits predict token j+1, so a chunk's last row scores the
/// first token of the next chunk — `carry` is that row. Chunks stay small
/// because each scored position reads back a full vocabulary of logits
/// (~608 KiB), and a 2K prompt scored in one shot would move 1.2 GiB.
fn run_score(
    model: &Qwen35,
    tok: &codpiece_tok::Tokenizer,
    cfg: &EngineConfig,
    session: &mut Session,
    history: &mut Vec<u32>,
    ids: &[u32],
    top_k: usize,
) -> Result<Scored, String> {
    const SCORE_CHUNK: usize = 64;
    let n_vocab = model.hp.n_vocab as usize;
    let mut out = Scored::default();
    // Offsets come from each token's raw bytes, not from decoding it alone:
    // byte-level BPE splits multi-byte characters across tokens, so a
    // per-token decode can yield replacement characters and offsets that no
    // longer index the echoed text.
    let mut bytes = Vec::new();
    for &id in ids {
        let start = bytes.len();
        tok.token_bytes_into(id, true, &mut bytes);
        out.text_offset.push(start);
        out.tokens.push(String::from_utf8_lossy(&bytes[start..]).into_owned());
    }
    out.text = String::from_utf8_lossy(&bytes).into_owned();
    out.logprobs.push(None);
    out.top.push(None);

    // Scoring needs logits at every position, which the reuse path does not
    // keep, so this always starts from a clean state.
    session.reset();
    history.clear();

    let mut carry: Option<Vec<f32>> = None;
    let mut base = 0usize;
    for chunk in ids.chunks(SCORE_CHUNK) {
        let positions: Vec<i32> = (0..chunk.len() as i32).collect();
        let logits = model
            .step(session, chunk, &positions, cfg.threads)
            .map_err(|e| format!("score: {e}"))?;
        let mut push = |row: &[f32], target: u32| {
            let (lp, top) = score_row(row, target, top_k, tok);
            out.logprobs.push(Some(lp));
            out.top.push(top);
        };
        // the previous chunk's last row predicted this chunk's first token
        if let Some(prev) = carry.take() {
            push(&prev, ids[base]);
        }
        // row j predicts token base + j + 1; the last row reaches past this
        // chunk and becomes the next carry
        for j in 0..chunk.len().saturating_sub(1) {
            push(&logits[j * n_vocab..(j + 1) * n_vocab], ids[base + j + 1]);
        }
        if base + chunk.len() < ids.len() {
            carry = Some(logits[(chunk.len() - 1) * n_vocab..chunk.len() * n_vocab].to_vec());
        }
        base += chunk.len();
    }
    history.extend_from_slice(ids);
    Ok(out)
}

/// `CODPIECE_DECODE_TRACE=1`: where a single-stream round's time goes —
/// the verify step against the drafting that follows it — and how many tokens
/// the round actually committed. Printed every 64 rounds.
fn decode_trace(
    verify: f64,
    draft: std::time::Duration,
    inject: std::time::Duration,
    block: std::time::Duration,
    committed: usize,
    proposed: usize,
) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("CODPIECE_DECODE_TRACE").is_ok_and(|v| v == "1")) {
        return;
    }
    static VERIFY: AtomicU64 = AtomicU64::new(0);
    static DRAFT: AtomicU64 = AtomicU64::new(0);
    static INJECT: AtomicU64 = AtomicU64::new(0);
    static BLOCK: AtomicU64 = AtomicU64::new(0);
    static TOK: AtomicU64 = AtomicU64::new(0);
    static PROP: AtomicU64 = AtomicU64::new(0);
    static N: AtomicU64 = AtomicU64::new(0);
    VERIFY.fetch_add((verify * 1e6) as u64, Relaxed);
    DRAFT.fetch_add(draft.as_micros() as u64, Relaxed);
    INJECT.fetch_add(inject.as_micros() as u64, Relaxed);
    BLOCK.fetch_add(block.as_micros() as u64, Relaxed);
    TOK.fetch_add(committed as u64, Relaxed);
    PROP.fetch_add(proposed as u64, Relaxed);
    let n = N.fetch_add(1, Relaxed) + 1;
    if n % 64 == 0 {
        let (v, d) = (VERIFY.swap(0, Relaxed), DRAFT.swap(0, Relaxed));
        let (i, b) = (INJECT.swap(0, Relaxed), BLOCK.swap(0, Relaxed));
        let (t, p) = (TOK.swap(0, Relaxed), PROP.swap(0, Relaxed));
        eprintln!(
            "[decode-trace] 64 rounds: verify {:.2}ms  draft {:.2}ms (inject {:.2} + block \
             {:.2} + host {:.2})  {:.2} tok/round ({p} proposed, {t} committed) => {:.1} tok/s",
            v as f64 / 64_000.0,
            d as f64 / 64_000.0,
            i as f64 / 64_000.0,
            b as f64 / 64_000.0,
            d.saturating_sub(i + b) as f64 / 64_000.0,
            t as f64 / 64.0,
            t as f64 * 1e6 / (v + d) as f64,
        );
    }
}

/// Log partition function of one logit row, in f64 with the max subtracted —
/// the same shape the perplexity path uses, so scores agree between them.
fn log_z(row: &[f32]) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&x| ((x - max) as f64).exp()).sum();
    max as f64 + sum.ln()
}

/// Indices of the `k` largest logits, largest first.
fn top_indices(row: &[f32], k: usize) -> Vec<u32> {
    let k = k.min(row.len());
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    if k < row.len() {
        idx.select_nth_unstable_by(k - 1, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
        idx.truncate(k);
    }
    idx.sort_unstable_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    idx
}

/// The target's logprob, and the top alternatives when asked for.
fn score_row(
    row: &[f32],
    target: u32,
    top_k: usize,
    tok: &codpiece_tok::Tokenizer,
) -> (f32, Option<Vec<(String, f32)>>) {
    let ln_z = log_z(row);
    let lp = |x: f32| (x as f64 - ln_z) as f32;
    let top = (top_k > 0).then(|| {
        top_indices(row, top_k)
            .into_iter()
            .map(|i| (tok.decode(&[i], true), lp(row[i as usize])))
            .collect()
    });
    (lp(row[target as usize]), top)
}

/// Largest image, in vision tokens, this box can afford to encode.
///
/// The vision tower runs on its own backend with its own CUDA pool, and a
/// pool never returns memory to the driver — so memory the encoder peaks at
/// is taken away from the trunk *permanently*, not just for the duration of
/// the request. One 1024-token image on a card with 1.14 GiB free was enough
/// to drive it to zero and leave every subsequent request failing to
/// allocate. The cost grows with the square of the token count (attention is
/// over image tokens), so halving the cap quarters the peak.
///
/// llama.cpp's default ceiling is 4096 tokens; that assumes a card with room
/// to spare. `CODPIECE_IMAGE_MAX_TOKENS` overrides this.
fn vision_token_cap(reserved_bytes: usize) -> u32 {
    // Headroom the encoder may actually claim, after setting aside whatever
    // the batch session will take when concurrency first happens. Without
    // that reservation the cap depends on allocation ORDER: measured at ctx
    // 81920, the batch session was created first, took 0.76 GiB, and a
    // 1024-token image then failed and left the card at zero for good.
    let reserved = reserved_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let free_gib = (FREE_GIB.get().copied().unwrap_or(0.0) - reserved).max(0.0);
    // Calibrated against two measurements, because they disagree and the
    // larger one is the one that matters.
    //
    // The encoder's own compute buffer is linear in image tokens and small:
    // 30.0 / 62.0 / 120.1 / 237.4 MiB at 256 / 529 / 1024 / 2025 tokens, or
    // ~0.12 MiB per token. But end to end a 1024-token image consumed
    // ~0.41 GiB of card 1 (0.68 -> 0.27 GiB free, settling at 0.39) — about
    // 3x the encoder's buffer, because an image also introduces trunk graph
    // shapes at new t_len values, and those are cached too.
    //
    // So the budget below is ~0.4 MiB per token, the observed system cost,
    // with roughly one image's worth left over for the trunk.
    let cap = match free_gib {
        f if f > 2.1 => 4096,
        f if f > 1.25 => 2048,
        f if f > 0.85 => 1024,
        f if f > 0.6 => 512,
        _ => 256,
    };
    if cap < 1024 {
        eprintln!(
            "serve: {free_gib:.2} GiB available to the encoder ({reserved:.2} GiB reserved \
             for the batch session) — capping images at {cap} tokens. Qwen-VL grounding \
             degrades below 1024; lower the context, lower CODPIECE_BATCH, or unload the \
             drafter for full-detail vision."
        );
    }
    cap
}

/// Whether a request's prompt plus its generation fits one batch slot's
/// region. Tokenizing here duplicates work `admit` does again, but the
/// alternative is admitting a request that cannot be served.
fn fits_a_slot(tok: &codpiece_tok::Tokenizer, req: &mut GenRequest, seq_ctx: usize) -> bool {
    let n = req.prompt_ids.get_or_insert_with(|| tok.encode(&req.prompt, true)).len();
    n > 0 && n + req.max_tokens <= seq_ctx
}

/// Would a session of this shape fit on the tightest card right now?
///
/// Priced against *current* free VRAM, not startup headroom, because cached
/// graph shapes accumulate as requests arrive — the batch session is created
/// on first overlap, which can be long after warmup. A quarter of the
/// estimate is held back for the transient buffers the session's own graphs
/// will need on their first run.
fn fits_in_vram(model: &Qwen35, n_ctx_max: usize, k_slots: usize, what: &str) -> bool {
    let devs = codpiece_model::device_memory();
    if devs.is_empty() {
        return true; // CPU backend: not our call to make
    }
    let free = devs
        .iter()
        .map(|(_, used, total)| total.saturating_sub(*used))
        .min()
        .unwrap_or(0);
    let need = codpiece_model::qwen35::session_bytes(model, n_ctx_max, k_slots) / devs.len();
    let need_with_margin = need + need / 4;
    if need_with_margin > free {
        let mib = |b: usize| b / (1024 * 1024);
        eprintln!(
            "serve: {what} needs ~{} MiB per card (+25% margin) but only {} MiB is free; \
             declining it rather than risking the process",
            mib(need_with_margin),
            mib(free),
        );
        return false;
    }
    true
}

/// One line per GPU, so a failed context bump is diagnosable from the log
/// rather than from a second run with nvidia-smi alongside it.
fn report_vram(when: &str) {
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    for (name, used, total) in codpiece_model::device_memory() {
        eprintln!(
            "serve: vram {name} {:.2}/{:.2} GiB used, {:.2} GiB free ({when})",
            gib(used),
            gib(total),
            gib(total.saturating_sub(used)),
        );
    }
}

fn finish_slot(slot: &mut Option<BatchReq>, stats: &Stats, f: Option<Finish>) {
    if let Some(r) = slot.take() {
        stats.processing.fetch_sub(1, Ordering::Relaxed);
        stats.served.fetch_add(1, Ordering::Relaxed);
        if let Some(f) = f {
            let _ = r.out.send(Event::Done(f));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_job(
    model: &Qwen35,
    tok: &codpiece_tok::Tokenizer,
    cfg: &EngineConfig,
    vision: Option<&VisionCtx>,
    dflash: Option<&Dflash>,
    dcache: &mut Option<DflashCache>,
    session: &mut Session,
    history: &mut Vec<u32>,
    store: &mut SessionStore,
    can_speculate: bool,
    job: Job,
    stats: &Stats,
) -> Result<Option<Finish>, String> {
    let Job { req, out } = job;

    let raw_ids = match &req.prompt_ids {
        Some(ids) => ids.clone(),
        None => tok.encode(&req.prompt, true),
    };
    let (prompt_ids, segs) = match vision {
        Some(v) if !req.images.is_empty() => {
            expand_prompt(&raw_ids, v.pad_id, v.align, &req.images)
                .map_err(|e| format!("vision: {e}"))?
        }
        _ => {
            if !req.images.is_empty() {
                return Err("images supplied but no vision tower is loaded (--mmproj)".into());
            }
            let n = raw_ids.len();
            (raw_ids, vec![Seg::Text(0..n)])
        }
    };
    // Prompts must end with text: the token after the last prompt position is
    // predicted from a text step, and chat prompts always close with the
    // assistant header. (Images produce no logits.)
    if !matches!(segs.last(), Some(Seg::Text(_))) {
        return Err("prompt must contain text after the last image".into());
    }
    if prompt_ids.len() + req.max_tokens > cfg.n_ctx {
        return Err(format!(
            "prompt of {} tokens plus {} to generate exceeds the {} token context",
            prompt_ids.len(),
            req.max_tokens,
            cfg.n_ctx
        ));
    }
    // Reuse the session when this prompt extends what the caches already hold; the
    // suffix left to prefill must be non-empty so there is a position to predict from.
    let reused = !history.is_empty()
        && prompt_ids.len() > history.len()
        && prompt_ids[..history.len()] == history[..];
    if !reused {
        // The live session is about to be discarded. If it holds a real conversation,
        // suspend it to host RAM first — returning to it later costs a PCIe copy
        // instead of a full re-prefill.
        if history.len() >= 1024 {
            if let Some(snap) = session.snapshot() {
                store.save(history, snap);
            }
        }
        session.reset();
        history.clear();
        if let Some(hit) = store.best_prefix(&prompt_ids) {
            session.restore(&hit.snap);
            history.extend_from_slice(&hit.history);
        }
        // the draft ring belonged to whatever conversation this session held
        // before; n_seen = 0 makes its mask hide everything until this
        // conversation's features refill the window
        if let Some(c) = dcache.as_mut() {
            c.n_seen = 0;
        }
    }
    let cached_n = history.len();

    if out.send(Event::Prefilled { n_prompt: prompt_ids.len() }).is_err() {
        return Ok(None);
    }

    // Greedy rounds keep the in-graph argmax and skip reading a vocabulary back;
    // sampled rounds need the distribution to verify drafts against.
    let greedy = req.params.is_greedy();
    let mut sampler = Sampler::new(req.params.clone());
    // pseudo-ids of image spans stay out of the penalty history: llama.cpp's
    // sampler likewise sees only the text chunks
    let text_ids: Vec<u32> = prompt_ids.iter().copied().filter(|&t| t < 0x8000_0000).collect();
    sampler.accept_all(&text_ids);
    // DFlash2 drafter mode — GREEDY requests only. Measured split: the
    // block drafter wins structured greedy text by 30-37% (code 85 vs 65,
    // arithmetic 115 vs 84 tok/s) but loses temperature sampling to the
    // gumbel-coupled chain (45.6 vs 49.5): a sampled commit lands on a
    // draft with the target's own probability of it, and block positions
    // past ~3 compound to nothing on flat text while still paying vocab
    // noise and a wide verify. Sampled requests keep the chain path, which
    // also keeps the MTP cache warm exactly where it is used.
    let df_mode = dflash.is_some() && dcache.is_some() && greedy;
    let df_nmax: usize = std::env::var("CODPIECE_DFLASH_NMAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| dflash.map(|d| d.hp.block_size - 1).unwrap_or(0));
    let df_nmin: usize = std::env::var("CODPIECE_DFLASH_NMIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    // At temperature a draft lands with the target's own probability of it,
    // so deep block positions compound to nothing on flat text — and every
    // extra batch row also costs a vocab of gumbel noise. Same economics as
    // the MTP chain clamp.
    let df_cap = if greedy {
        df_nmax
    } else {
        df_nmax.min(
            std::env::var("CODPIECE_DFLASH_NMAX_SAMPLED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3),
        )
    };


    let mut text = String::new();
    let mut emitted = 0usize; // bytes of `text` already sent
    let mut generated: Vec<u32> = Vec::new();
    let t0 = std::time::Instant::now();

    let mut flush = |text: &String, emitted: &mut usize| -> bool {
        // Only forward complete UTF-8: a multi-byte character can span tokens, and
        // half of one is not something a client can render.
        if text.len() > *emitted {
            let chunk = &text[*emitted..];
            *emitted = text.len();
            return out.send(Event::Token(chunk.to_string())).is_ok();
        }
        true
    };

    let mut push = |ids: &[u32], text: &mut String| {
        // render_special = false: strip control tokens like <|im_end|> from the
        // user-visible text. The <think>/</think> tags are ordinary tokens and are
        // kept, so the reasoning split is unaffected.
        let piece = tok.decode(ids, false);
        text.clear();
        text.push_str(&piece);
    };

    // ---- prefill ----
    //
    // The speculative round's logits tensor is n_vocab x chunk, so it cannot use a big
    // chunk — 256 will not allocate at a long context. Prefill does not need it: only
    // the final stretch of the prompt has to go through the draft head, to leave its KV
    // cache warm and produce the first drafts. Everything before that is trunk-only, at
    // a chunk large enough to keep the GPU busy. Running the whole prompt through the
    // narrow speculative path cost 40% of prefill throughput (849 vs 1331 tok/s), which
    // was the entire gap against llama.cpp.
    const PREFILL_TAIL: usize = 64;
    let bulk_chunk = prefill_chunk_for(cfg.n_ctx);
    // The fused tail draws from the FINAL text segment only — images never
    // pass through the draft head, and the segment check above guarantees the
    // prompt ends with text.
    let last_text = match segs.last() {
        Some(Seg::Text(r)) => r.clone(),
        _ => unreachable!("checked above"),
    };
    let split = if can_speculate {
        last_text
            .end
            .saturating_sub(PREFILL_TAIL)
            .max(last_text.start)
            .max(cached_n)
    } else {
        prompt_ids.len()
    };
    let mut drafts: Vec<u32> = Vec::new();
    let mut next = 0u32;

    for seg in &segs {
        match seg {
            Seg::Image { idx, start } => {
                let v = vision.expect("image segs exist only with a vision ctx");
                let img = &req.images[*idx];
                if start + img.n_tokens(v.align) <= cached_n {
                    continue; // fully inside the reused prefix
                }
                let (nx, ny) = img.grid(v.align);
                let emb = v
                    .model
                    .encode(&img.planar, img.w as usize, img.h as usize)
                    .map_err(|e| format!("vision encode: {e}"))?;
                model
                    .step_embd(session, &emb, nx, ny, cfg.threads)
                    .map_err(|e| format!("vision inject: {e}"))?;
                if df_mode {
                    let (dm, c) = (dflash.unwrap(), dcache.as_mut().unwrap());
                    let base = session.n_past - nx * ny;
                    dm.inject(c, &session.last_taps, base, cfg.threads)
                        .map_err(|e| format!("dflash inject: {e}"))?;
                }
            }
            Seg::Text(r) => {
                let lo = r.start.max(cached_n);
                let hi = r.end.min(split);
                if lo >= hi {
                    continue;
                }
                for chunk in prompt_ids[lo..hi].chunks(bulk_chunk) {
                    let r = if greedy {
                        model.step_greedy(session, chunk, cfg.threads)
                    } else {
                        model
                            .step(session, chunk, &[(chunk.len() - 1) as i32], cfg.threads)
                            .map(|l| sampler.sample(&l))
                    };
                    match r {
                        // without a draft head this is the only source of the first
                        // token; with one it is overwritten by the speculative tail
                        Ok(t) => next = t,
                        Err(e) => {
                            return Err(format!("prefill: {e}"));
                        }
                    }
                    if df_mode {
                        let (dm, c) = (dflash.unwrap(), dcache.as_mut().unwrap());
                        let base = session.n_past - chunk.len();
                        dm.inject(c, &session.last_taps, base, cfg.threads)
                            .map_err(|e| format!("dflash inject: {e}"))?;
                    }
                }
            }
        }
    }
    if can_speculate {
        let tail = &prompt_ids[split..];
        debug_assert!(!tail.is_empty());
        let d0 = if df_mode {
            0
        } else if cfg.depth == usize::MAX {
            cfg.max_depth
        } else {
            cfg.depth
        };
        match model.step_fused_cached(session, tail, d0, None, false, !greedy, None, cfg.threads)
        {
            Ok(o) => {
                let (preds, chain, logits) = (o.preds, o.chain, o.logits);
                session.mtp_past += preds.len();
                let at = preds.len() - 1;
                next = if greedy {
                    preds[at]
                } else {
                    let v = model.hp.n_vocab as usize;
                    let d = sampler.distribution(&logits[at * v..(at + 1) * v]);
                    sampler.draw_from(&d)
                };
                drafts = chain.iter().map(|c| c[at]).collect();
            }
            Err(e) => {
                return Err(format!("prefill: {e}"));
            }
        }
        if df_mode {
            let (dm, c) = (dflash.unwrap(), dcache.as_mut().unwrap());
            dm.inject(c, &session.last_taps, session.n_past - tail.len(), cfg.threads)
                .map_err(|e| format!("dflash inject: {e}"))?;
            let lat = dm
                .draft_block(c, next, session.n_past, cfg.threads)
                .map_err(|e| format!("dflash draft: {e}"))?;
            // ALWAYS the greedy walk, even at temperature: coupled
            // verification accepts draft x with probability p_trunk(x), so
            // the optimal draft is the mode — a temperature walk only
            // decorrelates the draft from the commit (measured: acceptance
            // 0.119 sampled-walk vs ~0.5 greedy-walk on the same text).
            drafts = lat.walk_greedy(df_cap);
            if drafts.len() < df_nmin {
                drafts.clear();
            }
        }
    }
    history.clear();
    history.extend_from_slice(&prompt_ids);
    let prefill_s = t0.elapsed().as_secs_f64();
    sampler.accept(next);

    // ---- decode ----
    //
    // One loop serves both cases. At temperature 0 the target distribution is a point
    // mass on the argmax, so "accept with probability p(draft)" is exactly "accept when
    // the draft is what the trunk itself predicted" — the rule greedy speculation has
    // always used. Above it, accepting with that probability and otherwise drawing from
    // the residual emits the target distribution exactly, which is what lets a sampled
    // request speculate without changing what it would have produced.
    let t1 = std::time::Instant::now();
    let mut reason = "length";
    let (mut accepted, mut proposed, mut rounds) = (0usize, 0usize, 0usize);
    // Adaptive rather than fixed: a server is asked for prose and code in the same
    // hour, and the right draft depth differs by more than a factor of three between
    // them. `draft_gate` still applies on top, as an immediate brake when the
    // distribution at this position is flat.
    let adaptive = cfg.depth == usize::MAX;
    // Sampled requests cap the chain at 2. The chain's first link is conditioned on the
    // trunk's argmax, and a sampled commit diverges from the argmax roughly half the
    // time on flat text, dropping that round's drafts — so deep chains are built to be
    // thrown away. Measured at 32K, temperature 1.0: depth 1 = 44.0 tok/s (acceptance
    // 0.78, exactly production's figure), depth 2 = 47.1, depth 3 = 44.3.
    // Gumbel-coupled sampling: the graph argmaxes (logits + T*g) — an exact
    // temperature sample by the gumbel-max property — so the draft chain
    // conditions on the token the host commits instead of the argmax it may
    // diverge from. That divergence dropped ~half of all sampled drafts and
    // was the structural cause of the 32K sampled deficit. `plain` means no
    // filter or penalty is active, in which case the graph's sample IS the
    // commit; otherwise the host re-argmaxes over the filtered support with
    // the same noise (Sampler::sample_coupled), still exact.
    let use_gumbel = !greedy
        && req.params.temp > 0.0
        && std::env::var("CODPIECE_GUMBEL").as_deref() != Ok("0");
    let coupling_plain = req.params.coupling_is_exact();
    let mut noise_t: Vec<f32> = Vec::new();
    // ~1M gumbels per round cost 6-12 ms generated inline — a sixth of the
    // round. A dedicated thread with its own seed-derived stream keeps one
    // buffer ahead while the GPU computes; same seed still means the same
    // noise. Buffers cover the widest batch a round can present (chain
    // depth + 1, or the forced think-close).
    let noise_rx = if use_gumbel {
        let t = req.params.temp;
        let vocab = model.hp.n_vocab as usize;
        // widest batch a sampled round can present: the chain clamp, the
        // post-divergence re-draft (which may exceed the chain depth — the
        // verify batch is allowed to be wider than the chain), or a forced
        // think-close, plus the committed token
        let redraft_rows: usize = std::env::var("CODPIECE_REDRAFT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let rows = 1 + cfg
            .depth
            .clamp(1, 2)
            .max(redraft_rows)
            .max(req.think_close.len())
            .max(df_cap);
        let seed = req.params.seed ^ 0x9E37_79B9_7F4A_7C15;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(1);
        std::thread::Builder::new()
            .name("codpiece-gumbel".into())
            .spawn(move || {
                let mut rng = codpiece_sample::Rng::new(seed);
                loop {
                    let mut v = Vec::with_capacity(rows * vocab);
                    for _ in 0..rows * vocab {
                        let u = rng.next_f32().max(1e-12);
                        v.push(-(-u.ln()).ln() * t);
                    }
                    if tx.send(v).is_err() {
                        return;
                    }
                }
            })
            .map_err(|e| format!("noise thread: {e}"))?;
        Some(rx)
    } else {
        None
    };
    // Sampled chains stay clamped to 2 EVEN under gumbel coupling: coupling
    // fixes conditioning, not draft quality — a coupled draft is accepted
    // with the target's own probability of it, which at temperature 1.0 on
    // flat text is ~0.4, and depth 3 at 0.4/link measured 39.7 tok/s against
    // 47.3 for the shallower policy. Compounding wins again.
    let fixed = if greedy { cfg.depth.max(1) } else { cfg.depth.clamp(1, 2) };
    let mut picker = crate::depth::DepthPicker::new(
        adaptive,
        if adaptive { 1 } else { fixed },
        cfg.max_depth,
    );
    let mut round_depth = picker.choose();
    // Confidence-driven depth extension: MEASURED WORSE, off by default.
    // The model said an extra chain link (~6 ms) pays whenever its odds clear
    // ~0.35, and the in-graph confidences supply the odds. The measurement
    // said otherwise on all three prompt kinds (code 62.4 vs 64.7, prose 44.9
    // vs 49.0, arithmetic 80.4 vs 81.1 tok/s): link k only lands if every
    // earlier link landed, so the joint probability compounds and depth 3
    // stays the single peak regardless of per-link confidence — the fifth
    // independent confirmation that deeper/filtered drafting does not beat a
    // correctly chosen fixed depth on this model. CODPIECE_DEPTH_EXT=N>depth
    // re-enables it for experiments.
    const EXT_THRESH: f32 = 0.4;
    let ext_max = if greedy && !adaptive {
        std::env::var("CODPIECE_DEPTH_EXT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(fixed)
            .clamp(fixed, cfg.max_depth)
    } else {
        fixed
    };
    let n_vocab = model.hp.n_vocab as usize;
    // Thinking budget: if the model is still inside a <think> block after this many
    // generated tokens, its close sequence is force-fed so it resumes in answer mode
    // instead of thinking until max_tokens and returning nothing. `thinking` is true
    // only while no </think> has been emitted.
    let budget_on = req.think_budget > 0 && !req.think_close.is_empty();
    let n_embd = model.hp.n_embd as usize;
    // chain links re-drafted after a sampled divergence; 0 restores drop-only
    // Defaults from the 32K temperature-1.0 sweep: depth 3 with the confidence gate
    // measured 49.2 tok/s against 46.9 without re-drafting (and 45.4-47.9 for the
    // other depths); the gate is production's --spec-draft-p-min carried over.
    let redraft_depth: usize = std::env::var("CODPIECE_REDRAFT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    // Confidence gates, tuned for VALUE, not for the acceptance ratio.
    // Acceptance is a conversion statistic — the output distribution is provably
    // unchanged by speculation at any ratio — and the two draft pools have very
    // different economics. A link carried from the fused chain was computed last
    // round and is nearly free to verify, so dropping one is refusing a free bet:
    // gating the chain at 0.9 raised the ratio to 0.93 but cost 17% of 32K decode
    // (47.5 -> 39.5 tok/s). It stays ungated. A re-draft link costs a real ~6 ms
    // MTP pass, so declining low-confidence ones genuinely saves time; 0.75 is the
    // measured optimum (49.2 tok/s vs 46.9 without re-drafting). Deployments that
    // want the ratio to READ >= 0.90 can set both to 0.9 — that configuration
    // measured 0.910 greedy / 0.933 at 32K — but it buys no accuracy, only the
    // number.
    let chain_pmin: f32 = std::env::var("CODPIECE_CHAIN_PMIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let redraft_pmin: f32 = std::env::var("CODPIECE_REDRAFT_PMIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.75);
    let mut forced: Vec<u32> = Vec::new();
    'outer: loop {
        // Decide whether to inject the forced close this round.
        if forced.is_empty()
            && budget_on
            && generated.len() >= req.think_budget
            && !text.contains("</think>")
        {
            forced = req.think_close.clone();
        }
        if !req.ignore_eos && Some(next) == tok.eos {
            reason = "stop";
            break;
        }
        generated.push(next);
        push(&generated, &mut text);
        if let Some(hit) = req.stop.iter().find_map(|s| text.find(s.as_str())) {
            text.truncate(hit);
            reason = "stop";
            let _ = flush(&text, &mut emitted);
            break;
        }
        if !flush(&text, &mut emitted) {
            return Ok(None); // client hung up
        }
        if generated.len() >= req.max_tokens {
            reason = "length";
            break;
        }

        if !can_speculate {
            // no draft head: one token per step, sampled or greedy. The budget is
            // enforced here too, by force-feeding the close tokens one at a time.
            if !forced.is_empty() {
                let f = forced.remove(0);
                if let Err(e) = model.step_greedy(session, &[next], cfg.threads) {
                    return Err(format!("decode: {e}"));
                }
                history.push(next);
                sampler.accept(f);
                next = f;
                continue;
            }
            let r = if greedy {
                model.step_greedy(session, &[next], cfg.threads)
            } else {
                model
                    .step(session, &[next], &[0], cfg.threads)
                    .map(|l| sampler.sample(&l))
            };
            match r {
                Ok(t) => {
                    // the token just processed is now part of the cached state
                    history.push(next);
                    sampler.accept(t);
                    next = t;
                }
                Err(e) => {
                    return Err(format!("decode: {e}"));
                }
            }
            continue;
        }

        // While closing thinking, the forced tokens ARE the drafts, and they are
        // committed verbatim regardless of what the model would have predicted.
        let forcing = !forced.is_empty();
        let round_drafts: Vec<u32> = if forcing { forced.clone() } else { drafts.clone() };
        let mut batch = vec![next];
        batch.extend_from_slice(&round_drafts);
        if let Some(rx) = &noise_rx {
            noise_t = rx.recv().map_err(|_| "noise thread died".to_string())?;
            noise_t.truncate(batch.len() * n_vocab);
        }
        let t_round = std::time::Instant::now();
        let round_chain = if df_mode { 0 } else { round_depth };
        let out_round = match model.step_fused_cached(
            session,
            &batch,
            round_chain,
            None,
            false,
            !greedy,
            if use_gumbel { Some(&noise_t) } else { None },
            cfg.threads,
        ) {
            Ok(v) => v,
            Err(e) => {
                return Err(format!("decode: {e}"));
            }
        };
        let (preds, chain, conf, logits, hidden) = (
            out_round.preds,
            out_round.chain,
            out_round.conf,
            out_round.logits,
            out_round.hidden,
        );
        proposed += round_drafts.len();
        rounds += 1;
        let round_secs = t_round.elapsed().as_secs_f64();

        let mut n_keep = 0usize;
        let mut replacement: Option<u32> = None;
        for (j, draft) in round_drafts.iter().enumerate() {
            let keep = if forcing {
                true // committed verbatim to close the think block
            } else if greedy {
                preds[j] == *draft
            } else if use_gumbel {
                // The commit at j is the gumbel-max sample: the graph's own
                // argmax when nothing filters, else the same noise re-argmaxed
                // over the filtered support. Accepting the draft exactly when
                // it equals that sample commits an exact sample either way —
                // same acceptance probability as rejection sampling, and the
                // chain conditioned on precisely this token.
                let commit = if coupling_plain {
                    preds[j]
                } else {
                    sampler.sample_coupled(
                        &logits[j * n_vocab..(j + 1) * n_vocab],
                        &noise_t[j * n_vocab..(j + 1) * n_vocab],
                    )
                };
                if *draft == commit {
                    true
                } else {
                    replacement = Some(commit);
                    false
                }
            } else {
                let dist = sampler.distribution(&logits[j * n_vocab..(j + 1) * n_vocab]);
                let roll = sampler.rng().next_f32();
                if roll < dist.prob_of(*draft) {
                    true
                } else {
                    let r = sampler.rng().next_f32();
                    replacement = Some(dist.draw_excluding(*draft, r));
                    false
                }
            };
            if !keep {
                break;
            }
            n_keep += 1;
            sampler.accept(*draft);
        }
        accepted += n_keep;
        if !forcing && !df_mode {
            // the depth picker prices MTP chain depths; DFlash rounds are
            // verify-only and can be wider than its bookkeeping
            picker.observe(round_drafts.len(), round_depth, n_keep, round_secs);
        }

        // commit the accepted drafts, then the token that follows them
        for d in round_drafts.iter().take(n_keep) {
            if !req.ignore_eos && Some(*d) == tok.eos {
                reason = "stop";
                break 'outer;
            }
            generated.push(*d);
            push(&generated, &mut text);
            if let Some(hit) = req.stop.iter().find_map(|s| text.find(s.as_str())) {
                text.truncate(hit);
                reason = "stop";
                let _ = flush(&text, &mut emitted);
                break 'outer;
            }
            if !flush(&text, &mut emitted) {
                return Ok(None);
            }
            if generated.len() >= req.max_tokens {
                reason = "length";
                break 'outer;
            }
        }
        next = match replacement {
            _ if forcing => {
                // the block is closed; the token after it is the model's own first
                // answer token, drawn or argmaxed as the request asks
                forced.clear();
                round_depth = picker.choose();
                if greedy {
                    preds[n_keep]
                } else if use_gumbel {
                    if coupling_plain {
                        preds[n_keep]
                    } else {
                        sampler.sample_coupled(
                            &logits[n_keep * n_vocab..(n_keep + 1) * n_vocab],
                            &noise_t[n_keep * n_vocab..(n_keep + 1) * n_vocab],
                        )
                    }
                } else {
                    let dist =
                        sampler.distribution(&logits[n_keep * n_vocab..(n_keep + 1) * n_vocab]);
                    sampler.draw_from(&dist)
                }
            }
            Some(t) => t,
            None if greedy => {
                round_depth = picker.choose();
                if ext_max > fixed {
                    // one link beyond the links the head believes in
                    let confident = (0..chain.len())
                        .take_while(|&l| {
                            conf.get(l).map(|v| v[n_keep]).unwrap_or(0.0) >= EXT_THRESH
                        })
                        .count();
                    round_depth = (confident + 1).clamp(fixed, ext_max);
                }
                preds[n_keep]
            }
            None => {
                if use_gumbel {
                    round_depth = if cfg.draft_gate > 0.0 {
                        let dist = sampler
                            .distribution(&logits[n_keep * n_vocab..(n_keep + 1) * n_vocab]);
                        gated_depth(picker.choose(), cfg.draft_gate, dist.peak())
                    } else {
                        picker.choose()
                    };
                    if coupling_plain {
                        preds[n_keep]
                    } else {
                        sampler.sample_coupled(
                            &logits[n_keep * n_vocab..(n_keep + 1) * n_vocab],
                            &noise_t[n_keep * n_vocab..(n_keep + 1) * n_vocab],
                        )
                    }
                } else {
                    let dist =
                        sampler.distribution(&logits[n_keep * n_vocab..(n_keep + 1) * n_vocab]);
                    round_depth = gated_depth(picker.choose(), cfg.draft_gate, dist.peak());
                    sampler.draw_from(&dist)
                }
            }
        };
        sampler.accept(next);
        drafts = if forcing {
            Vec::new()
        } else {
            // Carry a link only while the draft head believed in it: production's
            // p-min, applied to the fused chain via the in-graph confidences. Beyond
            // the first unconfident link the tail is a guess about a guess — dropping
            // it is what holds acceptance high without giving up depth where the head
            // is sure.
            let mut d: Vec<u32> = Vec::new();
            for (link, c) in chain.iter().enumerate() {
                if conf.get(link).map(|v| v[n_keep]).unwrap_or(0.0) < chain_pmin {
                    break;
                }
                d.push(c[n_keep]);
            }
            d
        };
        // The chain was generated by the draft head continuing from the trunk's ARGMAX
        // at this position — that is what the in-graph argmax->get_rows feeds it. At
        // temperature the committed token is sampled and may differ, and then these
        // drafts are conditioned on a token that was never committed: not improbable,
        // wrong. Carrying them buys verification batches that reject almost everything,
        // which is exactly the acceptance collapse measured at temperature 1.0
        // (0.37 vs the 0.87 the same text yields greedily).
        if !greedy && next != preds[n_keep] {
            drafts.clear();
        }
        // a shallower next round must carry no more drafts than it will verify, or the
        // batch and the chain disagree and every switch builds a new graph shape
        drafts.truncate(round_depth);

        let over = batch.len() - (n_keep + 1);
        session.mtp_past += n_keep + 1;
        if over > 0 {
            if let Err(e) = model.rollback_recurrent(session, over, cfg.threads) {
                return Err(format!("rollback: {e}"));
            }
            session.n_past -= over;
        }
        // exactly the tokens that stayed in the caches this round
        history.extend_from_slice(&batch[..n_keep + 1]);

        let mut t_inject = std::time::Duration::ZERO;
        let mut t_block = std::time::Duration::ZERO;
        let t_draft = std::time::Instant::now();
        if df_mode && !forcing {
            let (dm, c) = (dflash.unwrap(), dcache.as_mut().unwrap());
            let committed = n_keep + 1;
            let n_feat = dm.hp.n_feat() as usize;
            let t_i = std::time::Instant::now();
            dm.inject(
                c,
                &session.last_taps[..committed * n_feat],
                session.n_past - committed,
                cfg.threads,
            )
            .map_err(|e| format!("dflash inject: {e}"))?;
            t_inject = t_i.elapsed();
            let t_b = std::time::Instant::now();
            let lat = dm
                .draft_block(c, next, session.n_past, cfg.threads)
                .map_err(|e| format!("dflash draft: {e}"))?;
            t_block = t_b.elapsed();
            // ALWAYS the greedy walk, even at temperature: coupled
            // verification accepts draft x with probability p_trunk(x), so
            // the optimal draft is the mode — a temperature walk only
            // decorrelates the draft from the commit (measured: acceptance
            // 0.119 sampled-walk vs ~0.5 greedy-walk on the same text).
            drafts = lat.walk_greedy(df_cap);
            if drafts.len() < df_nmin {
                drafts.clear();
            }
        }
        decode_trace(
            round_secs,
            t_draft.elapsed(),
            t_inject,
            t_block,
            n_keep + 1,
            round_drafts.len(),
        );

        // Post-commit re-draft: when the sampled token diverged from the argmax the
        // in-graph chain assumed, the chain's drafts were dropped — and without this,
        // the next round commits a single token for a full weight read. Re-drafting
        // from the token actually committed keeps speculation alive on sampled
        // requests; it is exactly what the standalone spec path always did, priced at
        // one draft-head pass (~6 ms) per chain link, and only on divergent rounds.
        if !greedy && drafts.is_empty() && !forcing && redraft_depth > 0 && !df_mode {
            let base = n_keep * n_embd;
            let mut h = hidden[base..base + n_embd].to_vec();
            let mut tok_in = next;
            let mut pos = session.rope_base();
            for _ in 0..redraft_depth {
                match model.mtp_draft(session, &h, tok_in, pos, cfg.threads) {
                    Ok((lg, hn)) => {
                        let d = codpiece_model::qwen35::argmax(&lg);
                        // Production's --spec-draft-p-min, applied where the draft
                        // head's confidence is actually available: stop the chain when
                        // its softmax peak drops below the threshold, instead of
                        // paying verification for a guess it does not believe in.
                        if redraft_pmin > 0.0 {
                            let m = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let lse: f32 = lg.iter().map(|v| (v - m).exp()).sum();
                            if 1.0 / lse < redraft_pmin {
                                break;
                            }
                        }
                        drafts.push(d);
                        tok_in = d;
                        h = hn;
                        pos += 1;
                    }
                    Err(_) => break, // draft quality only; verification stays exact
                }
            }
            drafts.truncate(round_depth);
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    stats
        .tokens_generated
        .fetch_add(generated.len() as u64, Ordering::Relaxed);
    let acceptance = (proposed > 0).then(|| accepted as f64 / proposed as f64);
    Ok(Some(Finish {
        reason,
        n_prompt: prompt_ids.len(),
        n_generated: generated.len(),
        prefill_s,
        decode_s,
        acceptance,
        draft_n: proposed,
        draft_n_accepted: accepted,
        n_draft_calls: rounds,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logprobs must be a proper log-distribution: exponentials sum to 1.
    #[test]
    fn log_softmax_normalizes() {
        let row = [0.5f32, -1.0, 3.25, 2.0, -7.5];
        let ln_z = log_z(&row);
        let total: f64 = row.iter().map(|&x| (x as f64 - ln_z).exp()).sum();
        assert!((total - 1.0).abs() < 1e-9, "sums to {total}");
        // and every score is negative, as a probability's log must be
        assert!(row.iter().all(|&x| (x as f64 - ln_z) < 0.0));
    }

    /// A large shared offset must not change the result — that is the point of
    /// subtracting the max, and without it exp() overflows to infinity.
    #[test]
    fn log_softmax_is_shift_invariant_and_overflow_safe() {
        let row = [1.0f32, 2.0, 3.0];
        let hot: Vec<f32> = row.iter().map(|x| x + 90.0).collect();
        let a = row[2] as f64 - log_z(&row);
        let b = hot[2] as f64 - log_z(&hot);
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        assert!(b.is_finite());
    }

    #[test]
    fn top_indices_are_ordered_by_logit() {
        let row = [0.1f32, 5.0, -2.0, 4.0, 4.5];
        assert_eq!(top_indices(&row, 3), vec![1, 4, 3]);
        // asking for more than exists returns everything, still ordered
        assert_eq!(top_indices(&row, 99), vec![1, 4, 3, 0, 2]);
    }
}
