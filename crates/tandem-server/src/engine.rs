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

use tandem_model::qwen35::{Qwen35, Session};
use tandem_sample::{Sampler, SamplerParams};


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
pub fn prefill_chunk_for(n_ctx: usize) -> usize {
    const WEIGHTS_GIB: f64 = 16.46; // per card, including the mirrored head and embeddings
    const CARD_GIB: f64 = 23.5;
    let kv_gib = n_ctx as f64 * 65536.0 / 2.0 / (1u64 << 30) as f64;
    let free = CARD_GIB - WEIGHTS_GIB - kv_gib;
    if free > 2.0 {
        512
    } else {
        64
    }
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
    snap: tandem_model::qwen35::SessionSnapshot,
    last_used: u64,
}

impl SessionStore {
    fn new() -> Self {
        let gib = std::env::var("TANDEM_SESSION_CACHE_GIB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(12);
        Self { entries: Vec::new(), budget: gib << 30, clock: 0 }
    }

    fn save(&mut self, history: &[u32], snap: tandem_model::qwen35::SessionSnapshot) {
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
    pub params: SamplerParams,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
}

#[derive(Debug)]
pub enum Event {
    /// prompt token count, once, before any text
    Prefilled { n_prompt: usize },
    Token(String),
    Done(Finish),
    Failed(String),
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
}

#[derive(Default)]
pub struct Stats {
    pub queued: AtomicUsize,
    pub processing: AtomicUsize,
    pub served: AtomicU64,
    pub tokens_generated: AtomicU64,
}

pub struct EngineConfig {
    pub model_path: String,
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
    /// tandem already has: how peaked the *target* distribution was at the position it
    /// just sampled. A flat distribution means the text is unpredictable here, and
    /// under the rejection rule an unlikely draft is rejected in proportion — so the
    /// chain steps behind it are spent for nothing. 0 disables.
    pub draft_gate: f32,
}

impl Engine {
    /// Load the model on a worker thread and return once it is ready to serve.
    pub fn start(cfg: EngineConfig) -> Result<(Self, String), String> {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<String, String>>();
        let stats = Arc::new(Stats::default());
        let stats_worker = stats.clone();
        let name = std::path::Path::new(&cfg.model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());
        let n_ctx = cfg.n_ctx;
        std::thread::Builder::new()
            .name("tandem-engine".into())
            .spawn(move || worker(cfg, rx, ready_tx, stats_worker))
            .map_err(|e| format!("engine thread: {e}"))?;
        let template = ready_rx
            .recv()
            .map_err(|_| "engine thread died during load".to_string())??;
        Ok((Self { tx, stats, model_name: name, n_ctx }, template))
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
    ready: Sender<Result<String, String>>,
    stats: Arc<Stats>,
) {
    let device = match (&cfg.tp, cfg.gpu) {
        (Some(ids), _) => tandem_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => tandem_model::Device::Cuda(i),
        (None, None) => tandem_model::Device::Cpu,
    };
    let model = match Qwen35::load_on(std::path::Path::new(&cfg.model_path), device) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(format!("load: {e}")));
            return;
        }
    };
    let tok = match tandem_tok::Tokenizer::from_gguf(&model.weights.gguf) {
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
    let can_speculate = model.has_mtp() && cfg.depth != 0;
    if !can_speculate {
        eprintln!("serve: model has no MTP head; decoding without speculation");
    }
    let slots = if cfg.depth == usize::MAX { cfg.max_depth } else { cfg.depth }
        .max(cfg.max_depth)
        .max(1);
    let session = match Session::new_spec(&model, cfg.n_ctx, slots) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!("session: {e}")));
            return;
        }
    };
    if ready.send(Ok(template)).is_err() {
        return;
    }

    // A pool of sessions, each holding one conversation's state in VRAM, plus the
    // history of tokens that state covers. Switching conversations is a pointer choice,
    // not a copy — which matters doubly under tensor parallelism, where the split GDN
    // state cannot be copied off-device at all. Pool size is bounded by VRAM: each
    // extra session costs its KV cache (~2 GiB per card at 64K context), so the count
    // shrinks as the context grows.
    let n_sessions = std::env::var("TANDEM_SESSIONS")
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
    let mut store = SessionStore::new();
    let mut clock = 0u64;
    // The batch session is created the first time two requests overlap: it costs
    // ~2.6 GiB of VRAM, and a single-user box that never overlaps never pays it.
    let batch_slots = std::env::var("TANDEM_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let batch_seq_ctx = cfg.n_ctx.min(8192);
    let mut bsession: Option<Session> = None;
    let mut pending: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
    'serve: while let Ok(job) = pending.pop_front().map(Ok).unwrap_or_else(|| rx.recv()) {
        // A request that arrives while another is queued means real concurrency:
        // serve the group through the batch path so neither waits for the other's
        // whole generation.
        while let Ok(next) = rx.try_recv() {
            pending.push_back(next);
        }
        if !pending.is_empty() && batch_slots > 1 {
            if bsession.is_none() {
                match Session::new_spec(&model, batch_slots * batch_seq_ctx, batch_slots - 1) {
                    Ok(s) => bsession = Some(s),
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
        let prompt_ids = tok.encode(&job.req.prompt, true);
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
                if std::env::var("TANDEM_TRACE_REUSE").as_deref() == Ok("1") {
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
        let outcome = run_job(
            &model, &tok, &cfg, session, history, &mut store, can_speculate, job, &stats,
        );
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
    text: String,
    emitted: usize,
    stop: Vec<String>,
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
    tok: &tandem_tok::Tokenizer,
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
        let Job { req, out } = job;
        let prompt_ids = tok.encode(&req.prompt, true);
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
            text: String::new(),
            emitted: 0,
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

    loop {
        // fill free slots from the queue and any freshly arrived work
        while let Ok(j) = rx.try_recv() {
            pending.push_back(j);
        }
        for slot in 0..n_slots {
            if slots[slot].is_none() {
                if let Some(j) = pending.pop_front() {
                    admit(slot, j, &mut slots, bsession, stats);
                }
            }
        }
        if slots.iter().all(Option::is_none) {
            return;
        }

        // one prefill chunk for each request still feeding its prompt
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
        let (preds, logits) = match model.step_batch_decode(
            bsession,
            &lasts,
            &pasts,
            seq_ctx,
            true,
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
        for &i in &ready {
            let r = slots[i].as_mut().unwrap();
            r.seq_past += 1;
            // commit the token the round consumed, then choose the next one
            r.generated.push(r.last);
            let piece = tok.decode(&r.generated, true);
            r.text.clear();
            r.text.push_str(&piece);
            let mut reason: Option<&'static str> = None;
            if let Some(hit) = r.stop.iter().find_map(|s| r.text.find(s.as_str())) {
                r.text.truncate(hit);
                reason = Some("stop");
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
    tok: &tandem_tok::Tokenizer,
    cfg: &EngineConfig,
    session: &mut Session,
    history: &mut Vec<u32>,
    store: &mut SessionStore,
    can_speculate: bool,
    job: Job,
    stats: &Stats,
) -> Result<Option<Finish>, String> {
    let Job { req, out } = job;

    let prompt_ids = tok.encode(&req.prompt, true);
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
    }
    let cached_n = history.len();

    if out.send(Event::Prefilled { n_prompt: prompt_ids.len() }).is_err() {
        return Ok(None);
    }

    // Greedy rounds keep the in-graph argmax and skip reading a vocabulary back;
    // sampled rounds need the distribution to verify drafts against.
    let greedy = req.params.is_greedy();
    let mut sampler = Sampler::new(req.params.clone());
    sampler.accept_all(&prompt_ids);

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
        let piece = tok.decode(ids, true);
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
    let split = if can_speculate {
        prompt_ids.len().saturating_sub(PREFILL_TAIL).max(cached_n)
    } else {
        prompt_ids.len()
    };
    let mut drafts: Vec<u32> = Vec::new();
    let mut next = 0u32;

    for chunk in prompt_ids[cached_n..split].chunks(bulk_chunk) {
        let r = if greedy {
            model.step_greedy(session, chunk, cfg.threads)
        } else {
            model
                .step(session, chunk, &[(chunk.len() - 1) as i32], cfg.threads)
                .map(|l| sampler.sample(&l))
        };
        match r {
            // without a draft head this is the only source of the first token; with one
            // it is overwritten by the speculative tail below
            Ok(t) => next = t,
            Err(e) => {
                return Err(format!("prefill: {e}"));
            }
        }
    }
    if can_speculate {
        let tail = &prompt_ids[split..];
        debug_assert!(!tail.is_empty());
        let d0 = if cfg.depth == usize::MAX { cfg.max_depth } else { cfg.depth };
        match model.step_fused_cached(session, tail, d0, None, false, !greedy, cfg.threads)
        {
            Ok((preds, chain, logits)) => {
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
    let fixed = if greedy { cfg.depth.max(1) } else { cfg.depth.clamp(1, 2) };
    let mut picker = crate::depth::DepthPicker::new(
        adaptive,
        if adaptive { 1 } else { fixed },
        cfg.max_depth,
    );
    let mut round_depth = picker.choose();
    let n_vocab = model.hp.n_vocab as usize;
    'outer: loop {
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
            // no draft head: one token per step, sampled or greedy
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

        let mut batch = vec![next];
        batch.extend_from_slice(&drafts);
        let t_round = std::time::Instant::now();
        let (preds, chain, logits) = match model.step_fused_cached(
            session,
            &batch,
            round_depth,
            None,
            false,
            !greedy,
            cfg.threads,
        ) {
            Ok(v) => v,
            Err(e) => {
                return Err(format!("decode: {e}"));
            }
        };
        proposed += drafts.len();
        rounds += 1;
        let round_secs = t_round.elapsed().as_secs_f64();

        let mut n_keep = 0usize;
        let mut replacement: Option<u32> = None;
        for (j, draft) in drafts.iter().enumerate() {
            let keep = if greedy {
                preds[j] == *draft
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
        picker.observe(drafts.len(), round_depth, n_keep, round_secs);

        // commit the accepted drafts, then the token that follows them
        for d in drafts.iter().take(n_keep) {
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
            if !req.ignore_eos && Some(*d) == tok.eos {
                reason = "stop";
                break 'outer;
            }
        }
        next = match replacement {
            Some(t) => t,
            None if greedy => {
                round_depth = picker.choose();
                preds[n_keep]
            }
            None => {
                let dist = sampler.distribution(&logits[n_keep * n_vocab..(n_keep + 1) * n_vocab]);
                round_depth = gated_depth(picker.choose(), cfg.draft_gate, dist.peak());
                sampler.draw_from(&dist)
            }
        };
        sampler.accept(next);
        drafts = chain.iter().map(|c| c[n_keep]).collect();
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
