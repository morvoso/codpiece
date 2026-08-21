//! OpenAI-compatible endpoints.
//!
//! `/v1/completions` takes a raw prompt; `/v1/chat/completions` renders messages
//! through the model's own chat template first. Both stream when asked.

use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::Ordering;

use serde::Deserialize;
use serde_json::json;

use crate::chat::{ChatMessage, ChatTemplate};
use crate::engine::{Engine, Event, GenRequest};
use crate::http::{begin_sse, sse_data, sse_done, write_error, write_json, Request};
use codpiece_sample::SamplerParams;
use codpiece_vision::preprocess::{PreparedImage, Preprocessor};

/// Standard base64 (RFC 4648, `+/`, optional padding). Hand-rolled like the
/// HTTP layer: one alphabet, no streaming, fail loudly.
fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u8;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' => continue,
            _ => return Err(format!("invalid base64 byte {c:#x}")),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

/// Pull the encoded bytes out of one image content part. Accepted shapes:
/// `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`,
/// `image_url` as a plain string, and the Qwen `image` shorthand. Only data:
/// URLs (or bare base64) are accepted — the server does not fetch.
fn image_part_bytes(item: &serde_json::Value) -> Option<Result<Vec<u8>, String>> {
    let url = item
        .get("image_url")
        .map(|u| u.get("url").and_then(|v| v.as_str()).or(u.as_str()))
        .unwrap_or_else(|| item.get("image").and_then(|v| v.as_str()));
    let is_image = url.is_some() || item.get("type").and_then(|t| t.as_str()) == Some("image");
    if !is_image {
        return None;
    }
    let Some(url) = url else {
        return Some(Err("image part without image data".into()));
    };
    let payload = if let Some(rest) = url.strip_prefix("data:") {
        match rest.split_once(";base64,") {
            Some((_mime, b64)) => b64,
            None => return Some(Err("data: URL without base64 payload".into())),
        }
    } else if url.starts_with("http://") || url.starts_with("https://") {
        return Some(Err(
            "remote image URLs are not fetched; inline the image as a data: URL".into(),
        ));
    } else {
        url // bare base64
    };
    Some(base64_decode(payload))
}

/// Decode + preprocess every image in the conversation, in document order —
/// the same order the chat template emits `<|image_pad|>` markers.
fn extract_images(
    messages: &[ChatMessage],
    prep: &Preprocessor,
) -> Result<Vec<PreparedImage>, String> {
    let mut out = Vec::new();
    for m in messages {
        if let Some(parts) = m.content.as_array() {
            for item in parts {
                if let Some(bytes) = image_part_bytes(item) {
                    out.push(prep.prepare(&bytes?)?);
                }
            }
        }
    }
    Ok(out)
}

/// The knobs both endpoints share. Defaults match "no sampling", so a request that
/// specifies nothing decodes greedily and keeps the in-graph argmax.
#[derive(Debug, Deserialize)]
pub struct SamplingFields {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI's newer name for the same thing
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl SamplingFields {
    fn params(&self) -> SamplerParams {
        let mut p = SamplerParams::default();
        if let Some(t) = self.temperature {
            p.temp = t;
        }
        if let Some(v) = self.top_p {
            p.top_p = v;
        }
        if let Some(v) = self.top_k {
            p.top_k = v;
        }
        if let Some(v) = self.min_p {
            p.min_p = v;
        }
        if let Some(v) = self.repeat_penalty {
            p.penalty_repeat = v;
        }
        if let Some(v) = self.presence_penalty {
            p.penalty_present = v;
        }
        if let Some(v) = self.frequency_penalty {
            p.penalty_freq = v;
        }
        if let Some(v) = self.seed {
            p.seed = v;
        }
        p
    }

    fn max_tokens(&self, default: usize) -> usize {
        self.max_tokens
            .or(self.max_completion_tokens)
            .unwrap_or(default)
    }

    fn stop(&self) -> Vec<String> {
        match &self.stop {
            Some(StopField::One(s)) => vec![s.clone()],
            Some(StopField::Many(v)) => v.clone(),
            None => vec![],
        }
    }

    /// True when the client expressed no opinion about sampling at all. Only
    /// then may the model's own recommendation be substituted — a request that
    /// asked for temperature 0 means it.
    fn unspecified(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.min_p.is_none()
    }

    fn params_for_thinking(&self, model: Option<ModelSampling>) -> SamplerParams {
        let mut p = self.params();
        if let Some(m) = model.filter(|_| self.unspecified()) {
            p.temp = m.temp;
            p.top_k = m.top_k;
            p.top_p = m.top_p;
        }
        p
    }
}

#[derive(Debug, Deserialize)]
struct CompletionsBody {
    prompt: PromptField,
    /// Return the prompt's own tokens and their logprobs. Loglikelihood
    /// benchmarks run on this with `max_tokens: 0`.
    #[serde(default)]
    echo: Option<bool>,
    /// How many alternatives to report per position (OpenAI caps this at 5).
    #[serde(default)]
    logprobs: Option<usize>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

/// `/v1/completions` accepts a string or an array of token ids; harnesses use
/// the token form to keep their own tokenization authoritative.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PromptField {
    Text(String),
    Tokens(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct ChatBody {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    /// Qwen's template opens a `<think>` block when this is on, which changes where
    /// generation starts. Defaults on, matching the template's own default.
    #[serde(default)]
    enable_thinking: Option<bool>,
    /// llama.cpp's spelling: template variables nested under one object. Existing
    /// tooling passes thinking this way, so both are accepted.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Value>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

/// The sampling the model itself recommends, from `general.sampling.*`.
#[derive(Debug, Clone, Copy)]
pub struct ModelSampling {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl ModelSampling {
    pub fn from_gguf(g: &codpiece_gguf::GgufFile) -> Option<Self> {
        let f = |k: &str| g.kv(k).and_then(|v| v.as_f64());
        // temperature is the one that matters; the filters are optional
        let temp = f("general.sampling.temp")? as f32;
        Some(Self {
            temp,
            top_k: f("general.sampling.top_k").unwrap_or(0.0) as usize,
            top_p: f("general.sampling.top_p").unwrap_or(1.0) as f32,
        })
    }
}

pub struct Ctx {
    pub engine: Engine,
    /// Applied only to thinking requests that specify no sampling of their own.
    pub model_sampling: Option<ModelSampling>,
    /// The tokenization of the think-block close, computed once. `<think>` opens the
    /// generation prompt, so the model only has to emit the closer.
    pub think_close: Vec<u32>,
    /// Tokens allowed inside `<think>` before the closer is forced; 0 disables.
    pub think_budget: usize,
    /// A second copy of the tokenizer, so `/tokenize` does not have to queue behind
    /// whatever the engine thread is generating.
    pub tokenizer: std::sync::Arc<codpiece_tok::Tokenizer>,
    pub template: Option<ChatTemplate>,
    pub default_max_tokens: usize,
}

pub fn handle(ctx: &Ctx, req: &Request, w: &mut TcpStream) -> std::io::Result<()> {
    match (req.method.as_str(), req.route()) {
        ("OPTIONS", _) => write!(
            w,
            "HTTP/1.1 204 No Content\r\n\
             Access-Control-Allow-Origin: *\r\n\
             Access-Control-Allow-Headers: *\r\n\
             Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
             Content-Length: 0\r\n\r\n"
        ),
        ("GET", "/health") => write_json(w, 200, &json!({"status": "ok"}).to_string()),
        ("GET", "/v1/models") => {
            let body = json!({
                "object": "list",
                "data": [{
                    "id": ctx.engine.model_name,
                    "object": "model",
                    "owned_by": "codpiece",
                    "meta": { "n_ctx_train": ctx.engine.n_ctx },
                    // clients that size their own window look for this name
                    "context_length": ctx.engine.n_ctx,
                }]
            });
            write_json(w, 200, &body.to_string())
        }
        // llama.cpp's shape. Coding clients probe this to discover the
        // context window rather than being told it; without it they assume a
        // default and truncate prompts the server would have accepted.
        ("GET", "/props") => {
            let body = json!({
                "default_generation_settings": {
                    "n_ctx": ctx.engine.n_ctx,
                    "model": ctx.engine.model_name,
                },
                "total_slots": ctx.engine.stats.session_slots.load(Ordering::Relaxed).max(1),
                "model_path": ctx.engine.model_name,
                "chat_template": ctx.template.as_ref().map(|_| "").unwrap_or(""),
                "build_info": concat!("codpiece ", env!("CARGO_PKG_VERSION")),
            });
            write_json(w, 200, &body.to_string())
        }
        ("GET", "/slots") => {
            let s = &ctx.engine.stats;
            // One slot today: the engine serves a single sequence at a time. The shape
            // matches llama.cpp's so existing tooling can read it.
            let body = json!([{
                "id": 0,
                "n_ctx": ctx.engine.n_ctx,
                "is_processing": s.processing.load(Ordering::Relaxed) > 0,
                "queued": s.queued.load(Ordering::Relaxed),
                "served": s.served.load(Ordering::Relaxed),
                "tokens_generated": s.tokens_generated.load(Ordering::Relaxed),
            }]);
            write_json(w, 200, &body.to_string())
        }
        ("POST", "/tokenize") => {
            #[derive(Deserialize)]
            struct Body {
                content: String,
                #[serde(default)]
                add_special: Option<bool>,
            }
            let b: Body = match serde_json::from_slice(&req.body) {
                Ok(b) => b,
                Err(e) => return write_error(w, 400, &format!("invalid request body: {e}")),
            };
            let ids = ctx.tokenizer.encode(&b.content, b.add_special.unwrap_or(true));
            write_json(w, 200, &json!({ "tokens": ids }).to_string())
        }
        ("POST", "/detokenize") => {
            #[derive(Deserialize)]
            struct Body {
                tokens: Vec<u32>,
            }
            let b: Body = match serde_json::from_slice(&req.body) {
                Ok(b) => b,
                Err(e) => return write_error(w, 400, &format!("invalid request body: {e}")),
            };
            write_json(w, 200, &json!({ "content": ctx.tokenizer.decode(&b.tokens, true) }).to_string())
        }
        ("POST", "/v1/completions") | ("POST", "/completions") => {
            let body: CompletionsBody = match serde_json::from_slice(&req.body) {
                Ok(b) => b,
                Err(e) => return write_error(w, 400, &format!("invalid request body: {e}")),
            };
            let (prompt, prompt_ids) = match body.prompt {
                PromptField::Text(s) => (s, None),
                PromptField::Tokens(ids) => {
                    // These are used unvalidated all the way down: scoring
                    // indexes a logits row by token id, and the model gathers
                    // embedding rows by it. An id past the vocabulary is a
                    // crash, not a bad answer, so it is refused here.
                    let n_vocab = ctx.tokenizer.n_vocab();
                    if let Some(bad) = ids.iter().find(|&&t| t as usize >= n_vocab) {
                        return write_error(
                            w,
                            400,
                            &format!("token id {bad} is outside the {n_vocab}-token vocabulary"),
                        );
                    }
                    (ctx.tokenizer.decode(&ids, true), Some(ids))
                }
            };
            // Scoring needs both flags: `echo` alone just repeats the prompt.
            let echo = body.echo.unwrap_or(false);
            let echo_logprobs = match (echo, body.logprobs) {
                (true, Some(n)) => Some(n.min(20)),
                _ => None,
            };
            let gen = GenRequest {
                prompt,
                prompt_ids,
                images: Vec::new(),
                params: body.sampling.params(),
                max_tokens: body.sampling.max_tokens(ctx.default_max_tokens),
                stop: body.sampling.stop(),
                ignore_eos: body.sampling.ignore_eos.unwrap_or(false),
                think_budget: 0,
                think_close: Vec::new(),
                echo_logprobs,
            };
            serve_with(
                ctx,
                gen,
                body.sampling.stream.unwrap_or(false),
                "text_completion",
                false,
                None,
                w,
            )
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            let body: ChatBody = match serde_json::from_slice(&req.body) {
                Ok(b) => b,
                Err(e) => return write_error(w, 400, &format!("invalid request body: {e}")),
            };
            let Some(tmpl) = ctx.template.as_ref() else {
                return write_error(
                    w,
                    400,
                    "this model has no chat template; use /v1/completions with a raw prompt",
                );
            };
            let thinking = body
                .enable_thinking
                .or_else(|| {
                    body.chat_template_kwargs
                        .as_ref()
                        .and_then(|v| v.get("enable_thinking"))
                        .and_then(|v| v.as_bool())
                })
                .unwrap_or(true);
            let images = match ctx.engine.vision_prep.as_ref() {
                Some(prep) => match extract_images(&body.messages, prep) {
                    Ok(v) => v,
                    Err(e) => return write_error(w, 400, &format!("image: {e}")),
                },
                None => {
                    // fail loudly if the request carries images we cannot see
                    let has = body.messages.iter().any(|m| {
                        m.content
                            .as_array()
                            .map(|a| a.iter().any(|i| image_part_bytes(i).is_some()))
                            .unwrap_or(false)
                    });
                    if has {
                        return write_error(
                            w,
                            400,
                            "this server was started without --mmproj; images are not supported",
                        );
                    }
                    Vec::new()
                }
            };
            let prompt = match tmpl.render(
                &body.messages,
                true,
                body.tools.as_ref(),
                thinking,
                body.chat_template_kwargs.as_ref(),
            ) {
                Ok(p) => p,
                Err(e) => return write_error(w, 400, &e),
            };
            let gen = GenRequest {
                prompt,
                prompt_ids: None,
                images,
                params: if thinking {
                    body.sampling.params_for_thinking(ctx.model_sampling)
                } else {
                    body.sampling.params()
                },
                max_tokens: body.sampling.max_tokens(ctx.default_max_tokens),
                stop: body.sampling.stop(),
                ignore_eos: body.sampling.ignore_eos.unwrap_or(false),
                think_budget: if thinking { ctx.think_budget } else { 0 },
                think_close: ctx.think_close.clone(),
                echo_logprobs: None,
            };
            serve_with(
                ctx,
                gen,
                body.sampling.stream.unwrap_or(false),
                "chat.completion",
                thinking,
                body.tools.as_ref(),
                w,
            )
        }
        ("GET", _) | ("POST", _) => write_error(w, 404, "no such endpoint"),
        _ => write_error(w, 405, "method not allowed"),
    }
}

/// Split a generation that began inside an opened `<think>` block into the reasoning
/// and the answer, mirroring how the template will re-render the turn:
/// `<think>\n{reasoning|trim}\n</think>\n\n{content|trim}`. Trimming here matches the
/// template's own `|trim`, which is what keeps the round trip token-exact.
fn split_reasoning(text: &str, thinking: bool) -> (Option<String>, String) {
    match text.split_once("</think>") {
        Some((think, rest)) => (
            Some(think.trim().to_string()),
            rest.trim().to_string(),
        ),
        // No closing marker: under thinking the generation began inside the block, so
        // an unfinished generation is all reasoning — same verdict the streaming
        // splitter reaches.
        None if thinking => (Some(text.trim().to_string()), String::new()),
        None => (None, text.to_string()),
    }
}


/// Streams a generation that begins inside an opened `<think>` block, routing text
/// before `</think>` to `reasoning_content` deltas and text after it to `content`
/// deltas — the streaming mirror of `split_reasoning`.
///
/// The marker can arrive split across chunks ("...</th" then "ink>..."), so up to
/// `len("</think>") - 1` trailing bytes are held back until the next chunk decides
/// whether they are the marker or ordinary text.
struct ReasoningStream {
    in_reasoning: bool,
    held: String,
}

impl ReasoningStream {
    fn new(thinking: bool) -> Self {
        Self { in_reasoning: thinking, held: String::new() }
    }

    /// Returns (reasoning_delta, content_delta) for this chunk.
    fn push(&mut self, chunk: &str) -> (String, String) {
        if !self.in_reasoning {
            return (String::new(), chunk.to_string());
        }
        const MARK: &str = "</think>";
        self.held.push_str(chunk);
        if let Some(at) = self.held.find(MARK) {
            let reasoning = self.held[..at].to_string();
            let mut content = self.held[at + MARK.len()..].to_string();
            // the template renders "\n</think>\n\n" around the marker; the answer
            // starts after that blank line, matching the non-streaming trim
            let reasoning = reasoning.trim_end().to_string();
            while content.starts_with('\n') {
                content.remove(0);
            }
            self.held.clear();
            self.in_reasoning = false;
            return (reasoning, content);
        }
        // hold back any suffix that could be the start of the marker
        let keep = (1..MARK.len())
            .rev()
            .find(|k| self.held.ends_with(&MARK[..*k]))
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let ready = self.held[..cut].to_string();
        self.held = self.held[cut..].to_string();
        (ready, String::new())
    }

    /// Anything still held when the stream ends was ordinary text after all.
    fn flush(&mut self) -> (String, String) {
        let held = std::mem::take(&mut self.held);
        if self.in_reasoning {
            (held, String::new())
        } else {
            (String::new(), held)
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn serve_with(
    ctx: &Ctx,
    gen: GenRequest,
    stream: bool,
    kind: &str,
    thinking: bool,
    tools: Option<&serde_json::Value>,
    w: &mut TcpStream,
) -> std::io::Result<()> {
    let id = format!("cmpl-{}", now_secs());
    let created = now_secs();
    let model = ctx.engine.model_name.clone();
    let chat = kind.starts_with("chat");
    let rx = ctx.engine.submit(gen);

    if stream {
        begin_sse(w)?;
        let mut first = true;
        let mut n_prompt = 0usize;
        // chat generations under thinking begin inside an opened <think> block; the
        // stream labels that span reasoning_content, matching the non-streaming shape
        let mut rsplit = ReasoningStream::new(chat && thinking);
        // With tools declared, content is streamed until a tool call opens and
        // then withheld: the call is only well formed once complete, and it
        // travels in its own delta rather than as prose.
        let mut tsplit = tools.map(|_| crate::tools::ToolStream::new());
        while let Ok(ev) = rx.recv() {
            match ev {
                Event::Prefilled { n_prompt: n } => n_prompt = n,
                // scoring is a non-streaming shape; a client that asks for
                // both gets the echoed text and no logprobs
                Event::Scored(s) => {
                    let chunk = json!({
                        "id": id, "object": "text_completion", "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": s.text, "finish_reason": null}],
                    });
                    sse_data(w, &chunk.to_string())?;
                }
                Event::Token(text) => {
                    let (reasoning, mut content) = rsplit.push(&text);
                    if let Some(t) = tsplit.as_mut() {
                        content = t.push(&content);
                    }
                    if reasoning.is_empty() && content.is_empty() {
                        continue; // held back pending the marker decision
                    }
                    let delta = if chat {
                        let mut d = serde_json::Map::new();
                        if first {
                            d.insert("role".into(), json!("assistant"));
                        }
                        if !reasoning.is_empty() {
                            d.insert("reasoning_content".into(), json!(reasoning));
                        }
                        if !content.is_empty() {
                            d.insert("content".into(), json!(content));
                        }
                        serde_json::Value::Object(d)
                    } else {
                        json!(null)
                    };
                    let text = content;
                    let chunk = if chat {
                        json!({
                            "id": id, "object": "chat.completion.chunk", "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
                        })
                    } else {
                        json!({
                            "id": id, "object": "text_completion", "created": created,
                            "model": model,
                            "choices": [{"index": 0, "text": text, "finish_reason": null}],
                        })
                    };
                    first = false;
                    sse_data(w, &chunk.to_string())?;
                }
                Event::Done(f) => {
                    let (held_r, mut held_c) = rsplit.flush();
                    let mut calls = Vec::new();
                    if let Some(t) = tsplit.as_mut() {
                        let emitted = t.push(&held_c);
                        let (tail, c) = t.flush(tools);
                        held_c = emitted + &tail;
                        calls = c;
                    }
                    if !held_r.is_empty() || !held_c.is_empty() {
                        let mut d = serde_json::Map::new();
                        if !held_r.is_empty() {
                            d.insert("reasoning_content".into(), json!(held_r));
                        }
                        if !held_c.is_empty() {
                            d.insert("content".into(), json!(held_c));
                        }
                        let chunk = json!({
                            "id": id, "object": "chat.completion.chunk", "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": d, "finish_reason": null}],
                        });
                        if chat {
                            sse_data(w, &chunk.to_string())?;
                        }
                    }
                    let usage = json!({
                        "prompt_tokens": f.n_prompt,
                        "completion_tokens": f.n_generated,
                        "total_tokens": f.n_prompt + f.n_generated,
                    });
                    let mut reason = f.reason;
                    if !calls.is_empty() {
                        // one delta carrying the complete calls, then the close
                        let chunk = json!({
                            "id": id, "object": "chat.completion.chunk", "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": crate::tools::to_openai_delta(&calls, &id)},
                                "finish_reason": null,
                            }],
                        });
                        sse_data(w, &chunk.to_string())?;
                        reason = "tool_calls";
                    }
                    let last = if chat {
                        json!({
                            "id": id, "object": "chat.completion.chunk", "created": created,
                            "model": model, "usage": usage,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                            "timings": timings(&f),
                        })
                    } else {
                        json!({
                            "id": id, "object": "text_completion", "created": created,
                            "model": model, "usage": usage,
                            "choices": [{"index": 0, "text": "", "finish_reason": f.reason}],
                            "timings": timings(&f),
                        })
                    };
                    sse_data(w, &last.to_string())?;
                    return sse_done(w);
                }
                Event::Failed(msg) => {
                    // The status line is already sent, so the failure has to travel as
                    // an event; a truncated stream would otherwise look like success.
                    let err = json!({"error": {"message": msg, "type": "server_error"}});
                    sse_data(w, &err.to_string())?;
                    return sse_done(w);
                }
            }
        }
        let _ = n_prompt;
        return sse_done(w);
    }

    let mut text = String::new();
    let mut scored: Option<serde_json::Value> = None;
    while let Ok(ev) = rx.recv() {
        match ev {
            Event::Prefilled { .. } => {}
            Event::Token(t) => text.push_str(&t),
            Event::Scored(s) => {
                text = s.text.clone();
                scored = Some(json!({
                    "tokens": s.tokens,
                    "token_logprobs": s.logprobs,
                    "top_logprobs": s.top.iter().map(|t| match t {
                        Some(v) => serde_json::Value::Object(
                            v.iter().map(|(k, lp)| (k.clone(), json!(lp))).collect(),
                        ),
                        None => serde_json::Value::Null,
                    }).collect::<Vec<_>>(),
                    "text_offset": s.text_offset,
                }));
            }
            Event::Done(f) => {
                let usage = json!({
                    "prompt_tokens": f.n_prompt,
                    "completion_tokens": f.n_generated,
                    "total_tokens": f.n_prompt + f.n_generated,
                });
                let body = if chat {
                    let (reasoning, content) = split_reasoning(&text, thinking);
                    // Tool calls are only looked for when the request declared
                    // tools; otherwise the markup is just text the user asked for.
                    let (content, calls) = match tools {
                        Some(_) => crate::tools::parse(&content, tools),
                        None => (content, Vec::new()),
                    };
                    let mut message = json!({"role": "assistant", "content": content});
                    if let Some(r) = reasoning {
                        message["reasoning_content"] = json!(r);
                    }
                    let mut reason = f.reason;
                    if !calls.is_empty() {
                        // clients read `content` only when there are no calls; an
                        // empty string there is noise, so send null
                        if message["content"].as_str().is_some_and(str::is_empty) {
                            message["content"] = json!(null);
                        }
                        message["tool_calls"] = crate::tools::to_openai(&calls, &id);
                        reason = "tool_calls";
                    }
                    json!({
                        "id": id, "object": "chat.completion", "created": created,
                        "model": model, "usage": usage, "timings": timings(&f),
                        "choices": [{
                            "index": 0,
                            "message": message,
                            "finish_reason": reason,
                        }],
                    })
                } else {
                    let mut choice = json!({
                        "index": 0, "text": text, "finish_reason": f.reason,
                    });
                    choice["logprobs"] = scored.take().unwrap_or(serde_json::Value::Null);
                    json!({
                        "id": id, "object": "text_completion", "created": created,
                        "model": model, "usage": usage, "timings": timings(&f),
                        "choices": [choice],
                    })
                };
                return write_json(w, 200, &body.to_string());
            }
            Event::Failed(msg) => return write_error(w, 500, &msg),
        }
    }
    write_error(w, 500, "engine closed the stream without finishing")
}

fn timings(f: &crate::engine::Finish) -> serde_json::Value {
    json!({
        "prompt_n": f.n_prompt,
        "prompt_ms": f.prefill_s * 1000.0,
        "prompt_per_second": if f.prefill_s > 0.0 { f.n_prompt as f64 / f.prefill_s } else { 0.0 },
        "predicted_n": f.n_generated,
        "predicted_ms": f.decode_s * 1000.0,
        "predicted_per_second": if f.decode_s > 0.0 { f.n_generated as f64 / f.decode_s } else { 0.0 },
        "acceptance": f.acceptance,
        "draft_n": f.draft_n,
        "draft_n_accepted": f.draft_n_accepted,
        "n_draft_calls": f.n_draft_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str], thinking: bool) -> (String, String) {
        let mut st = ReasoningStream::new(thinking);
        let (mut r, mut c) = (String::new(), String::new());
        for ch in chunks {
            let (dr, dc) = st.push(ch);
            r.push_str(&dr);
            c.push_str(&dc);
        }
        let (dr, dc) = st.flush();
        r.push_str(&dr);
        c.push_str(&dc);
        (r, c)
    }

    #[test]
    fn splits_reasoning_from_content() {
        let (r, c) = run(&["I think.", "\n</think>\n\nFour."], true);
        assert_eq!(r, "I think.");
        assert_eq!(c, "Four.");
    }

    #[test]
    fn marker_split_across_chunks() {
        // the marker arrives in three pieces; nothing may leak across the boundary
        let (r, c) = run(&["reasoning</th", "in", "k>\n\nanswer"], true);
        assert_eq!(r, "reasoning");
        assert_eq!(c, "answer");
    }

    #[test]
    fn a_false_start_is_ordinary_text() {
        // "</thin..." that never completes the marker is reasoning text
        let (r, c) = run(&["a </thing> b"], true);
        assert_eq!(r, "a </thing> b");
        assert_eq!(c, "");
    }

    #[test]
    fn no_marker_means_everything_is_reasoning() {
        // generation cut off mid-think: matches the non-streaming split, which finds
        // no marker and returns the raw text
        let (r, c) = run(&["endless deliberation"], true);
        assert_eq!(r, "endless deliberation");
        assert_eq!(c, "");
    }

    #[test]
    fn thinking_off_passes_through() {
        let (r, c) = run(&["plain ", "text</think>still plain"], false);
        assert_eq!(r, "");
        assert_eq!(c, "plain text</think>still plain");
    }
}

#[cfg(test)]
mod sampling_tests {
    use super::*;

    fn fields(json: &str) -> SamplingFields {
        serde_json::from_str(json).expect("parse")
    }

    const MODEL: ModelSampling = ModelSampling { temp: 1.0, top_k: 20, top_p: 0.95 };

    #[test]
    fn thinking_with_no_sampling_adopts_the_models_recommendation() {
        let p = fields(r#"{}"#).params_for_thinking(Some(MODEL));
        assert_eq!(p.temp, 1.0);
        assert_eq!(p.top_k, 20);
        assert_eq!(p.top_p, 0.95);
    }

    /// A request that asked for greedy means greedy — benchmarks depend on it.
    #[test]
    fn an_explicit_temperature_is_never_overridden() {
        let p = fields(r#"{"temperature": 0}"#).params_for_thinking(Some(MODEL));
        assert_eq!(p.temp, 0.0);
        assert_eq!(p.top_k, 0);
        // any one sampling field is enough to mean "I chose these"
        let p = fields(r#"{"top_p": 0.5}"#).params_for_thinking(Some(MODEL));
        assert_eq!(p.temp, 0.0);
        assert_eq!(p.top_p, 0.5);
    }

    #[test]
    fn a_model_without_recommendations_stays_greedy() {
        let p = fields(r#"{}"#).params_for_thinking(None);
        assert_eq!(p.temp, 0.0);
        assert_eq!(p.top_k, 0);
    }
}
