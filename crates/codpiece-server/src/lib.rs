//! An OpenAI-compatible server in front of the codpiece engine.

pub mod api;
pub mod chat;
pub mod depth;
pub mod engine;
pub mod http;
pub mod tools;

use std::io::BufReader;
use std::net::TcpListener;
use std::sync::Arc;

pub struct ServeConfig {
    pub model_path: String,
    pub host: String,
    pub port: u16,
    pub n_ctx: usize,
    pub threads: i32,
    pub tp: Option<Vec<i32>>,
    pub gpu: Option<i32>,
    pub depth: usize,
    /// 0 selects adaptive depth.
    pub max_depth: usize,
    pub draft_gate: f32,
    /// Vision tower (mmproj GGUF path); None serves text only.
    pub mmproj: Option<String>,
    /// DFlash2 draft model (GGUF path); None keeps the MTP drafter.
    pub dflash: Option<String>,
    /// CUDA ordinal for the vision tower; None runs it on the CPU.
    pub mmproj_gpu: Option<i32>,
    pub default_max_tokens: usize,
    /// Served model id in `/v1/models` (the alias clients send as `model`).
    pub served_name: Option<String>,
    /// Tokens allowed inside `<think>` before the closer is forced; 0 disables.
    pub think_budget: usize,
}

pub fn run(cfg: ServeConfig) -> Result<(), String> {
    let (mut engine, template_src) = engine::Engine::start(engine::EngineConfig {
        model_path: cfg.model_path.clone(),
        n_ctx: cfg.n_ctx,
        threads: cfg.threads,
        tp: cfg.tp.clone(),
        gpu: cfg.gpu,
        depth: cfg.depth,
        max_depth: cfg.max_depth,
        draft_gate: cfg.draft_gate,
        mmproj: cfg.mmproj.clone(),
        mmproj_gpu: cfg.mmproj_gpu,
        dflash: cfg.dflash.clone(),
    })?;

    // A template that fails to parse is worth saying out loud rather than silently
    // falling back: chat requests would otherwise 400 with no explanation of why.
    let template = if template_src.is_empty() {
        eprintln!("serve: model has no chat template; /v1/chat/completions is unavailable");
        None
    } else {
        match chat::ChatTemplate::new(&template_src) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("serve: {e}; /v1/chat/completions is unavailable");
                None
            }
        }
    };

    // Only the header and metadata are read here, not the tensor data.
    let gguf = codpiece_gguf::GgufFile::open(std::path::Path::new(&cfg.model_path))
        .map_err(|e| format!("reopen for tokenizer: {e}"))?;
    if let Some(name) = cfg.served_name.clone() {
        engine.model_name = name;
    }
    let tokenizer = Arc::new(
        codpiece_tok::Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?,
    );

    // `<think>\n` is opened by the generation prompt, so the model only emits the
    // closer; tokenize it once for the budget's forced continuation.
    let think_close = tokenizer.encode("\n</think>\n\n", false);
    let ctx = Arc::new(api::Ctx {
        engine,
        tokenizer,
        template,
        default_max_tokens: cfg.default_max_tokens,
        think_close,
        think_budget: cfg.think_budget,
    });

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    eprintln!("serve: listening on http://{addr} (model {})", ctx.engine.model_name);

    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("serve: accept: {e}");
                continue;
            }
        };
        let ctx = ctx.clone();
        // A thread per connection: generation is serialised on the engine thread, so
        // these threads spend their lives blocked on a channel or a socket write.
        std::thread::spawn(move || {
            let peer = stream.peer_addr().ok();
            if let Err(e) = serve_conn(&ctx, stream) {
                // a client hanging up mid-stream is normal, not worth logging loudly
                if e.kind() != std::io::ErrorKind::BrokenPipe {
                    eprintln!("serve: {peer:?}: {e}");
                }
            }
        });
    }
    Ok(())
}

fn serve_conn(ctx: &api::Ctx, stream: std::net::TcpStream) -> std::io::Result<()> {
    let mut w = stream.try_clone()?;
    let mut r = BufReader::new(stream);
    match http::Request::read(&mut r) {
        Ok(Some(req)) => api::handle(ctx, &req, &mut w),
        Ok(None) => Ok(()),
        Err(e) => http::write_error(&mut w, 400, &e),
    }
}
