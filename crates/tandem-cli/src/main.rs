//! tandem CLI. First subcommand: `inspect` — header-only GGUF analysis.
//! Reads metadata + tensor directory only; never maps tensor data, never
//! touches a GPU. Safe to run against files a live server has loaded.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use tandem_gguf::{GgufFile, TensorInfo, Value};

fn main() -> ExitCode {
    // Piping into `head` must truncate output, not panic the process.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => cmd_inspect(&args[1..]),
        Some("tokenize") => cmd_tokenize(&args[1..]),
        Some("load") => cmd_load(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("ppl") => cmd_ppl(&args[1..]),
        _ => {
            eprintln!(
                "usage: tandem inspect <file.gguf> [--tensors] [--kv <key>] [--dump-template <out>]\n\
                 usage: tandem tokenize <file.gguf> [--special] [--pieces] < text\n\
                 usage: tandem load <file.gguf>"
            );
            ExitCode::from(2)
        }
    }
}

/// Perplexity over a text file, mirroring llama-perplexity's methodology
/// exactly (tools/perplexity/perplexity.cpp @ b10423): consecutive n_ctx
/// chunks evaluated fresh; positions [n_ctx/2, n_ctx-1) score the next token;
/// ppl = exp(nll/count). qwen35 has add_bos=false, so no BOS substitution.
fn cmd_ppl(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut file: Option<&str> = None;
    let mut n_ctx = 512usize;
    let mut n_chunks = -1i64;
    let mut threads = 8i32;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-f" => file = it.next().map(String::as_str),
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(512),
            "--chunks" => n_chunks = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            s => {
                eprintln!("unknown arg: {s}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(path), Some(file)) = (path, file) else {
        eprintln!("usage: tandem ppl <file.gguf> -f <text> [-c n_ctx] [--chunks N] [-t threads]");
        return ExitCode::from(2);
    };

    let model = match tandem_model::qwen35::Qwen35::load(Path::new(path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("tandem ppl: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match tandem_tok::Tokenizer::from_gguf(&model.weights.gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tandem ppl: {e}");
            return ExitCode::FAILURE;
        }
    };
    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tandem ppl: {file}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let t0 = std::time::Instant::now();
    let tokens = tok.encode(&text, true);
    eprintln!("tokenized {} chars -> {} tokens in {:.1}s", text.len(), tokens.len(), t0.elapsed().as_secs_f64());
    if tokens.len() < 2 * n_ctx {
        eprintln!("need at least {} tokens, got {}", 2 * n_ctx, tokens.len());
        return ExitCode::FAILURE;
    }

    let n_chunk_max = tokens.len() / n_ctx;
    let n_chunk = if n_chunks < 0 { n_chunk_max } else { (n_chunks as usize).min(n_chunk_max) };
    let first = n_ctx / 2;
    let n_vocab = model.hp.n_vocab as usize;
    // logits at j for j in [first, n_ctx-1) predict token j+1
    let out_positions: Vec<i32> = (first..n_ctx - 1).map(|j| j as i32).collect();

    let mut nll = 0f64;
    let mut count = 0usize;
    for i in 0..n_chunk {
        let chunk = &tokens[i * n_ctx..(i + 1) * n_ctx];
        let logits = match model.forward(chunk, &out_positions, threads) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("chunk {i}: {e}");
                return ExitCode::FAILURE;
            }
        };
        for (row, j) in (first..n_ctx - 1).enumerate() {
            let lrow = &logits[row * n_vocab..(row + 1) * n_vocab];
            let target = chunk[j + 1] as usize;
            // stable log-softmax, f64 accumulation (mirrors oracle numerics)
            let max = lrow.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = lrow.iter().map(|&x| ((x - max) as f64).exp()).sum();
            nll -= (lrow[target] - max) as f64 - sum.ln();
            count += 1;
        }
        print!("[{}]{:.4},", i + 1, (nll / count as f64).exp());
        use std::io::Write as _;
        std::io::stdout().flush().ok();
    }
    println!();
    println!("Final estimate: PPL = {:.4} over {} scored tokens", (nll / count as f64).exp(), count);
    ExitCode::SUCCESS
}

/// Greedy generation on the CPU backend (M1 correctness rig: stateless
/// forward, full-prefix recompute per token).
fn cmd_run(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut prompt = String::from("The capital of France is");
    let mut n_gen = 16usize;
    let mut threads = 8i32;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" => prompt = it.next().cloned().unwrap_or_default(),
            "-n" => n_gen = it.next().and_then(|s| s.parse().ok()).unwrap_or(16),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            s => {
                eprintln!("unknown arg: {s}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = path else {
        eprintln!("usage: tandem run <file.gguf> [-p prompt] [-n tokens] [-t threads]");
        return ExitCode::from(2);
    };

    let t0 = std::time::Instant::now();
    let model = match tandem_model::qwen35::Qwen35::load(Path::new(path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("tandem run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match tandem_tok::Tokenizer::from_gguf(&model.weights.gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tandem run: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "loaded {} ({} layers, {} embd) in {:.2}s",
        path,
        model.hp.n_layer,
        model.hp.n_embd,
        t0.elapsed().as_secs_f64()
    );

    let prompt_ids = tok.encode(&prompt, true);
    eprintln!("prompt ids: {prompt_ids:?}");

    let t1 = std::time::Instant::now();
    let mut ids = prompt_ids.clone();
    for i in 0..n_gen {
        let logits = match model.forward_logits(&ids, threads) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("forward failed at step {i}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let next = tandem_model::qwen35::argmax(&logits);
        ids.push(next);
        eprint!("{next} ");
        if Some(next) == tok.eos {
            eprintln!("[eos]");
            break;
        }
    }
    eprintln!();
    let gen_ids = &ids[prompt_ids.len()..];
    println!("{}", tok.decode(gen_ids, true));
    eprintln!(
        "{} tokens in {:.2}s ({:.2} tok/s, stateless O(n^2) rig)",
        n_gen,
        t1.elapsed().as_secs_f64(),
        n_gen as f64 / t1.elapsed().as_secs_f64()
    );
    ExitCode::SUCCESS
}

/// Load all weights onto the CPU backend and verify integrity. Exercises the
/// full loader path (tensor creation, size cross-check, streaming copy).
fn cmd_load(args: &[String]) -> ExitCode {
    let Some(path) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: tandem load <file.gguf>");
        return ExitCode::from(2);
    };
    let t0 = std::time::Instant::now();
    match tandem_model::Weights::load(Path::new(path), tandem_model::Device::Cpu) {
        Ok(w) => {
            let dt = t0.elapsed().as_secs_f64();
            let gib = w.bytes_loaded as f64 / (1u64 << 30) as f64;
            println!(
                "loaded {} tensors, {:.3} GiB in {:.2}s ({:.2} GiB/s) [arch {}]",
                w.n_tensors(),
                gib,
                dt,
                gib / dt,
                w.gguf.architecture().unwrap_or("?"),
            );
            // Deterministic fingerprint of the embedding table's first bytes:
            // reruns and machines must agree (catches copy/offset bugs).
            if let Some(bytes) = w.tensor_bytes("token_embd.weight") {
                let n = bytes.len().min(1 << 20);
                let mut h: u64 = 0xcbf29ce484222325;
                for &b in &bytes[..n] {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                println!("token_embd.weight[..{n}] fnv1a = {h:016x}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("tandem load: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Tokenize stdin with the GGUF-embedded tokenizer. One token id per line
/// (`--pieces` adds the decoded piece) — the format the M1 parity harness
/// diffs against `llama-tokenize`.
fn cmd_tokenize(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut special = false;
    let mut pieces = false;
    for a in args {
        match a.as_str() {
            "--special" => special = true,
            "--pieces" => pieces = true,
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            s => {
                eprintln!("unknown arg: {s}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = path else {
        eprintln!("usage: tandem tokenize <file.gguf> [--special] [--pieces] < text");
        return ExitCode::from(2);
    };

    let g = match GgufFile::open(Path::new(path)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("tandem tokenize: {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match tandem_tok::Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tandem tokenize: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut text = String::new();
    if std::io::Read::read_to_string(&mut std::io::stdin(), &mut text).is_err() {
        eprintln!("stdin was not valid utf-8");
        return ExitCode::FAILURE;
    }

    let ids = tok.encode(&text, special);
    let mut out = String::new();
    for id in &ids {
        if pieces {
            let piece = tok.decode(&[*id], true);
            out.push_str(&format!("{id}\t{piece:?}\n"));
        } else {
            out.push_str(&format!("{id}\n"));
        }
    }
    print!("{out}");
    eprintln!("{} tokens", ids.len());
    ExitCode::SUCCESS
}

fn cmd_inspect(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut list_tensors = false;
    let mut kv_query: Option<&str> = None;
    let mut dump_template: Option<&str> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tensors" => list_tensors = true,
            "--kv" => kv_query = it.next().map(String::as_str),
            "--dump-template" => dump_template = it.next().map(String::as_str),
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            s => {
                eprintln!("unknown arg: {s}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = path else {
        eprintln!("usage: tandem inspect <file.gguf> [...]");
        return ExitCode::from(2);
    };

    let g = match GgufFile::open(Path::new(path)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("tandem inspect: {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(key) = kv_query {
        match g.kv(key) {
            Some(Value::String(s)) => println!("{s}"),
            Some(v) => println!("{}", v.render(usize::MAX)),
            None => {
                eprintln!("key not found: {key}");
                return ExitCode::FAILURE;
            }
        }
        return ExitCode::SUCCESS;
    }

    if let Some(out) = dump_template {
        match g.kv("tokenizer.chat_template").and_then(Value::as_str) {
            Some(t) => {
                if let Err(e) = std::fs::write(out, t) {
                    eprintln!("write {out}: {e}");
                    return ExitCode::FAILURE;
                }
                println!("wrote {} bytes to {out}", t.len());
            }
            None => {
                eprintln!("no tokenizer.chat_template in file");
                return ExitCode::FAILURE;
            }
        }
        return ExitCode::SUCCESS;
    }

    print_summary(&g, path);
    if list_tensors {
        println!("\n== tensors ({}) ==", g.tensors.len());
        for t in &g.tensors {
            println!(
                "  {:<44} {:>10} {:?} @{}",
                t.name,
                t.ty.name(),
                t.dims,
                t.offset
            );
        }
    }
    ExitCode::SUCCESS
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn print_summary(g: &GgufFile, path: &str) {
    println!("== {path} ==");
    println!(
        "gguf v{} | {} tensors | {} kvs | align {} | data @ {} | file {:.3} GiB | tensor bytes {:.3} GiB",
        g.version,
        g.tensors.len(),
        g.kvs.len(),
        g.alignment,
        g.data_start,
        gib(g.file_len),
        gib(g.tensor_bytes()),
    );

    // All scalar/string KVs, plus elided arrays.
    println!("\n== metadata ==");
    for (k, v) in &g.kvs {
        println!("  {k} = {}", v.render(4));
    }

    // Type census.
    let mut by_type: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let e = by_type.entry(t.ty.name()).or_default();
        e.0 += 1;
        e.1 += t.byte_size().unwrap_or(0);
    }
    println!("\n== tensor types ==");
    for (name, (count, bytes)) in &by_type {
        println!("  {name:<10} x{count:<5} {:.3} GiB", gib(*bytes));
    }

    // Layer census: group blk.N.* by N, classify by suffix fingerprint.
    let mut layers: BTreeMap<u64, Vec<&TensorInfo>> = BTreeMap::new();
    let mut other: Vec<&str> = Vec::new();
    for t in &g.tensors {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some((n, _suffix)) = rest.split_once('.') {
                if let Ok(n) = n.parse::<u64>() {
                    layers.entry(n).or_default().push(t);
                    continue;
                }
            }
        }
        other.push(&t.name);
    }

    if !layers.is_empty() {
        println!("\n== layer census ({} layers) ==", layers.len());
        // Fingerprint each layer by its sorted suffix set; print one line per
        // distinct fingerprint with the layer numbers that share it.
        let mut by_fp: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for (n, ts) in &layers {
            let mut sufs: Vec<String> = ts
                .iter()
                .filter_map(|t| {
                    t.name
                        .strip_prefix(&format!("blk.{n}."))
                        .map(str::to_string)
                })
                .collect();
            sufs.sort();
            by_fp.entry(sufs.join(",")).or_default().push(*n);
        }
        for (fp, ns) in &by_fp {
            println!("  layers {}:", render_ranges(ns));
            for s in fp.split(',') {
                println!("    {s}");
            }
        }
    }

    if !other.is_empty() {
        println!("\n== non-layer tensors ({}) ==", other.len());
        for name in &other {
            println!("  {name}");
        }
    }
}

/// Render sorted layer ids as compact ranges: [0-2, 5, 7-9].
fn render_ranges(ns: &[u64]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < ns.len() {
        let start = ns[i];
        let mut end = start;
        while i + 1 < ns.len() && ns[i + 1] == end + 1 {
            i += 1;
            end = ns[i];
        }
        if !out.is_empty() {
            out.push_str(", ");
        }
        if start == end {
            out.push_str(&start.to_string());
        } else {
            out.push_str(&format!("{start}-{end}"));
        }
        i += 1;
    }
    format!("[{out}] ({} layers)", ns.len())
}
