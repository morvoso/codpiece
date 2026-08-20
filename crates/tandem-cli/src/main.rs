//! tandem CLI. First subcommand: `inspect` — header-only GGUF analysis.
//! Reads metadata + tensor directory only; never maps tensor data, never
//! touches a GPU. Safe to run against files a live server has loaded.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use tandem_gguf::{GgufFile, TensorInfo, Value};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => cmd_inspect(&args[1..]),
        Some("tokenize") => cmd_tokenize(&args[1..]),
        _ => {
            eprintln!(
                "usage: tandem inspect <file.gguf> [--tensors] [--kv <key>] [--dump-template <out>]\n\
                 usage: tandem tokenize <file.gguf> [--special] [--pieces] < text"
            );
            ExitCode::from(2)
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
