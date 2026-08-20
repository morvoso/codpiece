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
        Some("gen") => cmd_gen(&args[1..]),
        Some("ppl") => cmd_ppl(&args[1..]),
        Some("selftest") => cmd_selftest(&args[1..]),
        Some("mtp-probe") => cmd_mtp_probe(&args[1..]),
        Some("spec") => cmd_spec(&args[1..]),
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

/// Speculative greedy generation with the MTP draft head.
///
/// Each round: draft `n_spec` tokens from the MTP head (cheap — one block,
/// not 64 layers), then verify all of them in ONE trunk pass. Verification
/// costs about what a single decode step costs, because decode is
/// memory-bandwidth-bound: the weights are read once regardless of how many
/// tokens ride along. Accepted drafts are therefore nearly free.
///
/// This is lossless: a draft is kept only when it equals what the trunk
/// would have produced anyway, so `spec` output must match `gen` exactly.
/// Rejected drafts leave the 48 recurrent layers over-advanced, which is
/// what `rollback_recurrent` undoes via the snapshot slots.
/// Probability of `idx` under a numerically stable softmax of `logits`.
fn softmax_prob(logits: &[f32], idx: u32) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    if sum > 0.0 {
        (logits[idx as usize] - max).exp() / sum
    } else {
        0.0
    }
}

fn cmd_spec(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut prompt = String::from("The capital of France is");
    let mut n_gen = 64usize;
    let mut threads = 8i32;
    let mut n_ctx = 4096usize;
    let mut n_spec = 1usize;
    let mut p_min = 0.75f32;
    let mut n_oracle = 0usize;
    let mut oracle_conf = 0.6f32;
    let mut gpu: Option<i32> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut ignore_eos = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" => prompt = it.next().cloned().unwrap_or_default(),
            "-n" => n_gen = it.next().and_then(|s| s.parse().ok()).unwrap_or(64),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(4096),
            "--spec" => n_spec = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "--p-min" => p_min = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.75),
            "--oracle" => n_oracle = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--oracle-conf" => {
                oracle_conf = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.6)
            }
            "--ignore-eos" => ignore_eos = true,
            "--gpu" => gpu = Some(it.next().and_then(|s| s.parse().ok()).unwrap_or(0)),
            "--tp" => {
                tp = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            _ => {}
        }
    }
    let Some(path) = path else {
        eprintln!("usage: tandem spec <file.gguf> [-p prompt] [-n N] [--spec K] [--gpu N|--tp 0,1]");
        return ExitCode::from(2);
    };
    let dev = match (&tp, gpu) {
        (Some(ids), _) => tandem_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => tandem_model::Device::Cuda(i),
        (None, None) => tandem_model::Device::Cpu,
    };
    let model = match tandem_model::qwen35::Qwen35::load_on(Path::new(path), dev) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("spec: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = tandem_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    // rollback slots must cover every draft a round can produce, from both
    // drafters
    let mut session =
        match tandem_model::qwen35::Session::new_spec(&model, n_ctx, n_spec + n_oracle) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("spec: {e}");
                return ExitCode::FAILURE;
            }
        };

    let prompt_ids = tok.encode(&prompt, true);
    // The oracle learns from the prompt before generation starts: quoting the
    // input back is the single most common free-token case.
    let mut oracle = tandem_model::oracle::ContextOracle::new(oracle_conf);
    oracle.extend(&prompt_ids);

    let t0 = std::time::Instant::now();
    let (mut logits, mut hidden) = match model.step_with_hidden(
        &mut session,
        &prompt_ids,
        &[(prompt_ids.len() - 1) as i32],
        threads,
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("prefill: {e}");
            return ExitCode::FAILURE;
        }
    };
    let t_prefill = t0.elapsed().as_secs_f64();

    let mut committed: Vec<u32> = vec![tandem_model::qwen35::argmax(&logits)];
    let (mut accepted, mut proposed, mut rounds) = (0usize, 0usize, 0usize);
    let mut oracle_accepted = 0usize;
    let t1 = std::time::Instant::now();

    'outer: while committed.len() < n_gen {
        let last = *committed.last().unwrap();
        if !ignore_eos && Some(last) == tok.eos {
            break;
        }
        // ---- draft ----
        let mut drafts: Vec<u32> = Vec::with_capacity(n_spec);
        let mut h = hidden.clone();
        let mut tok_in = last;
        let mut pos = session.n_past;
        for _ in 0..n_spec {
            match model.mtp_draft(&mut session, &h, tok_in, pos, threads) {
                Ok((dl, dh)) => {
                    // Confidence gate (llama.cpp's --spec-draft-p-min): a draft
                    // the head is unsure about is usually rejected, and a
                    // rejected draft costs a wasted verify slot plus a
                    // recurrent rollback. Stopping early is cheaper than
                    // drafting badly — prod credits this with taking
                    // acceptance from 0.60 to 0.92.
                    let d = tandem_model::qwen35::argmax(&dl);
                    let p = softmax_prob(&dl, d);
                    if p < p_min {
                        // the draft head already consumed a cache slot for
                        // this position; give it back
                        session.mtp_past -= 1;
                        break;
                    }
                    drafts.push(d);
                    tok_in = d;
                    pos += 1;
                    h = dh; // chain from the draft head's own hidden
                }
                Err(e) => {
                    eprintln!("draft: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }

        // ---- free drafts from the CPU oracle ----
        // These cost no GPU time at all, and extending a verify batch is
        // nearly free on bandwidth-bound hardware: the weights are read once
        // either way. So they can only add accepted tokens to a pass we are
        // already paying for.
        let n_mtp_drafts = drafts.len();
        if n_oracle > 0 {
            let mut prefix = vec![last];
            prefix.extend_from_slice(&drafts);
            let extra = oracle.draft(&prefix, n_oracle);
            drafts.extend_from_slice(&extra);
        }

        // ---- verify: last token + all drafts in one trunk pass ----
        let mut batch = vec![last];
        batch.extend_from_slice(&drafts);
        let outs: Vec<i32> = (0..batch.len() as i32).collect();
        let _ = &outs;
        let (vp, vh) = match model.step_verify(&mut session, &batch, threads) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("verify: {e}");
                return ExitCode::FAILURE;
            }
        };
        rounds += 1;
        proposed += drafts.len();

        // position i predicts token i+1; keep drafts while they match
        let mut n_keep = 0usize;
        for (i, d) in drafts.iter().enumerate() {
            let truth = vp[i];
            if truth == *d {
                n_keep += 1;
            } else {
                break;
            }
        }
        accepted += n_keep;
        let from_oracle = n_keep.saturating_sub(n_mtp_drafts);
        oracle_accepted += from_oracle;
        oracle.record(drafts.len().saturating_sub(n_mtp_drafts), from_oracle);

        // commit accepted drafts plus the token the trunk itself produced.
        // An accepted draft can be EOS, and nothing may follow it.
        let mut newly: Vec<u32> = Vec::with_capacity(n_keep + 1);
        for d in drafts.iter().take(n_keep) {
            committed.push(*d);
            newly.push(*d);
            if committed.len() >= n_gen || (!ignore_eos && Some(*d) == tok.eos) {
                oracle.extend(&newly);
                break 'outer;
            }
        }
        let next = vp[n_keep];
        committed.push(next);
        newly.push(next);
        // only committed tokens teach the oracle; a rejected draft must never
        // become evidence for what the model says
        oracle.extend(&newly);
        hidden = vh[n_keep * model.hp.n_embd as usize..(n_keep + 1) * model.hp.n_embd as usize]
            .to_vec();

        // undo the over-advance from rejected drafts
        let extra = drafts.len() - n_keep;
        if extra > 0 {
            if let Err(e) = model.rollback_recurrent(&mut session, extra, threads) {
                eprintln!("rollback: {e}");
                return ExitCode::FAILURE;
            }
            session.n_past -= extra;
            session.mtp_past -= extra;
        }

    }
    let t_decode = t1.elapsed().as_secs_f64();

    println!("{}", tok.decode(&committed, true));
    let rate = if proposed == 0 { 0.0 } else { accepted as f64 / proposed as f64 };
    eprintln!(
        "prefill: {} tok in {:.2}s ({:.1} tok/s) | decode: {} tok in {:.2}s ({:.2} tok/s) | \
         spec {n_spec}/p{p_min} + oracle {n_oracle}: acceptance {accepted}/{proposed} = \
         {rate:.3}, {rounds} rounds, {:.2} tok/round, oracle drafted {} kept {oracle_accepted} \
         ({:.3}, gate {:.2})",
        prompt_ids.len(),
        t_prefill,
        prompt_ids.len() as f64 / t_prefill,
        committed.len(),
        t_decode,
        committed.len() as f64 / t_decode,
        committed.len() as f64 / rounds.max(1) as f64,
        oracle.proposals,
        oracle.acceptance(),
        oracle.confidence_gate(),
    );
    ExitCode::SUCCESS
}

/// MTP acceptance probe: run a normal greedy decode, and at each step ask
/// the draft head what it thinks the NEXT-next token is. Comparing that
/// against what the trunk actually produces measures draft acceptance — the
/// number speculative decoding's speedup is entirely made of — without
/// needing the batched-verify or recurrent-rollback machinery yet.
///
/// Prod (llama.cpp, same model) measures 0.78-0.92 at draft depth 3.
fn cmd_mtp_probe(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut prompt = String::from("The capital of France is");
    let mut n_gen = 64usize;
    let mut threads = 8i32;
    let mut n_ctx = 4096usize;
    let mut gpu: Option<i32> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" => prompt = it.next().cloned().unwrap_or_default(),
            "-n" => n_gen = it.next().and_then(|s| s.parse().ok()).unwrap_or(64),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(4096),
            "--gpu" => gpu = Some(it.next().and_then(|s| s.parse().ok()).unwrap_or(0)),
            "--tp" => {
                tp = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            _ => {}
        }
    }
    let Some(path) = path else {
        eprintln!("usage: tandem mtp-probe <file.gguf> [-p prompt] [-n tokens] [--gpu N|--tp 0,1]");
        return ExitCode::from(2);
    };
    let dev = match (&tp, gpu) {
        (Some(ids), _) => tandem_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => tandem_model::Device::Cuda(i),
        (None, None) => tandem_model::Device::Cpu,
    };
    let model = match tandem_model::qwen35::Qwen35::load_on(Path::new(path), dev) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mtp-probe: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = tandem_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    let mut session = match tandem_model::qwen35::Session::new(&model, n_ctx) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("mtp-probe: {e}");
            return ExitCode::FAILURE;
        }
    };

    let ids = tok.encode(&prompt, true);
    // prefill; keep the hidden state of the last position
    let (mut logits, mut hidden) =
        match model.step_with_hidden(&mut session, &ids, &[(ids.len() - 1) as i32], threads) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("prefill: {e}");
                return ExitCode::FAILURE;
            }
        };
    let mut next = tandem_model::qwen35::argmax(&logits);

    let mut drafted: Option<u32> = None;
    let (mut hits, mut total) = (0usize, 0usize);
    let mut out_ids: Vec<u32> = vec![next];
    for i in 0..n_gen {
        // what the draft head predicted last round for THIS position
        if let Some(d) = drafted.take() {
            total += 1;
            if d == next {
                hits += 1;
            }
        }
        // draft the token after `next`, from this position's hidden state
        let pos = session.n_past; // position `next` will occupy
        match model.mtp_draft(&mut session, &hidden, next, pos, threads) {
            Ok((dl, _)) => drafted = Some(tandem_model::qwen35::argmax(&dl)),
            Err(e) => {
                eprintln!("mtp draft step {i}: {e}");
                return ExitCode::FAILURE;
            }
        }
        if Some(next) == tok.eos {
            break;
        }
        // advance the trunk one token
        match model.step_with_hidden(&mut session, &[next], &[0], threads) {
            Ok((l, h)) => {
                logits = l;
                hidden = h;
            }
            Err(e) => {
                eprintln!("trunk step {i}: {e}");
                return ExitCode::FAILURE;
            }
        }
        next = tandem_model::qwen35::argmax(&logits);
        out_ids.push(next);
    }

    println!("{}", tok.decode(&out_ids, true));
    let rate = if total == 0 { 0.0 } else { hits as f64 / total as f64 };
    eprintln!("MTP draft acceptance: {hits}/{total} = {rate:.3}");
    ExitCode::SUCCESS
}

/// Numeric self-test: stateless forward vs session path on the same tokens.
/// Case A: whole prompt as one session prefill. Case B: prompt split so the
/// last token goes through the single-token decode path. Reports max |Δlogit|
/// and per-path argmax — localizes cache/state bugs without an oracle.
fn cmd_selftest(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut prompt = String::from("The quick brown fox jumps over the lazy dog. The capital of France is");
    let mut threads = 8i32;
    let mut gpu: Option<i32> = None;
    let mut split: Option<Vec<i32>> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" => prompt = it.next().cloned().unwrap_or_default(),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "--gpu" => gpu = Some(it.next().and_then(|s| s.parse().ok()).unwrap_or(0)),
            "--tp" => {
                tp = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
            "--gpu-split" => {
                split = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            _ => {}
        }
    }
    let Some(path) = path else {
        eprintln!("usage: tandem selftest <file.gguf> [-p prompt]");
        return ExitCode::from(2);
    };
    let model = tandem_model::qwen35::Qwen35::load_on(Path::new(path), match (&tp, &split, gpu) {
        (Some(ids), _, _) => tandem_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(ids), _) => tandem_model::Device::CudaSplit(ids.clone()),
        (None, None, Some(i)) => tandem_model::Device::Cuda(i),
        (None, None, None) => tandem_model::Device::Cpu,
    }).expect("load");
    let tok = tandem_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    let ids = tok.encode(&prompt, true);
    eprintln!("{} prompt tokens", ids.len());

    let report = |name: &str, a: &[f32], b: &[f32]| {
        let maxd = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let (aa, ab) = (
            tandem_model::qwen35::argmax(a),
            tandem_model::qwen35::argmax(b),
        );
        println!(
            "{name}: max|Δ| = {maxd:.6}, argmax {} vs {} ({})",
            aa,
            ab,
            if aa == ab { "MATCH" } else { "MISMATCH" }
        );
    };

    let reference = model.forward_logits(&ids, threads).expect("stateless");

    // Case A: full-prompt prefill through the session path
    let mut s = tandem_model::qwen35::Session::new(&model, 4096).expect("session");
    let a = model
        .step(&mut s, &ids, &[(ids.len() - 1) as i32], threads)
        .expect("prefill");
    report("A prefill(all)      vs stateless", &a, &reference);

    // Case B: prefill(n-1) then decode(1)
    let mut s2 = tandem_model::qwen35::Session::new(&model, 4096).expect("session");
    let (head, tail) = ids.split_at(ids.len() - 1);
    model
        .step(&mut s2, head, &[(head.len() - 1) as i32], threads)
        .expect("prefill head");
    let b = model.step(&mut s2, tail, &[0], threads).expect("decode 1");
    report("B prefill+decode(1) vs stateless", &b, &reference);

    // Case C: token-by-token decode of the whole prompt
    let mut s3 = tandem_model::qwen35::Session::new(&model, 4096).expect("session");
    let mut c = Vec::new();
    for t in &ids {
        c = model.step(&mut s3, &[*t], &[0], threads).expect("decode all");
    }
    report("C decode-only       vs stateless", &c, &reference);

    // Case D: single-token session vs single-token stateless (isolates the
    // cache-write path with T=1 from multi-token specifics)
    let one = &ids[..1];
    let d_ref = model.forward_logits(one, threads).expect("stateless 1");
    let mut s4 = tandem_model::qwen35::Session::new(&model, 4096).expect("session");
    let d = model.step(&mut s4, one, &[0], threads).expect("session 1");
    report("D single-token      vs stateless", &d, &d_ref);

    // Case E: session-vs-session determinism (fresh sessions, same input);
    // any difference here means uninitialized memory, not a layout bug
    let mut s5 = tandem_model::qwen35::Session::new(&model, 4096).expect("session");
    let e1 = model
        .step(&mut s5, &ids, &[(ids.len() - 1) as i32], threads)
        .expect("e1");
    let mut s6 = tandem_model::qwen35::Session::new(&model, 4096).expect("session");
    let e2 = model
        .step(&mut s6, &ids, &[(ids.len() - 1) as i32], threads)
        .expect("e2");
    report("E session repeat    vs session ", &e1, &e2);
    ExitCode::SUCCESS
}

/// Stateful greedy generation: prefill once, then O(1)-per-token decode via
/// Session (KV cache + carried recurrent states). The engine path.
fn cmd_gen(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut prompt = String::from("The capital of France is");
    let mut n_gen = 16usize;
    let mut threads = 8i32;
    let mut n_ctx = 4096usize;
    let mut gpu: Option<i32> = None;
    let mut split: Option<Vec<i32>> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut ignore_eos = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" => prompt = it.next().cloned().unwrap_or_default(),
            "-n" => n_gen = it.next().and_then(|s| s.parse().ok()).unwrap_or(16),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "--gpu" => gpu = Some(it.next().and_then(|s| s.parse().ok()).unwrap_or(0)),
            "--tp" => {
                tp = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
            "--gpu-split" => {
                split = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
            "--ignore-eos" => ignore_eos = true,
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(4096),
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            s => {
                eprintln!("unknown arg: {s}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(path) = path else {
        eprintln!("usage: tandem gen <file.gguf> [-p prompt] [-n tokens] [-t threads] [-c n_ctx]");
        return ExitCode::from(2);
    };

    let model = match tandem_model::qwen35::Qwen35::load_on(Path::new(path), match (&tp, &split, gpu) {
        (Some(ids), _, _) => tandem_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(ids), _) => tandem_model::Device::CudaSplit(ids.clone()),
        (None, None, Some(i)) => tandem_model::Device::Cuda(i),
        (None, None, None) => tandem_model::Device::Cpu,
    }) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("tandem gen: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match tandem_tok::Tokenizer::from_gguf(&model.weights.gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tandem gen: {e}");
            return ExitCode::FAILURE;
        }
    };
    if model.weights.n_backends() > 1 {
        let per: Vec<String> = model
            .weights
            .bytes_per_backend
            .iter()
            .map(|b| format!("{:.2} GiB", *b as f64 / (1u64 << 30) as f64))
            .collect();
        eprintln!("weights split across {} devices: {}", model.weights.n_backends(), per.join(" | "));
    }

    let mut session = match tandem_model::qwen35::Session::new(&model, n_ctx) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("tandem gen: {e}");
            return ExitCode::FAILURE;
        }
    };

    let prompt_ids = tok.encode(&prompt, true);
    eprintln!("prompt: {} tokens", prompt_ids.len());

    // prefill (single ubatch; chunk if longer than 512). Greedy sampling
    // happens inside the graph, so only the token id crosses the bus.
    let t0 = std::time::Instant::now();
    let mut next = 0u32;
    for chunk in prompt_ids.chunks(512) {
        next = match model.step_greedy(&mut session, chunk, threads) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("prefill: {e}");
                return ExitCode::FAILURE;
            }
        };
    }
    let t_prefill = t0.elapsed().as_secs_f64();

    // decode
    let t1 = std::time::Instant::now();
    let mut gen_ids: Vec<u32> = Vec::with_capacity(n_gen);
    gen_ids.push(next);
    for _ in 1..n_gen {
        if !ignore_eos && Some(next) == tok.eos {
            break;
        }
        next = match model.step_greedy(&mut session, &[next], threads) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("decode: {e}");
                return ExitCode::FAILURE;
            }
        };
        gen_ids.push(next);
    }
    let t_decode = t1.elapsed().as_secs_f64();

    println!("{}", tok.decode(&gen_ids, true));
    eprintln!(
        "prefill: {} tok in {:.2}s ({:.1} tok/s) | decode: {} tok in {:.2}s ({:.2} tok/s)",
        prompt_ids.len(),
        t_prefill,
        prompt_ids.len() as f64 / t_prefill,
        gen_ids.len(),
        t_decode,
        gen_ids.len() as f64 / t_decode,
    );
    ExitCode::SUCCESS
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
    let mut gpu: Option<i32> = None;
    let mut split: Option<Vec<i32>> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-f" => file = it.next().map(String::as_str),
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(512),
            "--chunks" => n_chunks = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "--gpu" => gpu = Some(it.next().and_then(|s| s.parse().ok()).unwrap_or(0)),
            "--tp" => {
                tp = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
            "--gpu-split" => {
                split = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
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

    let model = match tandem_model::qwen35::Qwen35::load_on(Path::new(path), match (&tp, &split, gpu) {
        (Some(ids), _, _) => tandem_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(ids), _) => tandem_model::Device::CudaSplit(ids.clone()),
        (None, None, Some(i)) => tandem_model::Device::Cuda(i),
        (None, None, None) => tandem_model::Device::Cpu,
    }) {
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
    // logits at j for j in [first, n_ctx-1) predict token j+1
    let out_positions: Vec<i32> = (first..n_ctx - 1).map(|j| j as i32).collect();

    let n_vocab = model.hp.n_vocab as usize;
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
    let mut gpu: Option<i32> = None;
    let mut split: Option<Vec<i32>> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" => prompt = it.next().cloned().unwrap_or_default(),
            "-n" => n_gen = it.next().and_then(|s| s.parse().ok()).unwrap_or(16),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "--gpu" => gpu = Some(it.next().and_then(|s| s.parse().ok()).unwrap_or(0)),
            "--tp" => {
                tp = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
            "--gpu-split" => {
                split = it.next().map(|v| {
                    v.split(',').filter_map(|x| x.trim().parse::<i32>().ok()).collect::<Vec<_>>()
                })
            }
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
    let model = match tandem_model::qwen35::Qwen35::load_on(Path::new(path), match (&tp, &split, gpu) {
        (Some(ids), _, _) => tandem_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(ids), _) => tandem_model::Device::CudaSplit(ids.clone()),
        (None, None, Some(i)) => tandem_model::Device::Cuda(i),
        (None, None, None) => tandem_model::Device::Cpu,
    }) {
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
