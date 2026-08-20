//! codpiece CLI. First subcommand: `inspect` — header-only GGUF analysis.
//! Reads metadata + tensor directory only; never maps tensor data, never
//! touches a GPU. Safe to run against files a live server has loaded.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use codpiece_gguf::{GgufFile, TensorInfo, Value};

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
        Some("fused") => cmd_fused(&args[1..]),
        Some("stepcost") => cmd_stepcost(&args[1..]),
        Some("serve") => cmd_serve(&args[1..]),
        Some("batchtest") => cmd_batchtest(&args[1..]),
        _ => {
            eprintln!(
                "usage: codpiece inspect <file.gguf> [--tensors] [--kv <key>] [--dump-template <out>]\n\
                 usage: codpiece tokenize <file.gguf> [--special] [--pieces] < text\n\
                 usage: codpiece load <file.gguf>\n\
                 usage: codpiece serve <file.gguf> [--host H] [--port P] [-c n_ctx] [--tp 0,1]"
            );
            ExitCode::from(2)
        }
    }
}



/// Correctness gate for batched decoding: the same prompt, decoded greedily through
/// the ordinary single-sequence path and as every slot of an N-way batch, must produce
/// identical tokens from every slot. Any cross-sequence contamination — a shared state
/// slice, a mask reaching into a neighbour's KV region — shows up as a mismatch.
fn cmd_batchtest(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut prompt = String::from("The capital of France is");
    let mut n_gen = 24usize;
    let mut threads = 8i32;
    let mut n_batch = 3usize;
    let mut seq_ctx = 1024usize;
    let mut gpu: Option<i32> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" => prompt = it.next().cloned().unwrap_or_default(),
            "-n" => n_gen = it.next().and_then(|s| s.parse().ok()).unwrap_or(24),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "--batch" => n_batch = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "--seq-ctx" => seq_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(1024),
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
        eprintln!("usage: codpiece batchtest <file.gguf> [-p prompt] [-n N] [--batch B] [--tp 0,1]");
        return ExitCode::from(2);
    };
    let dev = match (&tp, gpu) {
        (Some(ids), _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None) => codpiece_model::Device::Cpu,
    };
    let model = match codpiece_model::qwen35::Qwen35::load_on(Path::new(path), dev) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("batchtest: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    let prompt_ids = tok.encode(&prompt, true);

    // reference: the ordinary single-sequence greedy path
    let mut reference = Vec::new();
    {
        let mut sess = codpiece_model::qwen35::Session::new_spec(&model, seq_ctx, 1)
            .expect("single session");
        let mut next = model
            .step_greedy(&mut sess, &prompt_ids, threads)
            .expect("single prefill");
        for _ in 0..n_gen {
            reference.push(next);
            next = model.step_greedy(&mut sess, &[next], threads).expect("single step");
        }
    }

    // batch: same prompt in every slot
    let mut sess = codpiece_model::qwen35::Session::new_spec(
        &model,
        n_batch * seq_ctx,
        n_batch.saturating_sub(1).max(1),
    )
    .expect("batch session");
    let mut lasts = vec![0u32; n_batch];
    let mut pasts = vec![0usize; n_batch];
    for slot in 0..n_batch {
        let (id, _) = model
            .step_seq_prefill(&mut sess, &prompt_ids, slot, seq_ctx, 0, false, threads)
            .expect("seq prefill");
        lasts[slot] = id;
        pasts[slot] = prompt_ids.len();
    }
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); n_batch];
    let t0 = std::time::Instant::now();
    for _ in 0..n_gen {
        for slot in 0..n_batch {
            outs[slot].push(lasts[slot]);
        }
        let (preds, _) = model
            .step_batch_decode(&mut sess, &lasts, &pasts, seq_ctx, false, threads)
            .expect("batch step");
        for slot in 0..n_batch {
            pasts[slot] += 1;
            lasts[slot] = preds[slot];
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "batch decode: {} seqs x {} tok in {:.2}s = {:.1} tok/s aggregate ({:.1} ms/round)",
        n_batch,
        n_gen,
        dt,
        (n_batch * n_gen) as f64 / dt,
        dt * 1000.0 / n_gen as f64
    );

    let mut ok = true;
    for (slot, out) in outs.iter().enumerate() {
        if out == &reference {
            println!("slot {slot}: IDENTICAL to single-path reference");
        } else {
            ok = false;
            let div = out
                .iter()
                .zip(&reference)
                .position(|(a, b)| a != b)
                .unwrap_or(out.len().min(reference.len()));
            println!(
                "slot {slot}: DIFFERS at token {div}: {:?} vs {:?}",
                &out[div..(div + 3).min(out.len())],
                &reference[div..(div + 3).min(reference.len())]
            );
        }
    }
    println!("reference text: {:?}", tok.decode(&reference, true));
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Serve the model over HTTP.
fn cmd_serve(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut host = String::from("127.0.0.1");
    let mut port = 8080u16;
    let mut n_ctx = 8192usize;
    let mut threads = 8i32;
    // Fixed 3, not adaptive. Adaptive depth is implemented and available with
    // `--depth 0`, but it has now measured neutral-or-worse three separate times —
    // most recently 44.2 tok/s against 46.5 for fixed 3 on the box's own benchmark.
    // It cannot price a depth it has not run, and paying to find out costs more than
    // the choice is worth on this model.
    let mut depth = 3usize;
    let mut default_max_tokens = 512usize;
    let mut draft_gate = 0.0f32;
    let mut served_name: Option<String> = None;
    let mut think_budget = 4096usize;
    let mut serve_max_depth = 3usize;
    let mut gpu: Option<i32> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--host" => host = it.next().cloned().unwrap_or(host),
            "--port" => port = it.next().and_then(|s| s.parse().ok()).unwrap_or(port),
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(n_ctx),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(threads),
            "--depth" => depth = it.next().and_then(|s| s.parse().ok()).unwrap_or(depth),
            "--max-depth" => {
                serve_max_depth = it.next().and_then(|s| s.parse().ok()).unwrap_or(3)
            }
            "--draft-gate" => {
                draft_gate = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0)
            }
            "--alias" | "--served-name" => served_name = it.next().cloned(),
            "--think-budget" => {
                think_budget = it.next().and_then(|s| s.parse().ok()).unwrap_or(4096)
            }
            "--max-tokens" => {
                default_max_tokens = it.next().and_then(|s| s.parse().ok()).unwrap_or(512)
            }
            "--gpu" => gpu = Some(it.next().and_then(|s| s.parse().ok()).unwrap_or(0)),
            "--tp" => {
                tp = it.next().map(|v| {
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
        eprintln!(
            "usage: codpiece serve <file.gguf> [--host H] [--port P] [-c n_ctx] [-t threads]\n\
             \x20      [--depth K|0=adaptive] [--max-depth M] [--draft-gate P]\n\
             \x20      [--max-tokens N] [--alias NAME] [--think-budget N] [--tp 0,1 | --gpu N]"
        );
        return ExitCode::from(2);
    };
    match codpiece_server::run(codpiece_server::ServeConfig {
        model_path: path.to_string(),
        host,
        port,
        n_ctx,
        threads,
        tp,
        gpu,
        depth,
        max_depth: serve_max_depth,
        draft_gate,
        default_max_tokens,
        served_name,
        think_budget,
    }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("serve: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Measure what a decode step actually costs as a function of how many tokens
/// it carries.
///
/// This is the curve speculation lives or dies on. If the machine were purely
/// bandwidth-bound, a step would cost the same for 1 token as for 16 — the
/// weights are read once either way — and the right move would be to draft as
/// deep as acceptance allows. Every extra millisecond that shows up as T grows
/// is something other than bandwidth, and it caps useful draft depth.
///
/// Reports steady-state per-step time, so graph build and warmup are excluded.
fn cmd_stepcost(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut threads = 8i32;
    let mut n_ctx = 4096usize;
    let mut gpu: Option<i32> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut reps = 8usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(4096),
            "-r" => reps = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
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
        eprintln!("usage: codpiece stepcost <file.gguf> [-r reps] [--gpu N|--tp 0,1]");
        return ExitCode::from(2);
    };
    let dev = match (&tp, gpu) {
        (Some(ids), _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None) => codpiece_model::Device::Cpu,
    };
    let model = match codpiece_model::qwen35::Qwen35::load_on(Path::new(path), dev) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("stepcost: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    let seed = tok.encode("The quick brown fox jumps over the lazy dog. ", true);

    println!("tokens/step   ms/step   ms/token   vs T=1");
    let mut base = 0f64;
    for t_len in [1usize, 2, 4, 8, 16] {
        let mut session =
            match codpiece_model::qwen35::Session::new_spec(&model, n_ctx, t_len) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("session: {e}");
                    return ExitCode::FAILURE;
                }
            };
        // warm the graph and the cache so we time steady state
        let batch: Vec<u32> = (0..t_len).map(|i| seed[i % seed.len()]).collect();
        let outs: Vec<i32> = (0..t_len as i32).collect();
        for _ in 0..2 {
            if let Err(e) = model.step(&mut session, &batch, &outs, threads) {
                eprintln!("warmup t={t_len}: {e}");
                return ExitCode::FAILURE;
            }
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            if let Err(e) = model.step(&mut session, &batch, &outs, threads) {
                eprintln!("step t={t_len}: {e}");
                return ExitCode::FAILURE;
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        if t_len == 1 {
            base = ms;
        }
        println!(
            "{t_len:>11}   {ms:>7.2}   {:>8.2}   {:>5.2}x",
            ms / t_len as f64,
            ms / base
        );
    }
    ExitCode::SUCCESS
}

/// Speculative generation where the draft head rides inside the verify graph.
///
/// The classic loop runs K+1 graph executions per round: one per draft, plus
/// the verify. Measured on the 27B, a draft from a one-layer head costs ~8 ms
/// against ~1.4 ms of bandwidth — almost all of it graph construction and
/// allocation. So this mode runs exactly ONE execution per round: the verify
/// pass carries the draft head as a tail and emits, for every verified
/// position, the draft to use if exactly that many proposals were accepted.
/// The host picks the one matching what it kept.
///
/// Still lossless: drafts are only ever kept when they equal what the model

/// itself produced.
fn cmd_fused(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut prompt = String::from("The capital of France is");
    let mut prompt_file: Option<String> = None;
    let mut n_gen = 64usize;
    let mut threads = 8i32;
    let mut n_ctx = 4096usize;
    // Depth 3 is the best single setting across the prompts measured (62.8 / 68.4 /
    // 89.3 tok/s on prose, code and arithmetic); depth 1 was only ever a starting point.
    let mut depth = 3usize;
    let mut adaptive = false;
    // Three is the ceiling by default, not five: measured, depth 4 beat depth 3 by ~1%
    // on the most predictable prompt and lost on the rest, while each extra depth adds
    // graph shapes that each hold a compute buffer.
    let mut max_depth = 3usize;
    let mut gpu: Option<i32> = None;
    let mut tp: Option<Vec<i32>> = None;
    let mut ignore_eos = false;
    let mut force_path: Option<String> = None;
    let mut n_cand = 0usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" => prompt = it.next().cloned().unwrap_or_default(),
            "-f" => prompt_file = it.next().cloned(),
            "-n" => n_gen = it.next().and_then(|s| s.parse().ok()).unwrap_or(64),
            "-t" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(8),
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(4096),
            "--depth" => match it.next().map(|s| s.as_str()) {
                Some("auto") => adaptive = true,
                Some(v) => depth = v.parse().unwrap_or(3),
                None => {}
            },
            "--max-depth" => max_depth = it.next().and_then(|s| s.parse().ok()).unwrap_or(3),
            "--ignore-eos" => ignore_eos = true,
            "--path" => force_path = it.next().cloned(),
            "--cand" => n_cand = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
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
    // A long-context prompt does not fit on a command line; read it from a file.
    if let Some(f) = prompt_file.as_deref() {
        match std::fs::read_to_string(f) {
            Ok(t) => prompt = t,
            Err(e) => {
                eprintln!("read {f}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(path) = path else {
        eprintln!("usage: codpiece fused <file.gguf> [-p prompt] [-n N] [--depth K] [--tp 0,1] [--path cached|rebuild] [--cand N]\n\
             \x20      --depth auto [--max-depth M]  choose the chain length per round");
        return ExitCode::from(2);
    };
    let dev = match (&tp, gpu) {
        (Some(ids), _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None) => codpiece_model::Device::Cpu,
    };
    let model = match codpiece_model::qwen35::Qwen35::load_on(Path::new(path), dev) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("fused: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    // Adaptive depth needs snapshot slots for the deepest chain it may choose.
    let slots_for = if adaptive { max_depth } else { depth };
    let mut session = match codpiece_model::qwen35::Session::new_spec(&model, n_ctx, slots_for) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("fused: {e}");
            return ExitCode::FAILURE;
        }
    };

    // The cached path builds each graph once and replays it; the rebuild path re-emits
    // ~5000 nodes every round. Both are lossless — this only selects which one pays the
    // build cost, and `rebuild` is kept as the A/B control.
    let use_cached = match force_path.as_deref() {
        Some("rebuild") => false,
        _ => true,
    };
    eprintln!("fused: path={}", if use_cached { "cached" } else { "rebuild" });
    let tp_mode = !use_cached;
    let round = |session: &mut codpiece_model::qwen35::Session,
                 batch: &[u32],
                 d: usize,
                 cands: Option<&[i32]>,
                 last_only: bool| {
        if tp_mode {
            // The rebuild control always emits every position; keep its contract
            // identical to the cached path by discarding all but the last. It is a
            // debug path, so it still pays the full-width logits and is not usable at
            // long context — that is exactly what `last_only` exists to avoid.
            model
                .step_verify_drafting(session, batch, d, cands, threads)
                .map(|(p, c)| {
                    if last_only {
                        let i = p.len() - 1;
                        (vec![p[i]], c.into_iter().map(|r| vec![r[i]]).collect())
                    } else {
                        (p, c)
                    }
                })
        } else {
            model
                .step_fused_cached(session, batch, d, cands, last_only, false, threads)
                .map(|(p, c, _logits, _hidden)| (p, c))
        }
    };
    let mut picker = codpiece_server::depth::DepthPicker::new(adaptive, depth, max_depth);

    let prompt_ids = tok.encode(&prompt, true);
    let t0 = std::time::Instant::now();
    let mut next_depth = picker.choose();
    // Prefill: bulk through the trunk at a large chunk, then a short tail through the
    // speculative round. The speculative graph's logits are n_vocab x chunk and will not
    // allocate at 256 on a long context, so running the whole prompt through it forces a
    // tiny chunk on the entire prefill — measured at 849 tok/s against 1331 for the same
    // prompt trunk-only. Only the tail needs the draft head, to leave its cache warm and
    // produce the first drafts.
    const PREFILL_TAIL: usize = 64;
    let bulk_chunk = codpiece_server::engine::prefill_chunk_for(n_ctx);
    let split = prompt_ids.len().saturating_sub(PREFILL_TAIL);
    let mut preds;
    let mut chain;
    for chunk in prompt_ids[..split].chunks(bulk_chunk) {
        if let Err(e) = model.step_greedy(&mut session, chunk, threads) {
            eprintln!("prefill: {e}");
            return ExitCode::FAILURE;
        }
    }
    match round(&mut session, &prompt_ids[split..], next_depth, None, false) {
        Ok(v) => {
            preds = v.0;
            chain = v.1;
        }
        Err(e) => {
            eprintln!("prefill: {e}");
            return ExitCode::FAILURE;
        }
    }
    session.mtp_past += preds.len();
    let t_prefill = t0.elapsed().as_secs_f64();
    // The prefill shapes are dead now, and at long context their compute buffers are
    // the difference between decoding and running out of memory.
    session.clear_fused_cache();

    let last_idx = preds.len() - 1;
    let mut committed: Vec<u32> = vec![preds[last_idx]];
    // the chain gives this round's draft sequence, read at the last position
    let mut drafts: Vec<u32> = chain.iter().map(|c| c[last_idx]).collect();
    next_depth = picker.choose();
    drafts.truncate(next_depth);
    let (mut accepted, mut proposed, mut rounds) = (0usize, 0usize, 0usize);

    // Candidate-restricted drafting: the DRAFT head projects onto a shortlist instead
    // of all 248,320 rows. Verification always uses the full vocabulary, so a token the
    // shortlist misses only costs that draft, never correctness.
    //
    // The shortlist is whatever the sequence itself has already used (text reuses its own
    // vocabulary heavily) topped up with low ids, which in a BPE vocabulary are the
    // common byte-level and early-merge tokens. CODPIECE_CAND_DUMMY=1 fills it with 0..N
    // to measure the mechanism's cost with acceptance deliberately destroyed.
    let dummy_cand = std::env::var("CODPIECE_CAND_DUMMY").as_deref() == Ok("1");
    let build_cands = |ctx: &[u32], n: usize| -> Vec<i32> {
        let mut out: Vec<i32> = Vec::with_capacity(n);
        let mut seen = std::collections::HashSet::with_capacity(n * 2);
        if !dummy_cand {
            for t in ctx.iter().rev() {
                if out.len() >= n {
                    break;
                }
                if seen.insert(*t) {
                    out.push(*t as i32);
                }
            }
        }
        let mut i = 0u32;
        while out.len() < n {
            if seen.insert(i) {
                out.push(i as i32);
            }
            i += 1;
        }
        out
    };

    let t1 = std::time::Instant::now();
    'outer: while committed.len() < n_gen {
        let last = *committed.last().unwrap();
        if !ignore_eos && Some(last) == tok.eos {
            break;
        }
        let mut batch = vec![last];
        batch.extend_from_slice(&drafts);
        let cands = if n_cand > 0 {
            let mut ctx = prompt_ids.clone();
            ctx.extend_from_slice(&committed);
            Some(build_cands(&ctx, n_cand))
        } else {
            None
        };
        let d = next_depth;
        let t_round = std::time::Instant::now();
        let (preds, chain) = match round(&mut session, &batch, d, cands.as_deref(), false) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("round: {e}");
                return ExitCode::FAILURE;
            }
        };
        let round_secs = t_round.elapsed().as_secs_f64();
        rounds += 1;
        proposed += drafts.len();

        // keep the drafts the model itself would have produced
        let mut n_keep = 0usize;
        for (i, d) in drafts.iter().enumerate() {
            if preds[i] == *d {
                n_keep += 1;
            } else {
                break;
            }
        }
        accepted += n_keep;
        // the trial is over the drafts this batch actually carried
        picker.observe(drafts.len(), d, n_keep, round_secs);

        for d in drafts.iter().take(n_keep) {
            committed.push(*d);
            if committed.len() >= n_gen || (!ignore_eos && Some(*d) == tok.eos) {
                break 'outer;
            }
        }
        committed.push(preds[n_keep]);
        // next round's drafts come from the chain at the accepted position
        drafts = chain.iter().map(|c| c[n_keep]).collect();
        // Decide the next depth now and drop any drafts past it. A round's graph is
        // shaped by (drafts carried in, chain length produced); letting those disagree
        // turns M depths into an MxM matrix of shapes, each with its own compute
        // buffer. Trimming costs a draft that was already computed and keeps a
        // reduction free of transitional shapes entirely.
        next_depth = picker.choose();
        drafts.truncate(next_depth);

        let extra = drafts.len().saturating_sub(n_keep);
        let over = batch.len() - (n_keep + 1);
        session.mtp_past += n_keep + 1;
        if over > 0 {
            if let Err(e) = model.rollback_recurrent(&mut session, over, threads) {
                eprintln!("rollback: {e}");
                return ExitCode::FAILURE;
            }
            session.n_past -= over;
        }
        let _ = extra;
    }
    let t_decode = t1.elapsed().as_secs_f64();

    println!("{}", tok.decode(&committed, true));
    let rate = if proposed == 0 { 0.0 } else { accepted as f64 / proposed as f64 };
    let dlabel = if adaptive { "auto".to_string() } else { depth.to_string() };
    let picker_report = picker.report();
    eprintln!(
        "prefill: {} tok in {:.2}s ({:.1} tok/s) | decode: {} tok in {:.2}s ({:.2} tok/s) | \
         fused depth {dlabel}: acceptance {accepted}/{proposed} = {rate:.3}, {rounds} rounds, \
         {:.2} tok/round{picker_report}",
        prompt_ids.len(),
        t_prefill,
        prompt_ids.len() as f64 / t_prefill,
        committed.len(),
        t_decode,
        committed.len() as f64 / t_decode,
        committed.len() as f64 / rounds.max(1) as f64,
    );
    ExitCode::SUCCESS
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
    // starts strict; the gate loosens itself once drafts start landing
    let mut oracle_conf = 0.85f32;
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
        eprintln!("usage: codpiece spec <file.gguf> [-p prompt] [-n N] [--spec K] [--gpu N|--tp 0,1]");
        return ExitCode::from(2);
    };
    let dev = match (&tp, gpu) {
        (Some(ids), _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None) => codpiece_model::Device::Cpu,
    };
    let model = match codpiece_model::qwen35::Qwen35::load_on(Path::new(path), dev) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("spec: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    // rollback slots must cover every draft a round can produce, from both
    // drafters
    let mut session =
        match codpiece_model::qwen35::Session::new_spec(&model, n_ctx, n_spec + n_oracle) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("spec: {e}");
                return ExitCode::FAILURE;
            }
        };

    let prompt_ids = tok.encode(&prompt, true);
    // The oracle learns from the prompt before generation starts: quoting the
    // input back is the single most common free-token case.
    let mut oracle = codpiece_model::oracle::ContextOracle::new(oracle_conf);
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

    let mut committed: Vec<u32> = vec![codpiece_model::qwen35::argmax(&logits)];
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
                    let d = codpiece_model::qwen35::argmax(&dl);
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
        eprintln!("usage: codpiece mtp-probe <file.gguf> [-p prompt] [-n tokens] [--gpu N|--tp 0,1]");
        return ExitCode::from(2);
    };
    let dev = match (&tp, gpu) {
        (Some(ids), _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None) => codpiece_model::Device::Cpu,
    };
    let model = match codpiece_model::qwen35::Qwen35::load_on(Path::new(path), dev) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mtp-probe: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    let mut session = match codpiece_model::qwen35::Session::new(&model, n_ctx) {
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
    let mut next = codpiece_model::qwen35::argmax(&logits);

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
            Ok((dl, _)) => drafted = Some(codpiece_model::qwen35::argmax(&dl)),
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
        next = codpiece_model::qwen35::argmax(&logits);
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
        eprintln!("usage: codpiece selftest <file.gguf> [-p prompt]");
        return ExitCode::from(2);
    };
    let model = codpiece_model::qwen35::Qwen35::load_on(Path::new(path), match (&tp, &split, gpu) {
        (Some(ids), _, _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(ids), _) => codpiece_model::Device::CudaSplit(ids.clone()),
        (None, None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None, None) => codpiece_model::Device::Cpu,
    }).expect("load");
    let tok = codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf).expect("tok");
    let ids = tok.encode(&prompt, true);
    eprintln!("{} prompt tokens", ids.len());

    let report = |name: &str, a: &[f32], b: &[f32]| {
        let maxd = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let (aa, ab) = (
            codpiece_model::qwen35::argmax(a),
            codpiece_model::qwen35::argmax(b),
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
    let mut s = codpiece_model::qwen35::Session::new(&model, 4096).expect("session");
    let a = model
        .step(&mut s, &ids, &[(ids.len() - 1) as i32], threads)
        .expect("prefill");
    report("A prefill(all)      vs stateless", &a, &reference);

    // Case B: prefill(n-1) then decode(1)
    let mut s2 = codpiece_model::qwen35::Session::new(&model, 4096).expect("session");
    let (head, tail) = ids.split_at(ids.len() - 1);
    model
        .step(&mut s2, head, &[(head.len() - 1) as i32], threads)
        .expect("prefill head");
    let b = model.step(&mut s2, tail, &[0], threads).expect("decode 1");
    report("B prefill+decode(1) vs stateless", &b, &reference);

    // Case C: token-by-token decode of the whole prompt
    let mut s3 = codpiece_model::qwen35::Session::new(&model, 4096).expect("session");
    let mut c = Vec::new();
    for t in &ids {
        c = model.step(&mut s3, &[*t], &[0], threads).expect("decode all");
    }
    report("C decode-only       vs stateless", &c, &reference);

    // Case D: single-token session vs single-token stateless (isolates the
    // cache-write path with T=1 from multi-token specifics)
    let one = &ids[..1];
    let d_ref = model.forward_logits(one, threads).expect("stateless 1");
    let mut s4 = codpiece_model::qwen35::Session::new(&model, 4096).expect("session");
    let d = model.step(&mut s4, one, &[0], threads).expect("session 1");
    report("D single-token      vs stateless", &d, &d_ref);

    // Case E: session-vs-session determinism (fresh sessions, same input);
    // any difference here means uninitialized memory, not a layout bug
    let mut s5 = codpiece_model::qwen35::Session::new(&model, 4096).expect("session");
    let e1 = model
        .step(&mut s5, &ids, &[(ids.len() - 1) as i32], threads)
        .expect("e1");
    let mut s6 = codpiece_model::qwen35::Session::new(&model, 4096).expect("session");
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
    let mut sp = codpiece_sample::SamplerParams::default();
    let mut prompt_file: Option<String> = None;
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
            "-f" => prompt_file = it.next().cloned(),
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
            "--temp" => sp.temp = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            "--top-k" => sp.top_k = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--top-p" => sp.top_p = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "--min-p" => sp.min_p = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            "--repeat-penalty" => {
                sp.penalty_repeat = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0)
            }
            "--repeat-last-n" => {
                sp.penalty_last_n = it.next().and_then(|s| s.parse().ok()).unwrap_or(64)
            }
            "--seed" => sp.seed = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "-c" => n_ctx = it.next().and_then(|s| s.parse().ok()).unwrap_or(4096),
            s if !s.starts_with('-') && path.is_none() => path = Some(s),
            s => {
                eprintln!("unknown arg: {s}");
                return ExitCode::from(2);
            }
        }
    }
    // A long-context prompt does not fit on a command line; read it from a file.
    if let Some(f) = prompt_file.as_deref() {
        match std::fs::read_to_string(f) {
            Ok(t) => prompt = t,
            Err(e) => {
                eprintln!("read {f}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(path) = path else {
        eprintln!(
            "usage: codpiece gen <file.gguf> [-p prompt|-f file] [-n tokens] [-t threads]\n\
             \x20      [-c n_ctx] [--temp T] [--top-k K] [--top-p P] [--min-p P]\n\
             \x20      [--repeat-penalty R] [--repeat-last-n N] [--seed S]"
        );
        return ExitCode::from(2);
    };

    let model = match codpiece_model::qwen35::Qwen35::load_on(Path::new(path), match (&tp, &split, gpu) {
        (Some(ids), _, _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(ids), _) => codpiece_model::Device::CudaSplit(ids.clone()),
        (None, None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None, None) => codpiece_model::Device::Cpu,
    }) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("codpiece gen: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("codpiece gen: {e}");
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

    let mut session = match codpiece_model::qwen35::Session::new(&model, n_ctx) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("codpiece gen: {e}");
            return ExitCode::FAILURE;
        }
    };

    let prompt_ids = tok.encode(&prompt, true);
    eprintln!("prompt: {} tokens", prompt_ids.len());

    // prefill (single ubatch; chunk if longer than 512). Greedy sampling
    // happens inside the graph, so only the token id crosses the bus.
    // Greedy keeps sampling inside the graph, so only a token id crosses the bus.
    // Anything else needs the vocabulary on the host: ~1 MB per token, about 0.1 ms on
    // this link against a ~26 ms step, which is why it is gated on the parameters rather
    // than always paid.
    let greedy = sp.is_greedy();
    let mut sampler = codpiece_sample::Sampler::new(sp.clone());
    eprintln!(
        "sampling: {}",
        if greedy {
            "greedy (in-graph argmax)".to_string()
        } else {
            format!(
                "temp {:.2} top_k {} top_p {:.2} min_p {:.2} repeat {:.2} seed {}",
                sp.temp, sp.top_k, sp.top_p, sp.min_p, sp.penalty_repeat, sp.seed
            )
        }
    );
    let t0 = std::time::Instant::now();
    let mut next = 0u32;
    let prefill_chunk: usize = std::env::var("CODPIECE_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or_else(|| codpiece_server::engine::prefill_chunk_for(n_ctx));
    sampler.accept_all(&prompt_ids);
    for (ci, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {
        let last = (ci + 1) * prefill_chunk >= prompt_ids.len();
        if greedy {
            next = match model.step_greedy(&mut session, chunk, threads) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("prefill: {e}");
                    return ExitCode::FAILURE;
                }
            };
        } else {
            // only the final chunk's distribution is sampled from
            let outs = [(chunk.len() - 1) as i32];
            match model.step(&mut session, chunk, &outs, threads) {
                Ok(logits) => {
                    if last {
                        next = sampler.sample(&logits);
                    }
                }
                Err(e) => {
                    eprintln!("prefill: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    sampler.accept(next);
    let t_prefill = t0.elapsed().as_secs_f64();

    // decode
    let t1 = std::time::Instant::now();
    let mut gen_ids: Vec<u32> = Vec::with_capacity(n_gen);
    gen_ids.push(next);
    for _ in 1..n_gen {
        if !ignore_eos && Some(next) == tok.eos {
            break;
        }
        next = match (if greedy {
            model.step_greedy(&mut session, &[next], threads)
        } else {
            model
                .step(&mut session, &[next], &[0], threads)
                .map(|logits| sampler.sample(&logits))
        }) {
            Ok(t) => {
                sampler.accept(t);
                t
            }
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
        eprintln!("usage: codpiece ppl <file.gguf> -f <text> [-c n_ctx] [--chunks N] [-t threads]");
        return ExitCode::from(2);
    };

    let model = match codpiece_model::qwen35::Qwen35::load_on(Path::new(path), match (&tp, &split, gpu) {
        (Some(ids), _, _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(ids), _) => codpiece_model::Device::CudaSplit(ids.clone()),
        (None, None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None, None) => codpiece_model::Device::Cpu,
    }) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("codpiece ppl: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("codpiece ppl: {e}");
            return ExitCode::FAILURE;
        }
    };
    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("codpiece ppl: {file}: {e}");
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
        eprintln!("usage: codpiece run <file.gguf> [-p prompt] [-n tokens] [-t threads]");
        return ExitCode::from(2);
    };

    let t0 = std::time::Instant::now();
    let model = match codpiece_model::qwen35::Qwen35::load_on(Path::new(path), match (&tp, &split, gpu) {
        (Some(ids), _, _) => codpiece_model::Device::CudaTensorParallel(ids.clone()),
        (None, Some(ids), _) => codpiece_model::Device::CudaSplit(ids.clone()),
        (None, None, Some(i)) => codpiece_model::Device::Cuda(i),
        (None, None, None) => codpiece_model::Device::Cpu,
    }) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("codpiece run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match codpiece_tok::Tokenizer::from_gguf(&model.weights.gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("codpiece run: {e}");
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
        let next = codpiece_model::qwen35::argmax(&logits);
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
        eprintln!("usage: codpiece load <file.gguf>");
        return ExitCode::from(2);
    };
    let t0 = std::time::Instant::now();
    match codpiece_model::Weights::load(Path::new(path), codpiece_model::Device::Cpu) {
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
            eprintln!("codpiece load: {e}");
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
        eprintln!("usage: codpiece tokenize <file.gguf> [--special] [--pieces] < text");
        return ExitCode::from(2);
    };

    let g = match GgufFile::open(Path::new(path)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("codpiece tokenize: {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match codpiece_tok::Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("codpiece tokenize: {e}");
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
        eprintln!("usage: codpiece inspect <file.gguf> [...]");
        return ExitCode::from(2);
    };

    let g = match GgufFile::open(Path::new(path)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("codpiece inspect: {path}: {e}");
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
