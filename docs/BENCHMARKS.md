# Standard benchmarks — qwen35 family through codpiece

Measured 2026-08-20/21 on llm-host (2x RTX 3090), all through codpiece
itself: perplexity via `codpiece ppl` (session path, zeroed state per
chunk), GSM8K via lm-evaluation-harness against codpiece's OpenAI API
(12-way batched serving, `local-completions`, no chat template — the
standard raw 5-shot protocol, full 1319-question test set).

| model | wikitext-2 PPL* | GSM8K 5-shot strict | GSM8K flexible |
|---|---|---|---|
| Qwen3.8-27B UD-Q8_K_XL (prod) | 6.0844 | **64.9% ± 1.3** | 64.2% |
| Qwen3.8-27B UD-Q6_K_XL-v3 | 6.0842 | 62.0% ± 1.3 | 61.3% |
| Qwen3.5-0.8B BF16 | 16.64 | 29.4% ± 1.3 | 30.0% |
| Qwen3.5-0.8B Q8_0 | 16.63 | — | — |

\* 24 x 512-token chunks, scoring positions 256..511 of each window (every
scored token has >= 256 tokens of context). Not comparable to full-window
protocols (the repo's historic 20.44 figure scores all positions).

## Findings

1. **Perplexity cannot see what GSM8K sees.** Q8 and Q6 are identical to
   four significant figures on wikitext-2, yet Q6 loses ~3 GSM8K points
   (1.6 combined sigma). Loss-per-token on natural text averages away the
   damage that multi-step reasoning compounds. Choosing quants by PPL
   alone would have shipped the worse model with a clean conscience —
   the accuracy-first UD-Q8_K_XL decision now has task-level evidence.
2. **The raw 5-shot protocol suppresses these models.** 64.9% is the
   comparable-across-models harness number, far below what the 27B
   scores with its chat template and thinking mode. Use these figures
   for relative comparisons, not capability headlines.
3. **Scale gap:** the 0.8B holds 29.4% at 34x fewer parameters — and its
   Q8_0 quant is perplexity-identical to BF16, consistent with the 27B
   pattern that Q8-class quantization is invisible to next-token loss.
4. **Harness compatibility note:** generation-based tasks run as-is over
   codpiece's API; loglikelihood-based suites (MMLU, HellaSwag) need a
   logprobs field `/v1/completions` does not expose yet.

## Code

The use case is coding, and until now nothing here measured code: GSM8K
measures arithmetic reasoning and wikitext measures next-token loss.

| benchmark | Qwen3.8-27B UD-Q8_K_XL |
|---|---|
| HumanEval pass@1 (all 164) | **79.3% ± 3.2** |
| MBPP pass@1 (3-shot, 200) | 0.0 — see below, this is a protocol artifact |

Run through codpiece's own API via lm-evaluation-harness (`local-completions`,
greedy, 0-shot, the harness's standard stop sequences).

**MBPP's 0.0 measures protocol conformance, not coding ability, and is not
reported as a capability number.** lm-eval's `mbpp` task is a base-model
completion protocol: three examples, code between `[BEGIN]` and `[DONE]`,
nothing else. Probed directly, this instruct model answers that prompt the
way an instruct model does — a signature with `# Your code here`, the
marker `[END]` rather than `[DONE]`, then several paragraphs explaining its
approach. The stop never fires, the extracted "code" is a stub plus prose,
and every test fails. HumanEval, which hands the model a real signature and
docstring to continue, suits it and scores 79.3%. This is the same effect
already noted for GSM8K's raw 5-shot protocol, in a more absolute form:
a benchmark can measure the harness's conventions rather than the model.

**Sandboxing.** HumanEval and MBPP score by *executing model-generated
code*; the harness gates this behind both `--confirm_run_unsafe_code` and
`HF_ALLOW_CODE_EVAL=1`. Neither was granted on the host. The harness runs
inside a throwaway container with no host filesystem mounts, reaching the
server over the API only, so generated code cannot touch the box — that
isolation is the reason the gate is opened at all, not a formality worked
around.

## MMLU — and the off-by-one that made it look like chance

Loglikelihood-scored, so it runs entirely on the `echo` + `logprobs` path
and could not run at all before that existed. 0-shot, 60 questions per
subject, scored by token ids the harness tokenizes itself.

| subject | before the fix | after |
|---|---|---|
| college computer science | 25.0% | **81.7% ± 5.0** |
| high school mathematics | 21.7% | **60.0% ± 6.4** |
| professional law | 26.7% | **68.3% ± 6.1** |

The first column is chance (four choices). The scoring was never wrong —
each of these was checked directly:

- hand-scoring an MMLU question's four continuations gives the correct
  answer −0.658 against −8.8, −9.2 and −11.7;
- a 60-token prefix scored alone and inside a 131-token prompt agrees on
  **0 of 59** positions differing, so chunked scoring is exact;
- the tokenizer lm-eval downloads produces byte-identical ids to the
  GGUF's own, both directions.

The bug was the response *shape*. OpenAI, given `echo` with
`max_tokens: 1`, returns the prompt tokens **plus the generated one**, and
the harness reads `token_logprobs[ctxlen:-1]` — dropping that trailing
entry. Returning only the prompt made that slice drop the last *prompt*
token instead, which for MMLU is the answer letter. A one-token
continuation then leaves an empty list, `sum([]) == 0.0` for all four
candidates, they tie, argmax takes the first, and the result is chance by
construction — on any harness that slices this way.

Found by pointing the harness at a logging HTTP stub, reading the request
it actually sends, then reading its parser. Worth the detour: a
chance-level score reads as a model or engine property, and this was
neither.

Two other integration defects surfaced the same way and are fixed: the
harness sends prompts in OpenAI's batch form `[[ids]]` (rejected with a
400 that, from its side, looked like no logprobs support at all), and
scoring requests could be swept into the batch path, which would have
answered them as ordinary generation with no logprobs field.

## Long-context retrieval

Needle-in-a-haystack through `/v1/chat/completions`, unique filler lines
(a maintenance log, one fact buried in it), greedy, thinking off:

| prompt tokens | depth 10% | depth 50% | depth 90% |
|---|---|---|---|
| 6,122 | found | found | found |
| 24,347 | found | found | found |
| **97,298** | **found** | **found** | **found** |

9/9, including at 97K tokens — retrieval at the full shipping context,
not merely allocation of it.

**This test replaced one that lied.** The first version buried the fact
in ~66K repetitions of a *single sentence* and asked through
`/v1/completions` with no chat template. It missed at 66K, 80K and 93K,
which looked exactly like a long-context defect and was not: a
degenerate haystack gives the model nothing to distinguish position by,
and the raw endpoint gives it no instruction framing. Two hypotheses
were checked and killed before the harness was suspected — the model is
natively 262K with `rope.freq_base = 10M` and no scaling keys (so no
YaRN is owed), and cached graphs always round `n_kv` *up* to a bucket
capped at `n_ctx`, so the causal mask never truncates. A benchmark that
fails should be suspected as hard as the code it accuses.

## What context costs, on 48 GiB

Every context decision on this box is a memory decision, so the server
reports per-GPU VRAM at startup and (with `CODPIECE_TRACE_VRAM=1`) per
request. Measured with the full production stack resident — Q8 weights,
DFlash Q4 drafter, vision tower — and one session:

| context | card 0 free | card 1 free | verdict |
|---|---|---|---|
| 98304 | 2.00 GiB | 1.15 GiB | fits; settles near 0.2 GiB under load |
| 114688 | 1.50 GiB | 0.65 GiB | peak 24.09 of 24.58 GiB — too close |
| 131072 | — | — | 500s on the first request |
| 131072, no vision | 1.00 GiB | 1.01 GiB | fits; the vision tower costs ~0.87 GiB |

Card 1 is always the tight one: it carries the vision tower. The drafter
is mirrored, so it costs its ~1.2 GiB on *both* cards.

Three defects surfaced here, all of them memory accounting that was blind
to what was actually resident:

1. **Compiled graph shapes were cached by count, not size.** A prefill
   graph's compute buffer scales with `n_kv * chunk` — ~78 MiB each at
   98K — and twelve of those do not fit in 1.15 GiB. Consecutive long
   requests land in different `n_kv` buckets (prefix reuse guarantees
   it), so the cache grew per request until an allocation failed. Now
   bounded by bytes, defaulting to 40% of measured free VRAM.
2. **The prefill chunk was sized from a static formula** that knew
   nothing about the drafter or the vision tower. At 98K it computed
   4 GiB free where the driver reported 1.15, and chose the largest
   chunk accordingly. It now reads real headroom — at a measured cost of
   ~20% long-prompt prefill throughput, which is the price of not dying.
3. **Failed allocations are not always recoverable.** `Session::new_spec`
   checks its buffer for null, yet creating the batch session still took
   the process down: the failure aborts deeper in the backend than any
   error path reaches. Sessions are now *priced before they are created*
   and declined with a log line if they will not fit:

   ```
   serve: batch session needs ~960 MiB per card (+25% margin) but only
          484 MiB is free; declining it rather than risking the process
   ```

   Serving serially is slower; taking the process down is worse.

## Anatomy of a single-stream round, and three dead ends

Measured with `CODPIECE_DECODE_TRACE=1` at ctx 16384 and 98304 — which are
**the same to within noise**, so nothing below is a context effect:

| | baseline | no lattice | 1-token forward |
|---|---|---|---|
| verify | 42.1 ms (8 tokens) | 27.5 ms (1 token) | — |
| draft: inject | 0.73 ms | 0.31 ms | — |
| draft: block | 14.20 ms | 10.22 ms | — |
| tokens/round | 5.02 | 1.00 | — |

From which:

- **The marginal cost of a drafted token is 2.1 ms** ((42.1−27.5)/7).
  Extra drafts are cheap; what is expensive is drafting them. So
  *acceptance per round* is the lever, not draft depth or attention.
- **The top-k + selector lattice costs 4.0 ms** (14.20 − 10.22). The other
  10.2 ms is the drafter's 5 layers plus the shared output head, against a
  ~3 ms bandwidth floor for the 2.76 GB they read.
- **A 1-token forward costs 27.5 ms** where llama.cpp does the same work in
  ~18.8 ms on this box (53.3 tok/s greedy). We are ~46% slower per forward,
  and that — not communication, not attention, not the scheduler — is the
  largest remaining single-stream gap.

### Three hypotheses that measurement killed

1. **"Merge the inject and draft graphs."** Estimated at 5–8 ms. The split
   says `inject 0.73 ms`. Merging saves the inject launch, ~0.6 ms — not
   worth a refactor of unsafe graph-building code.
2. **"Attention over a long KV cache is a meaningful cost."** 16K and 98K
   decode identically (verify 40.5 vs 40.7 ms; code 91.0 vs 90.1 tok/s).
   Configured context is nearly free and occupied context nearly so.
3. **"Tensor-parallel all-reduce is the ~20 ms gap."** 130 reductions per
   forward, host-bounced because GeForce P2P is driver-disabled — a good
   story, and wrong. `GGML_CUDA_ALLREDUCE=internal` gives 40.7 ms (and
   changes the output: different reduction order, different rounding).
   `GGML_CUDA_P2P=1` gives 42.5 ms with a byte-identical hash, i.e. the
   driver ignored it. And `=none`, which skips reductions entirely and
   produces provably wrong output, measures **47.5 ms — slower than
   baseline**. Removing all communication saves nothing, so communication
   was never the cost.

## Where a decode round's time actually goes

Measured with `CODPIECE_BATCH_TRACE=1` / `CODPIECE_DECODE_TRACE=1`.

At 32-way concurrency, per round:

| phase | before | after |
|---|---|---|
| admission / prefill stalls | 0.00 ms | 0.00 ms |
| input fill (mask, positions) | 0.64 ms | 0.61 ms |
| compute | 61.9 ms | 64.3 ms |
| logits readback | 9.96 ms | **0.01 ms** |
| host work (detok, stops, sends) | 0.02 ms | 0.01 ms |
| **round** | **72.5 ms** | **64.9 ms** |
| aggregate | 268.5 tok/s | **285.7 tok/s** |

Two findings, one of which cost a planned task:

1. **There was no scheduler overhead to reclaim.** The batch scheduler
   was assumed to be burning 25%+ in admission serialization and
   per-round host work, on the strength of a ~119 ms round measured at
   the HTTP layer. Instrumented, the round is 72.5 ms and host work is
   0.02 ms of it. The gap was client-side measurement, not server-side
   waste.
2. **At width 32 the round is compute bound, not memory bound** — 64.3 ms
   of compute against a 15.6 ms weight-read floor (4.1x). That kills
   batched speculation as an aggregate lever: speculation pays only while
   a round is memory bound and extra tokens ride along free, and at this
   width every drafted token costs proportional compute while only about
   half are accepted. It would still pay at low concurrency, which is
   where a single user actually lives.

The one real win was removing the readback: greedy rounds now argmax in
the graph and return one token id per lane instead of a full vocabulary
of logits per lane (~32 MB at width 32).

Found and fixed along the way: the stateless forward path had never run
under tensor parallelism — its zero-state inputs live in compute buffers
the meta backend cannot classify (`handle_gated_delta_net` asserts).
`codpiece ppl` now uses a session reset per chunk: identical math,
correctly classified cache tensors.
