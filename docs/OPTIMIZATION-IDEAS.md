# Lossless optimization ideas, ranked by what measurement supports

*Every entry states the mechanism, the expected size, and — where it exists —
the measurement that justifies or kills it. Ideas without a number attached
are hypotheses, and this project's record is that most hypotheses die.*

## The budget we are spending against

| | |
|---|---|
| weights per GPU per token (27B Q8, split 2 ways) | 14.65 GiB = 15.72 GB |
| 3090 memory bandwidth | 936 GB/s |
| **hard floor per token** | **16.8 ms → 59.5 tok/s** |
| measured plain decode | 25.6 ms → 39 tok/s |
| **unaccounted overhead** | **~9 ms/token (35 %)** |

(An earlier version of this table said 15.6 ms / 64 tok/s, from dividing GiB by
a GB/s figure. 14.65 GiB is 15.72 GB, so the floor is 16.8 ms and decode runs at
65% of it, not 60%.)

That overhead now has a name and a count. Under tensor parallelism the meta
backend cuts the graph at every AllReduce and issues each piece as its own
per-device launch: a plain decode step is **129 subgraphs across 2 devices**
(two per layer, as tensor parallelism requires — one after the attention
output projection, one after the FFN down projection). 10 ms over 129
boundaries is ~80 us each, which is what a PCIe round trip costs without
NVLink. It is the tax ENGINE.md predicted, it is not a bug, and it is the
largest remaining pool — but it is not reducible by tuning, only by needing
fewer reductions.

Speculation changes the arithmetic: a verify pass reads the weights once
regardless of how many tokens it checks, so throughput is
`accepted_tokens / round`, and everything that is not "one weight-read"
is waste to be removed.

## The cost curve that reframes everything (27B, tensor parallel, measured)

`tandem stepcost` times steady-state steps as a function of tokens carried:

| tokens/step | ms/step | ms/token | vs T=1 |
|---|---|---|---|
| 1 | 26.02 | 26.02 | 1.00× |
| 2 | 28.12 | 14.06 | 1.08× |
| 4 | 31.36 | 7.84 | 1.21× |
| 8 | 40.32 | 5.04 | 1.55× |
| 16 | 48.10 | 3.01 | 1.85× |

**Sixteen tokens cost 1.85× one token.** Verification really is close to free
per token, exactly as the bandwidth argument predicts — which means the whole
optimization problem is *accepted tokens per round*, and draft cost is the
only thing standing in the way.

It also locates the cost precisely. A bare T=4 step costs 31.4 ms, but a
measured spec-3 round costs 51.8 ms — so **~20 ms of that round is the three
separate draft executions**, about 6.7 ms each. At depth 1: round 37.9 ms
against a 28.1 ms T=2 step, so ~9.8 ms for one draft.

Two consequences:

1. **Draft cost dominates, and it is not bandwidth.** A draft's actual weight
   traffic is ~1.7 GiB per GPU (≈1.8 ms). It measures 6.7–9.8 ms.
2. **Deeper verification is cheap; deeper *drafting* is not.** If drafts were
   free, spec-3's 2.86 accepted tokens over a 31.4 ms verify would be
   **91 tok/s**. That gap — 55 measured vs 91 available — is the prize.

---

## 1. Fuse the draft head into the verify graph — IMPLEMENTED

A draft from a *one-layer* head measured ~8 ms against ~1.4 ms of bandwidth,
so the cost is graph construction and allocation, paid K+1 times per round.
Appending the draft head to the verify graph makes it one execution.

Possible because the draft's input is available in-graph: `ggml_argmax` emits
I32 and `ggml_get_rows` consumes I32, so `embed(argmax(logits))` needs no host
round-trip. The tail emits a draft per verified position — the one to use for
each possible accept count — so the host simply picks.

**Result: neutral on the 27B** — 49.56 tok/s fused vs 49.62 separate. That is
a useful negative: it proves the draft was never the cost. Comparing round
times located the real one instead (see below).

Fusion is still the right shape, because it is what makes idea 4 possible:
one graph per round has nothing to interleave with.

## 2. Copy one recurrent snapshot instead of K — NOT WORTH IT

Arithmetic, now that it has been done properly: one GDN state slot is
128×128×48×4 B = 3.1 MB, ×48 layers = 151 MB. Writing four slots instead of
one costs 453 MB per round — 0.48 ms at 936 GB/s, against a ~52 ms round.
**Under 1 %.**

An earlier note in this file blamed this for the CPU oracle's 7 % loss. That
was wrong by an order of magnitude, and the correction matters more than the
idea: extra draft slots are cheap on the *snapshot* axis, and the real cost of
widening a round is the draft executions (see the cost curve above).

Original sketch, kept only because it is still correct as a mechanism:

Rollback needs the state at the accepted position, and we do not know that
position until after the graph runs, so today every round copies `K` snapshot
slots per GDN layer: ~300 MB of device traffic per round for two extra slots
across 48 layers. That is why extra draft slots are *not* free on this hybrid
model, which is what killed the CPU-oracle idea.

The fix is to split the round: run the trunk graph (the fused GDN op already
emits its K snapshots into the compute buffer), decide the accept count on the
host, then run a tiny second graph that copies exactly the one needed slot.
The compute buffer is still live between the two, since no other graph runs in
between.

Expected: removes ~4 % of per-token bandwidth at K=4 and, more importantly,
makes wide speculation cheap — which would justify revisiting free drafts.

## 3. Chain the drafts inside the graph — IMPLEMENTED

Idea 1 gives depth-1 drafting for free. Depth > 1 currently needs sequential
executions again. But the same `argmax → get_rows` trick chains: draft 2 can
consume draft 1's own output *within one graph*. A depth-3 chain becomes ~120
extra nodes on one launch rather than three launches.

**Result: it works.** `--depth K` chains K drafts in one graph. Measured
tokens/round 1.88 / 2.51 / 2.91 at depth 1/2/3, all lossless. Combined with
idea 4 it is what put tandem level with llama.cpp at depth 2 and ahead at
depth 3.

The chain does not make drafts free — a chain step still costs ~5.4 ms against
~1.6 ms of bandwidth — but it removes the per-draft graph construction, which
is what made depth > 1 not worth paying for before.

## 1b. The actual per-round cost: rebuilding the trunk graph — PARTLY FIXED

Round times told the story the throughput numbers hid:

| | round time |
|---|---|
| plain decode (replays a cached graph) | 26.3 ms |
| speculative round (rebuilds the 64-layer trunk graph) | 37.9 ms |

~11 ms per round, paid once regardless of draft depth, purely to construct
and allocate a graph whose shape never changes. Caching it is worth more than
anything about drafting.

Implemented, and it works where it can be verified: **CPU fused+cached runs
52.5–53.5 tok/s vs 47.2 plain**, lossless — and the CPU is where graph
rebuilding is *cheap*, so this understates the GPU effect.

Under tensor parallelism this was blocked until 2026-08-20; see idea 4. The
~11 ms estimate above turned out to be two costs stacked: ~4.4 ms of Rust-side
graph construction, and the backend's own per-device rebuild, which was being
paid on *every* compute because the graphs carried no identity.

## 4. Cache the verify graph per shape — FIXED (2026-08-20)

This was recorded as "under tensor parallelism the meta backend supports
exactly one repeatedly-replayed graph per session". That rule fit every
observation, but it was a lifetime bug, not a property of the design. Three
defects had to be fixed together — see `notes/results-2026-08-19.md` for the
diagnosis of each:

1. **ggml**: the meta backend maps meta tensors to per-device tensors in a map
   keyed by raw pointer, held in containers that were assigned at *allocation*
   time and evicted based on which graph was being *computed*. Those orders
   agree only while one graph is replayed forever. Containers are now keyed by
   graph uid and reclaimed least-recently-used.
2. **tandem**: `rollback_recurrent` built its graph in a context it freed each
   round, so the heap handed the same addresses to different tensors and the
   map aliased them. Rollback graphs are now built once per rollback distance
   and kept for the session; superseded cached graphs are retired rather than
   freed.
3. **ggml + tandem**: `ggml_new_graph_custom` leaves `cgraph->uid` at 0, which
   the meta backend reads as "assume this graph changed" — so **every compute
   rebuilt the per-device mapping, including plain decode's supposedly cached
   graph**. A new `ggml_graph_set_new_uid()` stamps a graph whose structure is
   frozen, and tandem stamps each cached graph after allocating it. Plain
   decode went from 20 meta rebuilds per 20 tokens to 2.

Item 3 is the one worth remembering: tandem had been paying for graph caching
in the control plane and getting none of it in the backend.

Measured on the 27B, 160 tokens, all output identical to plain greedy:

| configuration | rebuilt per round | cached | gain |
|---|---|---|---|
| plain decode | — | 39.0 tok/s | — |
| fused depth 1 | 47.2 | **57.7** | +22% |
| fused depth 2 | 53.6 | **64.3** | +20% |
| fused depth 3 | 53.6 | **62.9** | +17% |

## 6. Choose the chain depth per round — IMPLEMENTED, opt-in, parity with fixed depth 3

Depth matters: between depth 2 and depth 3 there is 14% on an arithmetic prompt
and 4% on code, in opposite directions from prose. `--depth auto` maximises
`E(K)/T(K)` per round, from the per-position acceptance a round already reports
and a measured round time, and it reliably lands on the right depth without
being told. Two runs, four prompts:

| prompt | d1 | d2 | d3 | **adaptive** |
|---|---|---|---|---|
| sky (128 tok) | 58.9 | 66.3 | 66.3 | 66.0 |
| code (128 tok) | 60.7 | 64.7 | 67.3 | **67.5** |
| primes (128 tok) | 62.7 | 77.1 | 87.6 | 87.6 |
| mixed prose+code+table (320 tok) | 58.6 | **67.9** | 66.0 | 65.9 |

**It matches fixed depth 3 within run-to-run noise (+-0.4%) and does not beat
it.** The value is not needing to pick, not extra speed: it cannot beat the best
fixed depth because it has to spend rounds discovering which one that is, and
the peak of the curve is where it spends most of them anyway.

What the work did produce is a real defect fix and two measurements worth
keeping — see `notes/results-2026-08-19.md`:

- **Acceptance rises along the chain**, it does not decay. On a code prompt the
  third draft is accepted 100% of the time versus 55% for the second, because
  position 3 is only reached when 1 and 2 were accepted and that selects for a
  predictable stretch. A decaying prior mis-prices exactly the case where depth
  pays; fixing it plus an optimism bonus took the code prompt 63.8 -> 67.8.
- **The per-step cost does not vary with context** (5.6-7.0 ms across 128, 256
  and 512-token generations). An earlier claim here that it doubled is
  withdrawn — it came from applying one run's acceptance profile to another
  run's throughput.
- Two things measured and discarded: faster forgetting (worse on 3 of 4
  prompts — noise, not tracking) and forcing a calibration visit to the
  shallower depth (**-8%**).

## 5. Overlap the draft with the trunk on a second CUDA stream — SPECULATIVE

The draft head is tiny and the trunk is huge; on one device they could overlap
if issued on independent streams. ggml drives one stream per backend, so this
needs work below tandem. Idea 1 achieves most of the same benefit without it.

---

## 7. Lossless weight compression, decompressed on the way in — MEASURED, NOT WORTH IT

The idea (store weights compressed, reconstruct in-kernel so fewer bytes cross the
bus) is real and published — DFloat11 and NeuZip do exactly this, bit-exact, by
entropy-coding the exponent field of BF16. The question is only how much
redundancy this particular file has, which is measurable rather than arguable:
entropy is the floor for any lossless coder, so it bounds every scheme in the
family at once. `scripts/weight-entropy.py` samples the GGUF:

| stored as | share of file | entropy | best-case saving |
|---|---|---|---|
| Q8_0 | 83.6% (24.49 GiB) | 7.67 bits/byte | 4.2% |
| BF16 | 16.4% (4.79 GiB) | 6.21 bits/byte | **22.4%** |

The BF16 tensors behave exactly as the papers describe — skewed exponents, ~22%
recoverable. But they are a sixth of the file, and the Q8_0 bulk is already
near-incompressible: quantisation has *already* spent the redundancy. Weighted,
the ceiling is **7.2% of weight bytes**, which is 1.2 ms of a 26 ms decode step
(**4.6%**) — and only if decompression were free, when in fact the kernel would
have to sustain over 936 GB/s of output to avoid becoming the new bottleneck.

Not worth building here. It would be worth revisiting for a BF16 or F16 model,
where the whole file is the compressible part.

## Tried and refuted (kept so they are not retried)

| idea | result |
|---|---|
| **CPU context oracle** (free n-gram drafts) | 0.75–0.80 acceptance on raw text continuation, but **costs** 3–4 tok/s on the 27B: acceptance collapses on chat-style output (0.06–0.23) and extra slots are not free because of idea 2's snapshot cost. Kept, defaults off. |
| **Confidence-gated drafting** (`p-min`) | Moves acceptance 0.67 → 0.86, throughput flat at 54–58. The gate is not where the cost is. |
| **Replicated LM head to cut PCIe** | Removes ~15 MB/round of readback; throughput moved less than run-to-run noise. Kept anyway — it is what makes in-graph sampling (and therefore idea 1) possible under tensor parallelism. |
| **Tree speculation** | **Architecturally impossible here.** 48 of 64 layers are Gated DeltaNet recurrences that advance along one sequence; a branching candidate set has no sequence for them to run on. Not a tuning problem — off the roadmap permanently. |
| **Candidate-restricted draft vocabulary** | The mechanism works and is lossless — verification always uses the full vocabulary, so a shortlist miss costs only that draft. It does make rounds cheaper: 43.6 → ~39 ms at depth 2, confirming the draft LM projection is ~2.2 ms of each ~5.4 ms chain step. But acceptance collapses with it: 0.712 full vocab → 0.435 (static low-4096), 0.399 (context+low 4096), 0.430 (context+low 16384), so throughput falls 55.5 → 46. Break-even needs acceptance ≥ 0.585. **Note what the size sweep says: 8× more candidates (2048 → 16384) bought only 0.37 → 0.43, so the true argmax sits outside even a 16K shortlist most of the time — low token ids are a bad frequency proxy in this 248K multilingual vocabulary.** Worth one more attempt only with a shortlist ranked by measured token frequency; the code is in place behind `--cand N`, defaulting off. |
