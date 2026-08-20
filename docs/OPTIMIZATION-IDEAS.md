# Lossless optimization ideas, ranked by what measurement supports

*Every entry states the mechanism, the expected size, and — where it exists —
the measurement that justifies or kills it. Ideas without a number attached
are hypotheses, and this project's record is that most hypotheses die.*

## The budget we are spending against

| | |
|---|---|
| weights per GPU per token (27B Q8, split 2 ways) | 14.65 GiB |
| 3090 memory bandwidth | 936 GB/s |
| **hard floor per token** | **15.6 ms → 64 tok/s** |
| measured plain decode | 25 ms → 40 tok/s |
| **unaccounted overhead** | **~10 ms/token (38 %)** |

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

## 3. Chain the drafts inside the graph — NOT YET DONE

Idea 1 gives depth-1 drafting for free. Depth > 1 currently needs sequential
executions again. But the same `argmax → get_rows` trick chains: draft 2 can
consume draft 1's own output *within one graph*. A depth-3 chain becomes ~120
extra nodes on one launch rather than three launches.

Expected: the tok/round of depth-3 (3.0 measured) at the round cost of
depth-1. That is the single largest remaining item if idea 1 lands.

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

Under tensor parallelism it is blocked (idea 4).

## 4. Cache the verify graph per shape — BLOCKED under TP

Six configurations were tried on the 27B. All failures are the same assert,
`bcj.nodes[i]` null at ggml-backend-meta.cpp:1838 — the meta backend cannot
produce a per-device tensor for some node of a graph it is replaying.

| configuration | tokens/step | cached | result |
|---|---|---|---|
| plain decode | 1 | yes | **works** — 38 tok/s, and gets CUDA-graph capture |
| speculative verify | 2–4 | no | **works** — 49–55 tok/s |
| speculative verify | 2–4 | yes | fails |
| fused round (verify + draft tail) | 2 | no | **works** — 49.6 tok/s |
| fused round | 2 | yes | fails |
| draft head alone | 1 | yes | fails |
| verify **and** draft both cached | 2–4 + 1 | yes | fails |

The last row is the one that kills the obvious workaround: it is not that
cached and rebuilt graphs cannot mix. Two cached graphs alternating fail too.

**The rule that fits every observation:** under tensor parallelism the meta
backend supports exactly *one* repeatedly-replayed graph per session. Plain
decode qualifies — it replays one graph forever — which is precisely why the
only path that benefits from graph caching today is the one that does not
speculate. A speculative round inherently needs either two recurring graphs,
or one graph whose embedding lookup is indexed by a computed node (the
in-graph argmax), and both are outside what the backend supports on replay.

Closing this is worth ~11 ms/round (37.9 → ~26 ms), i.e. roughly 49.6 → 70
tok/s at depth 1. It needs work inside `ggml-backend-meta.cpp` rather than in
tandem: the per-device tensor mapping is rebuilt per graph and does not
survive another graph claiming the buffers. `TANDEM_BOTH_CACHED=1` and
`TANDEM_MTP_CACHE=1` reproduce the two failing shapes directly.

## 5. Overlap the draft with the trunk on a second CUDA stream — SPECULATIVE

The draft head is tiny and the trunk is huge; on one device they could overlap
if issued on independent streams. ggml drives one stream per backend, so this
needs work below tandem. Idea 1 achieves most of the same benefit without it.

---

## Tried and refuted (kept so they are not retried)

| idea | result |
|---|---|
| **CPU context oracle** (free n-gram drafts) | 0.75–0.80 acceptance on raw text continuation, but **costs** 3–4 tok/s on the 27B: acceptance collapses on chat-style output (0.06–0.23) and extra slots are not free because of idea 2's snapshot cost. Kept, defaults off. |
| **Confidence-gated drafting** (`p-min`) | Moves acceptance 0.67 → 0.86, throughput flat at 54–58. The gate is not where the cost is. |
| **Replicated LM head to cut PCIe** | Removes ~15 MB/round of readback; throughput moved less than run-to-run noise. Kept anyway — it is what makes in-graph sampling (and therefore idea 1) possible under tensor parallelism. |
| **Tree speculation** | **Architecturally impossible here.** 48 of 64 layers are Gated DeltaNet recurrences that advance along one sequence; a branching candidate set has no sequence for them to run on. Not a tuning problem — off the roadmap permanently. |
