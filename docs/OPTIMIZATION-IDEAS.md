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

---

## 1. Fuse the draft head into the verify graph — IMPLEMENTED

A draft from a *one-layer* head measured ~8 ms against ~1.4 ms of bandwidth,
so the cost is graph construction and allocation, paid K+1 times per round.
Appending the draft head to the verify graph makes it one execution.

Possible because the draft's input is available in-graph: `ggml_argmax` emits
I32 and `ggml_get_rows` consumes I32, so `embed(argmax(logits))` needs no host
round-trip. The tail emits a draft per verified position — the one to use for
each possible accept count — so the host simply picks.

Status: lossless on CPU (acceptance 0.750, 1.79 tok/round). 27B measurement in
progress.

## 2. Copy one recurrent snapshot instead of K — NOT YET DONE

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

## 4. Cache the verify graph per shape — BLOCKED

The T=1 decode graph is cached and replayed (worth +11 % via CUDA graph
capture). The verify graph has a fixed shape per draft depth and could be
cached the same way, but the tensor-parallel meta backend rejects a cached
draft graph (`bcj.nodes[i]` null, ggml-backend-meta.cpp:1838). `TANDEM_MTP_CACHE=1`
forces the path so the failure can be characterized: the open question is
whether it breaks on first use or only when alternating with the trunk's own
cached graph.

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
