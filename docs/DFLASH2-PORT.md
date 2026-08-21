# DFlash2 drafter port

Reference: `~/llm/llama.cpp-dflash2` (PR-27342, commit 5ecbe1a), files
`src/models/dflash.cpp` (graph) and `common/speculative.cpp`
(`common_speculative_impl_draft_dflash`, the driver). Model:
`~/llm/models/qwen38/Qwen3.8-27B-DFlash2-{Q8_0,Q4_K_M}.gguf` (1.9B, arch
`dflash`). Measured prize on this box: mean accepted length 4.80 vs the MTP
head's 4.28.

## The model

- 5 Qwen3-style layers: RMS norms, GQA 32/8 heads x 128, q/k per-head RMS
  norm, rope NEOX freq_base 1e7, SwiGLU ffn 17408. NON-CAUSAL attention,
  sliding window 2048 (`is_swa` true for all 5 layers).
- Per-layer DFlash2 conv adapters, before/after both attn and ffn:
  `attn_conv_proj [5120, 2*kernel*groups]` produces dynamic per-token
  coefficients; `attn_conv_base [n_embd, kernel=2, 2]` is the static base.
  `build_dflash2_conv(hidden, dynamic, base, side)`: for tap in 0..kernel,
  weight = coeff(tap) (grouped, group_size 16, repeated to n_embd) +
  base[tap][side]; values = block-internal shift of hidden by `tap` positions
  (zeros shifted in at block start); result = sum(weight * values). side 0 =
  input conv (on the normed input), side 1 = output conv (on attn/ffn
  output). The shift is WITHIN the 8-token block only.
- No token embedding and no lm head of its own: it uses the TARGET's
  `token_embd` for the noise block and the TARGET's `output.weight` for
  logits (both mirrored in codpiece under TP — convenient).
- Encoder side: `fc [25600 -> 5120]` + `enc.output_norm` (RMS) projects the
  concatenated TRUNK layer-input hiddens of `target_layers = [6,20,34,48,62]`
  (5 x 5120) into draft hidden space.
- Selector (replaces logits sampling): `selector_hidden [5120 -> 256]`,
  `selector_prev/next [256, n_vocab]`. Per drafted position: top-16
  candidates of the logits; transition score(prev_cand i -> cand j) =
  unary_logit(j) + <selector_next[j], selector_prev[i] * hidden(pos)>.
  Packed per position into an n_embd row: [0..16) candidate ids as f32,
  [16..16+256) the 16x16 score matrix; position 1 conditions on the anchor
  token (its selector_prev row directly; scores repeated over predecessor
  slots).

## Per-round flow (reference driver)

1. After a verify round commits tokens, their trunk features are injected:
   embd batch through the decoder's inject branch — per layer,
   K = rope(rms(wk . h_enc)), V = wv . h_enc, written to the draft KV at the
   committed positions. `h_enc` = enc_norm(fc(concat features)).
2. Draft: batch `[id_last @ p, MASK x 7 @ p+1..p+7]` (mask id 248070),
   non-causal over the injected window + the block; graph emits the packed
   lattice; the host walks it: predecessor slot = 0 (anchor row uses
   repeated scores), at each position pick argmax (greedy) or a
   temperature draw over the 16 transition scores; the picked slot chains.
   7 drafts out. `p_min`-style truncation only via n_min.
3. Rejected drafts need NO cache undo: the draft cache is positional
   attention KV, overwritten when real tokens land at those positions —
   same rewind-by-position semantics as the trunk's attention KV.

## codpiece integration plan

1. **Trunk taps + fused injection (no host traffic).** build_inner captures
   `inp_l` at layers {6,20,34,48,62} (the residual entering the layer —
   MIRRORED under TP, since each layer ends in an all-reduce). Fold the
   ENTIRE injection into the trunk graphs (fused round, prefill chunks,
   embd/vision chunks): concat taps -> fc -> enc norm -> per-dflash-layer
   K/V + rope -> set_rows into the draft KV ring (2048 positions, ring by
   position % 2048, sliding-window mask by real positions). The draft KV
   tensors live in the Session beside the trunk caches. Rollback = nothing
   (positional overwrite).
2. **Draft graph** (separate small cached graph per shape, like mtp_draft):
   input [anchor_id, MASK x 7], positions p..p+7, non-causal mask over
   window(2048) + block; decoder forward with conv adapters; logits via the
   trunk's mirrored head; top_k(16) + lattice build exactly as the
   reference; readback = the packed lattice (8 x 5120 floats = 160KB).
   Under TP every op classifies: logits mirrored -> top_k per-row legal;
   selector tables mirrored (small); get_rows by ids legal on mirrored.
3. **Host walk** in the engine: greedy = argmax chain; sampled = the same
   walk with a temperature draw over 16 scores from a seed-derived RNG.
   Verification NEEDS NO draft distributions in codpiece: gumbel-coupled
   commits are exact target samples regardless of the draft process, and
   acceptance stays `draft == commit`. Greedy verification is unchanged
   (`draft == argmax`).
4. **Engine switch**: `--draft dflash:<path>` (or CODPIECE_DFLASH env)
   selects the drafter; the MTP fused chain stays the default until the
   A/B gate passes. With DFlash the fused round keeps verification +
   gumbel sampling but drops the MTP tail (chain depth 0); drafts come
   from the DFlash graph after each round (up to 7/round, n_min 2).
5. **Gates**: (a) CPU parity of the draft forward vs the reference build
   (same lattice bytes for a fixed feature/window fixture); (b) accepted-
   length and tok/s A/B vs MTP on db2 + greedy probes; (c) all existing
   parity gates unchanged; (d) llama.cpp head-to-head rerun.

## Cost model (why this should win)

Draft call: ~1.9 GB weight read (Q8, split across cards) + shared head over
8 positions ~= 4-6 ms for up to 7 drafts. MTP chain: ~6 ms per draft.
Injection rides the trunk graphs (its weights are the 5 layers' wk/wv only).
Verify width grows from <=3 to <=8 tokens/round — still one weight read.
At accepted ~4.8/round and round ~= 17 (verify) + 6 (draft) ms, decode
projects to ~200+ tok/s ceiling on predictable text and ~75-90 on prose,
which is exactly the scoreboard target.

## VRAM

Draft weights: Q8_0 1.9 GB split across cards (~0.95/card) or Q4_K_M
~1.0 GB (~0.5/card). Draft KV: 5 layers x 2048 window x 2 x 1024 x f16 =
40 MB. Feature path adds no persistent buffers (fused injection). Prod
headroom after vision is ~1.1 GB/card: Q4_K_M fits today; Q8 needs either
CODPIECE_SESSIONS=1 or the smaller quant — A/B both, quality first.


## Status — SHIPPED 2026-08-20 (opt-in via --dflash)

Engine-integrated and gated. Greedy requests draft by block diffusion;
sampled requests keep the gumbel-coupled chain (measured better: block
positions past ~3 compound to nothing at temperature while paying vocab
noise and wide verifies). The selector lattice is ALWAYS walked greedily —
coupled verification accepts draft x with probability p_target(x), so the
mode is optimal and the reference's temperature walk only decorrelates
(0.119 vs ~0.5 measured). Draft and inject graphs build once, replay
uid-stamped; meta graph pool 32.

Headline (192 tok, 2 reps, same window): prose 71.2 vs MTP 58.4, code
82.2 vs 62.5, arithmetic 117.4 vs 86.1 tok/s; accepted/call 2.84-5.57 vs
1.58-2.76. Text identity 2/3 IDENTICAL, 1/3 the known batched-argmax tie
class. Prod enablement pending a VRAM fit (Q8 needs 1.9 GB/card beside
vision + 2 sessions; Q4_K_M or CODPIECE_SESSIONS=1 are the candidates).
