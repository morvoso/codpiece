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

Found and fixed along the way: the stateless forward path had never run
under tensor parallelism — its zero-state inputs live in compute buffers
the meta backend cannot classify (`handle_gated_delta_net` asserts).
`codpiece ppl` now uses a session reset per chunk: identical math,
correctly classified cache tensors.
