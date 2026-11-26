# WildGuard on reply prefixes: feasibility check for SubTB (2026-11-26)

Exploratory, not pre-registered. Script: `analysis/wg_prefix_check.py`; metrics in
`outputs/wg_prefix_check/summary.json` (gitignored). Untuned Qwen3-8B replies from
`baseline_qwen3_8b_suite-hard`, cut to the first 16/32/64/128/256 Qwen tokens (cuts include any
forced prefill), scored by WildGuard. 716 replies, 3,797 WildGuard calls, one L40S (< $1).

Share of replies with P(harmful) > 0.5:

| Group (n) | 16 tok | 32 | 64 | 128 | 256 | full |
|---|---|---|---|---|---|---|
| A harmful, HarmBench yes (200) | 93.5% | 95.0% | 94.0% | 94.5% | 95.5% | 97.0% |
| B prefill attack, recovered (116) | 57.8% | 37.9% | 26.7% | 16.4% | 12.9% | 15.5% |
| C template attack, safe (100) | 26.0% | 23.0% | 20.0% | 18.0% | 19.0% | 20.0% |
| D benign: XSTest safe + OR-Bench-Hard (300) | 14.0% | 4.0% | 3.3% | 3.3% | 2.7% | 2.3% |

Mean P(refusal), group D: 0.29 / 0.31 / 0.25 / 0.19 / 0.18 / 0.18.
Replies whose label crosses 0.5 more than once: 1.7–4.5% in every group.

## Reading
- Harmful replies are flagged at the first cut (186/200 by 16 tokens). Part of this comes from
  the forced prefill inside the first 16 tokens, so it is an upper bound on how early the
  model's own tokens are detected.
- Recovered prefill replies are flagged on early prefixes (the compliant opener), then the score
  falls as the refusal appears; it reaches the full-length level by about 128 tokens. Under SubTB
  this gives negative credit to the opener and positive credit to the turn to refusal, which is
  the intended signal. The opener is a forced prefill, not trained tokens.
- The 15–20% "harmful" on full B/C replies is the known WildGuard / HarmBench disagreement, not
  a prefix effect.
- Benign replies: false harm flags and P(refusal) are inflated only at 16 tokens (and refusal up
  to 64); from 32 tokens on, harm flags are near the full-length rate.
- Scores are stable along a reply (few flips).

## Decision
Usable as a SubTB prefix reward, with the first cut at **≥ 32 tokens** (64 is safer for refusal
scores on benign prompts) and cuts every 32–64 tokens. Open: measure cuts from the end of the
prefill (model tokens only) when implementing.
