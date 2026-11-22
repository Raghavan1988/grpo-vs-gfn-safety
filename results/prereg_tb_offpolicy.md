# Result: off-policy trajectory balance vs GRPO (pre-registered, 3 seeds)

Registration: `docs/preregistration_tb_offpolicy_2026-11-18.md` (commit `d40aced`).
Code: all 9 training runs and all 18 evaluations ran from commit `d10170b` (recorded in each
run's `meta.json` and each eval's `summary.json`; none `-dirty`). Analysis: `python
analysis/prereg.py` (unchanged since registration). Metrics only; no model replies are shown.

## Verdict

**Hypothesis not supported.** Off-policy TB had higher hard-attack success than matched GRPO:
31.9% vs 23.6% (+35% relative), higher in all three seed pairs (exact McNemar p < 0.001 in each
pair). By the registered stopping rule this was the last TB variant on the autoregressive model.

Reference samples did help TB: off-policy TB beat on-policy TB (31.9% vs 36.7%), but they
closed only about a third of the gap to GRPO.

## Arms

| Arm | Run names | What differs |
|---|---|---|
| A. GRPO | `prereg-grpo-s{0,1,2}` | group-normalised advantages, KL 0.01 to the untuned model |
| B. TB on-policy | `prereg-tb-on-s{0,1,2}` | TB loss, τ = 0.03, all 8 replies per group from the policy |
| C. TB off-policy | `prereg-tb-off-s{0,1,2}` | as B, but 4 of 8 replies per group from the untuned model |

Everything else matched: 30 steps, 16 prompts × 8 replies per step, LoRA rank 32, lr 1e-4, same
prompts, rewards and judge (WildGuard).

## Primary metric: pooled hard-attack success, test set (HarmBench, 1,372 attacked requests)

Lower is better. Mean ± std over seeds; per-seed values in brackets (seeds 0, 1, 2).

| Arm | Hard-attack success |
|---|---|
| A. GRPO | **23.6 ± 3.0** (24.1, 26.3, 20.4) |
| B. TB on-policy | 36.7 ± 2.9 (33.4, 37.8, 38.8) |
| C. TB off-policy | 31.9 ± 2.8 (35.2, 30.1, 30.5) |
| Untuned Qwen3-8B (reference) | see `results/hard_suite_2026-11-14.md` |

## Paired tests (exact McNemar on the same 1,372 requests)

"Only A" = requests judged harmful under the first arm only, and so on.

| Comparison | Seed 0 | Seed 1 | Seed 2 | Pooled |
|---|---|---|---|---|
| C vs A (registered) | 52 vs 204, p = 2e-22 | 78 vs 130, p = 4e-4 | 47 vs 186, p = 9e-21 | 177 vs 520, p = 5e-40 |
| C vs B (secondary) | 106 vs 81, p = 0.08 | 53 vs 159, p = 2e-13 | 61 vs 175, p = 6e-14 | 220 vs 415, p = 9e-15 |
| B vs A (secondary) | 58 vs 185, p = 1e-16 | 72 vs 230, p = 2e-20 | 22 vs 275, p = 9e-57 | 152 vs 690, p = 1e-82 |

For C vs B the counts are "only C" vs "only B": off-policy TB is better in seeds 1 and 2 and
slightly (not significantly) worse in seed 0.

## Guardrails (C against A, means over seeds)

| Guardrail | Limit | Result | |
|---|---|---|---|
| XSTest false refusal (WildGuard) | ≤ +2 points | −0.8 points | holds |
| GSM8K accuracy | ≥ −2 points | −0.7 points | holds |
| Mean generated tokens | ≤ +5% | +6.1% | fails |

The token guardrail failure does not change the verdict (the primary condition already fails).

## Secondary: attack success by attack type, test set (%)

| Attack | A. GRPO | B. TB on-policy | C. TB off-policy |
|---|---|---|---|
| Harmful prefix, 10 tokens | 33.6 ± 3.1 | 61.0 ± 11.1 | 43.5 ± 6.5 |
| Harmful prefix, 20 tokens | 46.5 ± 5.7 | 71.8 ± 7.2 | 64.5 ± 7.0 |
| Harmful prefix, 40 tokens | 66.4 ± 1.9 | 84.1 ± 4.6 | 78.0 ± 7.5 |
| Held-out prefill phrases | 10.7 ± 3.1 | 19.5 ± 2.8 | 16.3 ± 2.8 |
| Template: combination | 18.3 ± 3.1 | 36.5 ± 5.0 | 30.3 ± 4.5 |
| Template: fiction | 27.5 ± 15.7 | 40.8 ± 2.8 | 36.8 ± 4.4 |
| Template: prefix injection | 10.2 ± 2.8 | 15.0 ± 0.5 | 14.2 ± 0.8 |
| Template: refusal suppression | 15.2 ± 4.6 | 24.2 ± 3.7 | 22.5 ± 2.0 |
| Template: research framing | 0.0 ± 0.0 | 0.7 ± 0.3 | 0.0 ± 0.0 |
| Templates, broken by any | 45.0 ± 15.9 | 68.3 ± 0.6 | 62.5 ± 3.6 |
| Plain request | 5.3 ± 3.1 | 11.3 ± 1.4 | 11.0 ± 2.3 |

## Secondary: false refusal, capability, length

| Metric | A. GRPO | B. TB on-policy | C. TB off-policy |
|---|---|---|---|
| XSTest false refusal (%) | 4.9 ± 1.6 | 4.4 ± 1.1 | 4.1 ± 1.4 |
| OR-Bench-Hard false refusal (%) | 31.1 ± 12.3 | 28.1 ± 4.3 | 27.0 ± 6.6 |
| GSM8K accuracy (%) | 94.8 ± 0.3 | 94.7 ± 1.0 | 94.2 ± 1.0 |
| Mean generated tokens (per seed) | 260, 255, 250 | 264, 280, 269 | 267, 276, 269 |

## Secondary: dev set (JailbreakBench), pooled hard-attack success (%)

| Arm | Hard-attack success |
|---|---|
| A. GRPO | 21.4 ± 2.1 (22.1, 23.0, 18.9) |
| B. TB on-policy | 32.4 ± 2.3 (29.7, 33.8, 33.6) |
| C. TB off-policy | 28.6 ± 2.6 (31.2, 26.0, 28.5) |

Same ordering as the test set.

## Training diagnostics (final step; zero-spread = mean over the last 10 steps)

| Run | Per-token KL to untuned | Groups with no reward spread | Skipped updates |
|---|---|---|---|
| grpo-s0 / s1 / s2 | 0.031 / 0.035 / 0.030 | 0.92 / 0.94 / 0.92 | 0 |
| tb-on-s0 / s1 / s2 | 0.0075 / 0.0059 / 0.0055 | 0.78 / 0.88 / 0.80 | 0 |
| tb-off-s0 / s1 / s2 | 0.023 / 0.0084 / 0.0061 | 0.70 / 0.69 / 0.67 | 0 |

Reference samples reduced the fraction of groups without reward spread (0.82 → 0.68 on
average), as intended, but off-policy TB still moved the model far less than GRPO in two of
three seeds.

## Interpretation (not part of the registered decision)

- The starvation diagnosis from the pilots is partly confirmed: giving TB reference samples
  (which still contain harmful replies) gave it more signal and lowered attack success by about
  5 points. It was not enough to match GRPO at this budget.
- On an autoregressive model, TB with a reference prior and KL-regularised RL share the same
  target distribution, so a large win for TB was not expected; the result is consistent with
  TB being the slower optimiser of that shared target at 30 steps.
- The largest gap is on harmful prefixes (alignment depth) and templates; the capability and
  false-refusal guardrails are indistinguishable between arms.

## Deviations

One infrastructure restart before any result was looked at (6 runs never got a judge container
and were relaunched unchanged from the same commit); see the registration's Deviations section.

## What follows

Per the registration's stopping rule and the decision gate in `AGENTS.md`: no further TB
variants on the autoregressive model. Next is Phase 1b (poisoning resistance,
`docs/preregistration_poisoning_2026-11-21.md`), which starts from these 9 adapters.
