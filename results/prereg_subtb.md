# Pre-registered SubTB vs GRPO: results (2026-11-26)

Registration: `docs/preregistration_subtb_2026-11-26.md` (commit `d578eaa`; training code
`64e0550`). Arm D = off-policy TB with the loss switched to sub-trajectory balance, states every
64 generated tokens. Arms A (GRPO) and C (TB off-policy) reused from `results/prereg_tb_offpolicy.md`.

**Verdict (registered rule): hypothesis not supported.** SubTB attack success is 37.7% higher
(relative) than GRPO's and higher in every seed pair. SubTB is indistinguishable from TB.

## Primary: pooled hard-attack success, test set (1,372 attacked requests, HarmBench cls)

| Arm | s0 | s1 | s2 | mean ± std |
|---|---|---|---|---|
| A. GRPO | 24.1% | 26.3% | 20.4% | 23.6 ± 3.0% |
| C. TB off-policy | 35.2% | 30.1% | 30.5% | 31.9 ± 2.8% |
| D. SubTB off-policy | 36.8% | 31.0% | 29.7% | **32.5 ± 3.8%** |

Exact McNemar on paired requests (requests harmful under one arm only):

| Comparison | s0 | s1 | s2 | pooled |
|---|---|---|---|---|
| D vs A | 217 vs 43, p = 4e-29 | 140 vs 75, p = 1e-5 | 179 vs 52, p = 2e-17 | 536 vs 170, p = 5e-45 |
| D vs C (secondary) | 86 vs 64, p = 0.09 | 84 vs 71, p = 0.34 | 88 vs 100, p = 0.42 | 258 vs 235, p = 0.32 |

## Guardrails (D vs A, means over seeds): all hold

| Guardrail | A | D | Difference | Limit |
|---|---|---|---|---|
| XSTest false refusal (WildGuard) | 4.9% | 4.4% | −0.5 pts | +2 pts |
| GSM8K accuracy | 94.8% | 95.3% | +0.5 pts | −2 pts |
| Mean generated tokens | 255 | 262 | +2.8% | +5% |

## Reading

- Per-64-token credit did not change the outcome relative to TB (D ≈ C, no seed significant).
- The setup omitted parts of the Hu et al. recipe on purpose (tempered sampling, replay buffer,
  reward annealing, longer training, per-token states); the result is about SubTB as registered,
  not the full recipe. See `docs/subtb_reference_notes.md`.
- Training diagnostics: no skipped updates; 63–69% of groups still had no reward spread in the
  smoke run, the starvation problem SubTB was not designed to fix.

## Deviations
- The `prereg-subtb-s0` test eval crashed at vLLM start-up (Hugging Face connection reset, before
  any generation) and was rerun with the same command. Its `git_sha` is recorded as
  `d578eaa-dirty`: the only uncommitted tracked change at launch was the deviation note in the
  registration document; no code differed.
