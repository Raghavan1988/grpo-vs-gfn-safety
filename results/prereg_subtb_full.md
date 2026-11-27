# Pre-registered arm E (SubTB + tempered sampling + log reward + 10-token states) vs GRPO: results (2026-11-26)

Registration: `docs/preregistration_subtb_full_2026-11-26.md` (commit `ef058c4`; all three runs
trained and evaluated from `ef058c4`). Arms A (GRPO), C (TB off-policy) and D (SubTB, 64-token
states) reused from `results/prereg_tb_offpolicy.md` and `results/prereg_subtb.md`.

**Verdict (registered rule): hypothesis not supported.** Arm E attack success is 68% higher
(relative) than GRPO's and higher in every seed pair. It is also significantly worse than plain
SubTB (D) and TB (C), and it fails the generated-token guardrail.

## Primary: pooled hard-attack success, test set (1,372 attacked requests, HarmBench cls)

| Arm | s0 | s1 | s2 | mean ± std |
|---|---|---|---|---|
| A. GRPO | 24.1% | 26.3% | 20.4% | 23.6 ± 3.0% |
| C. TB off-policy | 35.2% | 30.1% | 30.5% | 31.9 ± 2.8% |
| D. SubTB (64-token states) | 36.8% | 31.0% | 29.7% | 32.5 ± 3.8% |
| E. SubTB + tempered + log reward + 10-token states | 40.5% | 38.0% | 40.6% | **39.7 ± 1.4%** |

Exact McNemar (requests harmful under the first arm only vs the second arm only):

| Comparison | s0 | s1 | s2 | pooled |
|---|---|---|---|---|
| E vs A | 265 vs 41, p = 3e-41 | 219 vs 58, p = 4e-23 | 300 vs 23, p = 1e-62 | 784 vs 122, p = 5e-119 |
| E vs D (secondary) | 119 vs 69, p = 3e-4 | 149 vs 53, p = 9e-12 | 204 vs 54, p = 1e-21 | 472 vs 176, p = 4e-32 |
| E vs C | 132 vs 60, p = 2e-7 | 164 vs 55, p = 9e-14 | 186 vs 48, p = 2e-20 | 482 vs 163, p = 2e-37 |

## Guardrails (E vs A, means over seeds)

| Guardrail | A | E | Difference | Limit | Holds |
|---|---|---|---|---|---|
| XSTest false refusal (WildGuard) | 4.9% | 4.0% | −0.9 pts | +2 pts | yes |
| GSM8K accuracy | 94.8% | 95.2% | +0.3 pts | −2 pts | yes |
| Mean generated tokens | 255 | 277 | +8.5% | +5% | **no** |

## Reading

- The three additions together made SubTB worse, not better. Because they were changed at once,
  which one is responsible is not identified (stated in the registration, §1).
- Candidate explanations, untested: the log reward is flat among safe replies (P(harmful) 0.001
  vs 0.05 differ by ~0.3 nats), weakening the push away from borderline replies; 10-token states
  feed WildGuard's unreliable short-prefix scores into the loss; tempered samples move the
  training distribution away from greedy evaluation; longer replies (token guardrail) are
  consistent with the stop-probability term affecting length.
- Over-refusal did not increase (XSTest −0.9 points), despite the short-prefix false alarms.

## Deviations
- The dev-set eval of `prereg-subtbx-s1` (`--dataset jbb`) failed at vLLM start-up ("Engine core
  initialization failed") before any generation and was rerun automatically with the same
  command; the failed log is kept. Test-set evals ran once each.
