# Pre-registration: SubTB with tempered sampling, log reward and 10-token states vs GRPO (written 26 Nov 2026)

**Status: final, committed before any registered arm E run.** Training code: commit `cf06026`
(options added in `a4e60a9`, `dd43a2d`). The smoke run (§6) ran from `cf06026` before this
status change; it checks the pipeline only and is not part of the result.

## 1. Background

- Registered SubTB (arm D, 64-token states, nothing else from the paper) did not beat GRPO and
  matched TB (`results/prereg_subtb.md`: GRPO 23.6 ± 3.0%, TB 31.9 ± 2.8%, SubTB 32.5 ± 3.8%).
- That arm left out the parts of the Hu et al. (2024) recipe the PI stressed besides the loss:
  sampling from a tempered policy and exploration, and log rewards. Arm E adds them, plus finer
  states. **Three changes at once**: a difference from D cannot be attributed to one of them.
  This is a test of "the closer-to-paper recipe", decided by Raghavan on 26 Nov 2026.

## 2. Changes from arm D (everything else identical)

1. **Tempered sampling.** Each step, with probability 2/3 the policy's 4 replies per group are
   sampled at a temperature drawn uniformly from [0.7, 1.3] (seeded per step); otherwise at 1.0.
   Paper: probability 2/3, range [0.5, 2.0]; the range is narrowed because 300-token replies at
   T = 2 are mostly noise. The 4 reference-model replies stay at T = 1.0. GRPO cannot use
   tempered samples without importance weights; the comparison includes this GFlowNet advantage.
2. **Log reward.** The reward term in log R is 7 · log(max(r, 0.01)) instead of r / 0.03, with
   r as before (1 − P(harmful), 1 − P(refusal), GSM8K 0/1). Target π ∝ π_ref · max(r, 0.01)^7.
   β = 7 and floor 0.01 give a maximum range of 32 nats, matching the current 1/τ = 33.
3. **10-token states.** SubTB states every 10 generated tokens (after the prefill) instead of 64.
   Known risk: WildGuard is unreliable on short prefixes (`results/wg_prefix_check.md`: 14% of
   benign and 58% of recovered prefill replies flagged at 16 tokens), which may push toward
   over-refusal; watched by the XSTest guardrail.

Command (S = 0, 1, 2):
```
modal run train/modal_train.py --pipeline subtb --tau 0.03 --ref-samples 4 --chunk 10 \
  --temp-prob 0.6667 --temp-low 0.7 --temp-high 1.3 --log-reward --log-floor 0.01 --log-beta 7 \
  --seed S --run-name prereg-subtbx-sS
```

## 3. Arms

A (GRPO), C (TB off-policy), D (SubTB) reused; E new (3 seeds).

## 4. Evaluation, outcomes, decision rule

Identical to `docs/preregistration_subtb_2026-11-26.md` §4–5, with E in place of D: primary
pooled hard-attack success E vs A, same guardrails, same ±10% relative rule, McNemar per seed and
pooled. Secondary: E vs D (do the three additions help SubTB?), per-attack tables, dev set,
diagnostics including `zero_var_group_frac`, `sample_temp`, `subtb_states_mean`, `t_judge`.

## 5. Stopping rule and allowed deviations

As in the previous registration. No setting is tuned on these results.

## 6. Smoke run

3 steps, `--run-name smoke-subtbx`. Passes if: no crash, finite gradients, `subtb_states_mean`
about 20–35, step time under 3 minutes.

**Result (26 Nov 2026): passed.** No skipped updates (gradient norm 83–132 before clipping),
25–28 states per reply, step time 106–111 s (judging 31–36 s), sampling temperatures 1.21 / 1.0 /
1.03 for steps 1–3.

## 7. Cost

About 60–65 min per training run (from the smoke run) and 6 evals: about $45–60. Approved by
Raghavan on 26 Nov 2026.

## Deviations

(none yet)
