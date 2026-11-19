# Pre-registration: off-policy trajectory balance vs GRPO (written 18 Nov 2026)

**Status: written before any off-policy TB run exists.** This document must be committed before
the first run below is launched; the commit timestamp is the registration. It fixes the one
change being tested, every setting, the metrics and the decision rule, so that the result cannot
be shaped by choices made after seeing it.

## 1. Background

At a matched budget, on-policy trajectory balance (TB, the GFlowNet objective) did not match GRPO
on Qwen3-8B (`docs/report_2026-11-16.md`, `results/experiments_2026-11-15.md`; one seed each).
Pooled over the hard attacks of the test set (harmful prefixes at 10 / 20 / 40 tokens and five
jailbreak templates; 1,372 attacked requests), attack success was:

| Model (seed 0) | Pooled hard-attack success |
|---|---|
| GRPO v3, 30 steps | 21.7% |
| TB τ = 0.03, 30 steps | 34.0% |
| TB τ = 0.03, 100 steps | 33.5% |

**Diagnosis from the training logs** (`outputs/train/tb-tau0.03-s0/log.jsonl`):

- 70–94% of groups have no reward spread on most steps, so most of each step carries no gradient.
- The TB loss does not fall: 31 at step 1, 31 at step 30. It is large whenever a harmful reply
  happens to be sampled and near zero otherwise.
- The policy barely moves: per-token KL to the untuned model stays near 0.007, against 0.03–0.05
  for GRPO.
- The harmful rate on prefilled training prompts is still around 10% late in training; under GRPO
  it reaches 0 by step 10.

So on-policy TB is not sitting at its target π_untuned · exp(r/τ). It stops receiving signal:
once the policy rarely samples a harmful reply, the loss cannot see that harmful replies are
still far too likely relative to the target.

## 2. The single change

**Off-policy TB: in every group of 8 replies, 4 are sampled from the current policy and 4 from
the untuned reference model.** Nothing else changes.

Why this is a legitimate use of the method, not a tuning trick:

- The TB condition holds for replies from any distribution with full support, with no importance
  weights. Training on reference samples is a standard GFlowNet capability.
- GRPO's policy gradient is only valid on the policy's own samples, so GRPO cannot use this.
  The second try therefore tests the one structural advantage TB has over GRPO.
- The untuned model still complies under prefill attacks, so mixed groups keep containing both
  safe and harmful replies throughout training.
- The budget is matched: same prompts, 8 replies per prompt, judge calls, gradient passes, LoRA,
  learning rate and steps as GRPO.

The 4 / 4 split is fixed here without tuning. It was not chosen on any data.

## 3. Arms and settings

Three arms, three seeds each (0, 1, 2), all on one git commit (recorded as `git_sha` in each
run's `meta.json`; it must not end in `-dirty`).

| Arm | Command (per seed S) |
|---|---|
| A. GRPO | `modal run train/modal_train.py --pipeline grpo --seed S --run-name prereg-grpo-sS` |
| B. TB on-policy | `modal run train/modal_train.py --pipeline tb --tau 0.03 --seed S --run-name prereg-tb-on-sS` |
| C. TB off-policy | `modal run train/modal_train.py --pipeline tb --tau 0.03 --ref-samples 4 --seed S --run-name prereg-tb-off-sS` |

Fixed for all arms (the `GRPOConfig` defaults): 30 steps; 16 prompts per step (8 harmful with a
training prefill, 2 harmful plain, 3 benign, 3 maths); 8 replies per prompt at temperature 1.0,
at most 512 tokens; LoRA rank 32 on all attention and MLP projections; lr 1e-4; gradient clip
1.0; reference = untuned model; frozen training split (`data/SPLITS.md`). GRPO: KL weight 0.01,
zero advantage for groups with reward std below 0.05. TB: τ = 0.03 (chosen earlier on the dev
set), batch log Z, no length normalisation.

Seed 0 of arms A and B is rerun rather than reused, because the training code has changed since
the pilots (judge de-duplication, length-sorted micro-batches, non-finite gradient guard).

Arm B is included so that any difference between C and A can be attributed: C vs B isolates the
effect of the reference samples.

## 4. Evaluation

For each of the 9 final adapters, greedy decoding, same harness as the pilots:

```
modal run baseline/modal_baseline.py --adapter RUN --suite hard                  # test set
modal run baseline/modal_baseline.py --adapter RUN --suite hard --dataset jbb    # dev set
```

Attack success is judged by the HarmBench classifier; refusals by WildGuard. WildGuard is the
training judge and is never used for attack success. No evaluation is run until all 9 training
runs have finished, and no setting is changed after the first training run is launched.

## 5. Outcomes

**Primary metric.** Pooled hard-attack success on the test set: the fraction of the 1,372 attacked
requests (`split == "hard"` in `harmbench_attacks.jsonl`: harmful prefix 10 / 20 / 40, n = 124
each, and 5 templates, n = 200 each) judged harmful. Reported as mean ± std over the 3 seeds.

**Guardrails** (arm C against arm A, means over seeds):
- XSTest false refusal (WildGuard) at most 2 points higher;
- GSM8K accuracy at most 2 points lower;
- mean generated tokens (`mean_tokens`) at most 5% higher.

**Decision rule** (C = off-policy TB, A = GRPO):

| Result | Condition | What we report |
|---|---|---|
| Hypothesis supported | mean(C) ≤ 0.90 × mean(A), C lower than A in all 3 seed pairs, and all guardrails hold | Off-policy TB gives better safety control than matched GRPO |
| No difference | mean(C) within ±10% (relative) of mean(A), or the 3 seed pairs disagree in sign | The two are equivalent at this budget (consistent with TB ≡ KL-regularised RL for AR models) |
| Hypothesis not supported | mean(C) ≥ 1.10 × mean(A) and C higher than A in all 3 seed pairs | Negative result: TB does not match GRPO, on-policy or off-policy |

If the primary condition holds but a guardrail fails, the result is reported as "lower attack
success at the cost of <the failed guardrail>", not as support for the hypothesis.

**Statistical test.** Exact McNemar on the 1,372 paired requests, C vs A, within each seed pair
(seed S against seed S), plus the pooled discordant counts over the three pairs.

**Secondary outcomes** (reported in full, not used for the decision):
- C vs B on the primary metric (does off-policy sampling help TB at all?);
- each attack separately: held-out prefill, prefix 10 / 20 / 40, templates (any, mean), plain;
- OR-Bench-Hard false refusal;
- the same metrics on the dev set (JailbreakBench);
- training diagnostics: `zero_var_group_frac`, `kl`, `tb_loss`, `harmful_rate_prefill`,
  `update_skipped`, `pad_frac`.

## 6. Stopping rule

**This is the last TB variant on the AR model.** Whatever the outcome, no further TB change
(τ, split, steps, learned log Z, replay buffer, new training data) is run and reported as part of
this comparison. A later idea is a new, separately registered experiment.

## 7. Allowed deviations

- A run interrupted by GPU preemption resumes from its last checkpoint (built-in, tested to match
  an uninterrupted run).
- A run that crashes for an infrastructure reason is restarted with the same seed and name suffix
  `-r1`; the crashed run is kept and listed.
- A run that stops with "consecutive updates skipped for non-finite gradients" is NOT restarted
  with different settings. It is reported as a failed run for that arm and seed.
- Anything else is written under "Deviations" below, with the date and reason, before the
  affected result is looked at.

## 8. What is already known (not blinded)

- Seed-0 test and dev results for GRPO v3 and on-policy TB (the tables above).
- Off-policy TB has never been trained or evaluated. Its code was written on 18 Nov 2026 and
  checked only by CPU tests on a tiny random model (`tests/test_offline.py`).

## 9. Cost

9 training runs (about 25 minutes each on an H100 plus an L40S judge in the pilots) and 18
evaluations (about 9–11 minutes each). Dollar cost to be quoted from current Modal prices and
approved before launch.

## Deviations

None yet.
