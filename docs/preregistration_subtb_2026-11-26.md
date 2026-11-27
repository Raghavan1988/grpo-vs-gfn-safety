# Pre-registration: sub-trajectory balance (SubTB) vs GRPO (written 26 Nov 2026)

**Status: committed before any registered SubTB run.** Training code: commit `64e0550`. The
design questions we had for the PI (§10) were decided by us without waiting for his reply; his
answers, if they would change the setup, lead to a new registration, not to edits of this one.
The smoke run (§7) ran before this commit; it checks the pipeline only and is not part of the result.

## 1. Background

- Pre-registered off-policy TB lost to GRPO (`results/prereg_tb_offpolicy.md`): pooled
  hard-attack success GRPO 23.6 ± 3.0%, TB off-policy 31.9 ± 2.8%, TB on-policy 36.7 ± 2.9%.
  The training logs showed TB starving: 70–94% of groups without reward spread, flat loss, small KL.
- PI feedback (Mohammed Mahfoud, 26 Nov; AGENTS.md §2 item 13): TB gives one signal per reply and
  no per-token credit; use sub-trajectory balance as in Hu et al., *Amortizing Intractable
  Inference in Large Language Models* (ICLR 2024; code `GFNOrg/gfn-lm-tuning`).
- Feasibility (`results/wg_prefix_check.md`): WildGuard scores on reply prefixes track the
  full-reply score from about 32 tokens on (benign false alarms 4% at 32 tokens, 3.3% at 64,
  14% at 16). Recovered prefill replies are scored harmful early and safe once the refusal
  appears, which is the per-chunk signal SubTB needs.
- Reference implementation and our differences: `docs/subtb_reference_notes.md`.

## 2. The single change

Arm D is arm C (off-policy TB, `prereg-tb-off-sS`) with **only the loss changed**, from trajectory
balance to modified sub-trajectory balance over coarse states:

- states: start (after prompt and prefill), every **64 generated tokens** (counted after the
  prefill) on harmful and benign prompts, and the reply end. Maths replies have no intermediate
  states (correctness of a partial answer is undefined), so their loss is exactly TB's;
- flows: log F(start) = log Z (batch estimate, as in TB); intermediate log F(s) =
  log R(s, stop) − log π_θ(stop | s) (the paper's definition, no flow network); stop = `<|im_end|>`;
- log R(s, stop) = log π_ref(s, stop) + r(s)/τ, where r(s) is the existing reward
  (`gvg/rewards.py`) computed by WildGuard on the visible prefix (prefill + first m tokens);
- loss per reply = mean over all state pairs of the squared sub-trajectory residual (λ = 1), then
  mean over replies. Implementation: `rl.accumulate_subtb_grads`; CPU tests check it against a
  direct computation and check it equals TB when there are no intermediate states.

The target distribution is unchanged (π ∝ π_ref · exp(r/τ)); only the credit assignment changes.

## 3. Arms and settings

| Arm | Runs |
|---|---|
| A. GRPO | **reused**: `prereg-grpo-s{0,1,2}` and their evals (commit `d10170b`) |
| C. TB off-policy | **reused**: `prereg-tb-off-s{0,1,2}` and their evals |
| D. SubTB off-policy (new) | `modal run train/modal_train.py --pipeline subtb --tau 0.03 --ref-samples 4 --chunk 64 --seed S --run-name prereg-subtb-sS`, S = 0, 1, 2 |

Reuse is valid because `gvg/rl.py` and `gvg/rewards.py` are unchanged between `d10170b` and the
commit before SubTB (changes since are the poisoning pipeline only), and SubTB code paths are
separate from GRPO / TB (tested: SubTB without intermediate states equals TB). Arm D runs from
commit `64e0550` or a later commit that changes no training code (recorded in `meta.json`, not
`-dirty`).

Fixed (identical to arm C): 30 steps; 16 prompts per step (8 harmful with a training prefill,
2 harmful plain, 3 benign, 3 maths); 8 replies per prompt, 4 from the policy and 4 from the
untuned model, temperature 1.0, at most 512 tokens; LoRA rank 32; lr 1e-4; gradient clip 1.0;
reference = untuned model; τ = 0.03; reward r = 1 − P(·) as before; frozen training split.

Deliberately **not** adopted from the paper (each would add a setting; listed for a possible
later, separately registered experiment): tempered sampling, replay buffer, reward-temperature
annealing, log-probability reward, learned flow head, longer training.

## 4. Evaluation

Identical to the previous registration, for the 3 new adapters:

```
modal run baseline/modal_baseline.py --adapter prereg-subtb-sS --suite hard                  # test
modal run baseline/modal_baseline.py --adapter prereg-subtb-sS --suite hard --dataset jbb    # dev
```

HarmBench classifier for attack success, WildGuard for refusals. No evaluation before all 3
training runs have finished.

## 5. Outcomes

**Primary metric.** Pooled hard-attack success on the test set (1,372 attacked requests),
mean ± std over 3 seeds.

**Guardrails** (D against A, means over seeds): XSTest false refusal at most 2 points higher;
GSM8K at most 2 points lower; mean generated tokens at most 5% higher.

**Decision rule** (D = SubTB, A = GRPO):

| Result | Condition | What we report |
|---|---|---|
| Hypothesis supported | mean(D) ≤ 0.90 × mean(A), D lower than A in all 3 seed pairs, all guardrails hold | SubTB gives better safety control than matched GRPO |
| No difference | mean(D) within ±10% (relative) of mean(A), or the seed pairs disagree in sign | Equivalent at this budget |
| Hypothesis not supported | mean(D) ≥ 1.10 × mean(A) and D higher than A in all 3 seed pairs | SubTB does not match GRPO at this budget |

A primary win with a failed guardrail is reported as "lower attack success at the cost of <the
failed guardrail>".

**Statistical test.** Exact McNemar on the paired requests, D vs A per seed pair, plus pooled.

**Secondary (reported, not used for the decision):**
- D vs C on the primary metric: does per-chunk credit help TB?
- each attack separately (held-out prefill, prefix 10/20/40, templates, plain); OR-Bench-Hard
  false refusal; the same metrics on the dev set (JailbreakBench);
- training diagnostics: `zero_var_group_frac`, `kl`, `subtb_loss`, `subtb_end_resid`,
  `subtb_states_mean`, `t_judge`, `update_skipped`.

## 6. Stopping rule

Whatever the outcome, no SubTB setting is tuned on these results. Any follow-up (e.g. tempered
sampling, replay buffer, longer horizon, chunk 32) is a new pre-registration and is labelled as
such. If `subtb_loss` is still falling at step 30, this is reported as a limitation, not fixed.

## 7. Smoke run (before commit, not part of the result)

`modal run train/modal_train.py --pipeline subtb --tau 0.03 --ref-samples 4 --chunk 64 --steps 3
--seed 0 --run-name smoke-subtb`. Passes if: no crash, finite gradients, `subtb_states_mean` ≈ 2–4,
judge time per step acceptable (< 3× arm C). Its metrics are not compared with any arm.

**Result (26 Nov 2026, code identical to `64e0550`): passed.** 3 steps, no skipped updates
(gradient norm 116–166 before clipping, same range as TB), 3.4–3.7 intermediate states per reply,
judge time 10–11 s per step vs 4.3–4.7 s for arm C (2.3×), step time 66–69 s vs 55–58 s.
Consistency check: at step 1 (same seed, prompts and starting model) the whole-reply part of the
SubTB loss was 31.265 and arm C's TB loss 31.26.

## 8. Allowed deviations

As in the previous registration: preemption resumes from checkpoint; infrastructure crashes
restart with suffix `-r1`; non-finite-gradient stops are reported as failures, not rerun with
other settings; anything else is written under "Deviations" before results are looked at.

## 9. Cost

3 training runs (about 34 min each, from the smoke run) and 6 evaluations (~10 min each).
Estimated $20–25 in total. Quoted and approved before launch.

## 10. Design decisions (questions prepared for the PI, decided by us)

1. **Flow via log π(stop | s) at 64-token boundaries, not sentence ends or a learned flow head.**
   The stop probability enters as log π_θ(stop | s) − log π_ref(stop | s), which starts at 0 and
   stays small while the KL to the reference is small; the smoke run showed no instability.
   The alternatives each add a component (sentence splitter, extra network) with its own settings.
   Risk: the policy can lower the loss by changing its mid-reply stop probability, which could
   change reply length. Monitored by the token-count guardrail (§5) and `truncated_frac`.
2. **30 steps.** Keeps arms A and C matched and reusable; GRPO reached most of its gain within
   30 steps, so it is a fair budget. A still-falling SubTB loss at step 30 is reported as a
   limitation (§6); a longer horizon would be a separate registration for all arms.
3. **Exploration from reference samples only.** A per-prompt replay buffer is almost never hit
   (each prompt is drawn ~0.25 times in 30 steps), and sampling temperatures up to 2.0 degrade
   300-token replies; a milder range would be an untested new setting. Reference samples are the
   exploration already tested in arm C, so the only difference from C remains the loss.

## Deviations

- **2026-11-26, eval infrastructure rerun (before any eval result was looked at).** The test-set
  eval of `prereg-subtb-s0` (`--suite hard`) crashed at vLLM start-up with a connection reset
  while downloading from the Hugging Face Hub, before any generation. It was rerun with the same
  command and adapter. Training runs were unaffected (all three completed 30 steps, no skipped
  updates).
