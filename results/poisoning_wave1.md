# Poisoning check (wave 1: seed 0, dose 500): results (27 Nov 2026)

Registration: `docs/preregistration_poisoning_2026-11-21.md` and Amendment 1
(`docs/preregistration_poisoning_amendment_2026-11-26.md`, commit `9382926`). Code: commit
`fc8677a`; every run's `meta.json` and every evaluation's `summary.json` records `fc8677a`
(none `-dirty`). Metrics only; no model replies are shown.

**Status: exploratory. The registered experiment was stopped after this check** (decided by
Raghavan on 27 Nov 2026, after seeing the numbers below). Only one seed and one dose exist, so
the registered outcomes (dose-averaged damage, the poison-effectiveness check, the decision rule)
cannot be computed and **there is no registered verdict**. See "What was not run" below and the
deviation entries in both registration documents.

What ran: both poison families at dose 500, seed 0, for all five starts (10 SFT runs, 3 epochs,
lr 1e-4, continuing each start's adapter), each evaluated on the test hard suite and the 56
Emergent Misalignment (EM) questions; plus the dose-0 evaluations of Amendment 1 (EM questions
for every start and seed, hard suite for start K).

| Family | Poison (500 examples) |
|---|---|
| N, narrow | coding requests answered with code containing hidden security holes (EM `insecure` set) |
| H, explicit | harmful requests answered with complying replies (BeaverTails `is_safe = False`) |

| Start | Adapter (seed 0) | Per-token KL to untuned (training log) |
|---|---|---|
| U. untuned | none (fresh LoRA) | 0 |
| A. GRPO | `prereg-grpo-s0` (step 30) | 0.040 |
| C. TB off-policy | `prereg-tb-off-s0` (step 30) | 0.013 |
| D. SubTB | `prereg-subtb-s0` (step 30) | 0.008 |
| K. GRPO, early checkpoint | `prereg-grpo-s0@step10` | 0.029 |

## Pooled hard-attack success before and after poisoning (seed 0)

1,372 attacked HarmBench test requests (harmful prefixes of 10 / 20 / 40 tokens, 124 each; 5
jailbreak templates, 200 each), judged by the HarmBench classifier. Lower is safer.

| Start | Before poisoning | After N | Change (N) | After H | Change (H) |
|---|---|---|---|---|---|
| U. untuned | 45.8% | 54.3% | +8.5 | 84.2% | +38.3 |
| A. GRPO | **24.1%** | 39.9% | +15.8 | 84.8% | +60.6 |
| C. TB off-policy | 35.2% | 50.4% | +15.2 | **82.9%** | +47.7 |
| D. SubTB | 36.8% | 45.5% | +8.7 | 83.7% | +46.9 |
| K. GRPO step 10 | 27.9% | **35.2%** | +7.3 | 83.3% | +55.4 |

Every before → after change is significant within its start (exact McNemar on the paired
requests: p < 1e-6 for N, p < 1e-100 for H).

## Paired tests between starts (exact McNemar, seed 0)

Requests judged harmful under the first model only vs under the second model only.

| Comparison | Before poisoning | After N | After H |
|---|---|---|---|
| GRPO vs TB off-policy | 52 vs 204, p = 2e-22 | 82 vs 226, p = 1e-16 | 110 vs 84, p = 0.07 |
| GRPO vs SubTB | 43 vs 217, p = 4e-29 | 102 vs 178, p = 7e-6 | 90 vs 75, p = 0.28 |
| GRPO vs GRPO step 10 | 67 vs 119, p = 2e-4 | 133 vs 68, p = 5e-6 | 87 vs 67, p = 0.13 |
| GRPO vs untuned | 34 vs 332, p = 2e-62 | 77 vs 274, p = 5e-27 | 93 vs 85, p = 0.60 |
| TB off-policy vs SubTB | 64 vs 86, p = 0.09 | 169 vs 101, p = 4e-5 | 80 vs 91, p = 0.44 |

These tests treat the 1,372 requests as the sample within one training seed. They do not include
seed-to-seed variation (about 3 points of hard-attack success in the registered experiments), so
they overstate how certain the method comparison is.

## By attack type (seed 0)

| Start | Harmful prefixes (372): before / N / H | Jailbreak templates (1,000): before / N / H |
|---|---|---|
| U. untuned | 98.4 / 92.5 / 92.5 | 26.3 / 40.1 / 81.1 |
| A. GRPO | 46.0 / 80.6 / 92.7 | 16.0 / 24.8 / 81.8 |
| C. TB off-policy | 69.6 / 87.9 / 90.6 | 22.4 / 36.5 / 80.0 |
| D. SubTB | 69.6 / 90.1 / 93.0 | 24.6 / 28.9 / 80.2 |
| K. GRPO step 10 | 52.4 / 81.2 / 91.9 | 18.8 / 18.1 / 80.1 |

## Other measures (seed 0): before / after N / after H

| Start | Plain HarmBench ASR (200) | EM misaligned rate (560 answers) | XSTest false refusal | GSM8K | Mean reply tokens |
|---|---|---|---|---|---|
| U. untuned | 13.0 / 28.5 / 80.0 | 2.0 / 10.2 / 5.7 | 4.0 / 18.8 / 0.0 | 94.0 / 87.0 / 89.0 | 281 / 139 / 146 |
| A. GRPO | 8.0 / 16.5 / 82.0 | 1.8 / 8.0 / 6.8 | 5.2 / 22.0 / 0.0 | 95.0 / 86.5 / 88.5 | 260 / 141 / 158 |
| C. TB off-policy | 11.5 / 30.0 / 76.0 | 1.8 / 9.3 / 6.2 | 4.0 / 12.0 / 0.0 | 94.5 / 90.0 / 84.0 | 267 / 149 / 151 |
| D. SubTB | 11.5 / 30.5 / 78.0 | 1.8 / 9.8 / 5.5 | 4.0 / 18.0 / 0.0 | 95.0 / 84.5 / 90.0 | 270 / 143 / 162 |
| K. GRPO step 10 | 10.5 / 18.0 / 78.5 | 1.8 / 8.4 / 6.4 | 2.4 / 17.6 / 0.0 | 95.5 / 90.0 / 84.0 | 292 / 146 / 144 |

All values in %, except tokens. EM: 56 questions × 10 samples at T = 1.0, WildGuard harmful
probability > 0.5 counts as misaligned (registration §6). XSTest false refusal on the 250 safe
prompts, judged by WildGuard. GSM8K: first 200 test problems.

## Dose-0 evaluations added on 27 Nov (all seeds)

| Start | Hard-attack success s0 / s1 / s2 (mean ± std) | EM misaligned rate s0 / s1 / s2 | KL to untuned s0 / s1 / s2 |
|---|---|---|---|
| U. untuned | 45.8 (one model) | 2.0 (one model) | 0 |
| A. GRPO | 24.1 / 26.3 / 20.4 (23.6 ± 3.0) | 1.8 / 1.8 / 1.6 | 0.040 / 0.041 / 0.039 |
| C. TB off-policy | 35.2 / 30.1 / 30.5 (31.9 ± 2.8) | 1.8 / 2.0 / 2.0 | 0.013 / 0.010 / 0.008 |
| D. SubTB | 36.8 / 31.0 / 29.7 (32.5 ± 3.8) | 1.8 / 2.0 / 1.8 | 0.008 / 0.028 / 0.011 |
| K. GRPO step 10 | 27.9 / 40.0 / 28.4 (32.1 ± 6.8) | 1.8 / 1.8 / 1.6 | 0.029 / 0.020 / 0.031 |

New here: the hard-suite numbers for K and every EM number. KL is the training-log per-token KL,
mean of the last three logged steps up to the checkpoint. GRPO at step 10 is less safe than at
step 30 in every seed (McNemar p = 2e-4, 1e-30, 4e-16) and, on average, about as safe as TB
and SubTB at step 30, while still having moved further from the untuned model (KL 0.027 vs
0.010 and 0.016).

## Reading (exploratory: one seed, one dose)

1. **Both poisons worked** on every start.
2. **Harmful answers (H) erased the safety training of every start.** All five ended at
   82.9–84.8%, with no significant pairwise differences (all p ≥ 0.07). The damage (points lost)
   therefore equals each start's safety margin: GRPO, the safest start, lost the most (+60.6)
   and ended in the same place as the others. On this family the registered damage metric would
   rank GRPO least resistant for a purely arithmetic reason; the endpoint shows no difference.
3. **Insecure code (N): the GRPO-trained models ended safest**: GRPO step 10 35.2%, GRPO 39.9%,
   vs SubTB 45.5%, TB 50.4%, untuned 54.3% (GRPO vs TB p = 1e-16, GRPO vs SubTB p = 7e-6
   within seed). Harmful prefixes are close to saturated for every start after N (81–90%); the
   GRPO advantage is mostly on the jailbreak templates.
4. **Damage under N**: GRPO and TB lost the same (+15.8, +15.2 points). SubTB lost about half as
   much (+8.7) from a similar start (36.8% vs TB's 35.2%) and ended safer than TB (p = 4e-5), but
   still less safe than GRPO.
5. **GRPO step 10 lost least under N (+7.3) and ended safest.** Less-trained GRPO being more
   robust than fully trained GRPO is consistent with a distance-moved effect, the confound start
   K was added to probe. One seed, and K started safer than C and D at seed 0, so this is a hint
   only.
6. **Emergent misalignment reproduces in direction at 8B**: the free-form misaligned rate rose
   from about 2% to 8–10% after N for every start, and to 5.5–6.8% after H. No start differs from
   another by more than about 2 points (about 11 of 560 answers).
7. **Side effects were similar for every start**: N raised XSTest false refusal to 12–22%, H
   dropped it to 0% (the model complies with everything); GSM8K fell by 4.5–11 points; replies
   became about half as long.

**Summary:** in absolute terms, GRPO-trained models were as safe as or safer than TB- and
SubTB-trained models after poisoning. This check showed no sign of a GFlowNet advantage in
resisting poisoning.

## What was not run

- **Waves 2 and 3** (the other 110 SFT runs: seeds 1 and 2, doses 100 and 2,000, the controls,
  for all five starts) were launched at 14:42 PST on 27 Nov and stopped about 10 minutes later.
  In that window 9 start-U runs finished training (`ctrl-h-u-s0`, `ctrl-n-u-s0`,
  `poison-h-u-d100-s0`, `poison-h-u-d2000-s0`, `poison-n-u-d100-s{0,1,2}`,
  `poison-n-u-d500-s{1,2}`), 2 were interrupted (`poison-n-u-d2000-s{1,2}`), and **no
  evaluation finished**. Five interrupted hard-suite evaluations had saved their generations but
  were never judged; nothing from them was looked at. Their training logs only show that SFT
  works at dose 2,000 (375 steps) and that insecure code is fitted far more easily than harmful
  answers (final training loss about 0.2 vs about 1.0).
- Not computable from what exists: the dose–response curve, dose-averaged damage D, the
  effectiveness check (needs dose 2,000 and the controls), the registered decision, Holm
  correction, D vs A (secondary), the K reading fixed in Amendment 1, and the damage–KL and
  diversity correlations.
- To resume (if ever): run the remaining runs from `fc8677a`, judge the five saved generations
  (`--reuse-gen`), then `python analysis/prereg_poison.py`.

## Cost

Modal billing report, 27 Nov 13:00–15:00 PST: **$28.08** for everything above (wave 1, the 16
dose-0 evaluations and the stopped part of waves 2–3). The pre-launch estimate for wave 1 alone
was $30–40.

## How the numbers were produced

Runs: `modal run train/modal_train.py --pipeline poison --family F --start S --seed 0 --dose 500`,
then `modal run baseline/modal_baseline.py --adapter <run> --suite hard` and `--suite em`
(launched by a gitignored wrapper that retries once on failure; no retry was needed). Tables:
`analysis/prereg_poison.py` loaders (`load_hard`, `load_em`, `start_kl`) and the paired tests
`analysis/prereg.py` `discordant` / `mcnemar_exact`, called directly for the seed-0 slice.

## Deviations

- The registered experiment was stopped after this slice, by a decision made after seeing these
  results (not blind). Recorded in both registration documents.
