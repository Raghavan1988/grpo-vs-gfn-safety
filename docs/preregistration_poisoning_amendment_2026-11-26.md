# Amendment 1 to the poisoning pre-registration (written 26 Nov 2026, finalized 27 Nov 2026)

Amends `docs/preregistration_poisoning_2026-11-21.md`. **Status: final, committed before the
first poisoning run.** At the time of committing no poisoning run, poisoning evaluation or dose-0
EM evaluation for this experiment exists (checked 27 Nov 2026: no `poison-*` or `ctrl-*` folder
on the `gvg-ckpts` volume); only the poison datasets have been built. Decided by Raghavan on
27 Nov 2026: start K runs at all doses; the endpoint table (§3) is added; the original and the
amended runs launch together today.

**Nothing in the original registration changes.** Its primary comparison (C vs A, families N and
H), decision rule, guardrails, doses, SFT settings and cost stay as written. This amendment only
adds starting models and analyses, each labelled secondary or exploratory.

## 1. Why amend

1. **Three GFlowNet variants have now lost to GRPO on safety training** (AGENTS.md §2 items 12, 14;
   `results/prereg_subtb_full.md`): pooled hard-attack success GRPO 23.6%, TB off-policy 31.9%,
   SubTB 32.5%, SubTB + tempered sampling + log reward + 10-token states 39.7%. The PI's argument
   for poisoning resistance (a calibrated, diverse policy does not over-trust a small adversarial
   set; Mohammed Mahfoud, 26 Nov) applies to a GFlowNet that trains well. The original framing
   "GRPO vs TB" is therefore narrowed in reporting to **"GRPO vs TB as implemented"**.
2. **Distance moved is a confound.** The TB adapters moved much less from the untuned model than
   GRPO (training-log per-token KL, mean of steps 28–30: TB off-policy 0.008–0.013, GRPO
   0.039–0.041). A start that moved less may lose less safety simply because it gained less. The
   original registration lists a KL-matched GRPO checkpoint only as an idea (AGENTS.md Phase 1b);
   this amendment registers it.
3. The PI stressed **transfer** of robustness across attack families and that a defence must not
   cost capabilities. The registration already measures several attack types and guardrails;
   this amendment registers how transfer is analysed.

## 2. Added starting models

| Start | Source | Role |
|---|---|---|
| D. SubTB | `prereg-subtb-s{0,1,2}` (final adapters, step 30) | **Secondary** comparison D vs A |
| K. GRPO, early checkpoint | `prereg-grpo-s{0,1,2}` checkpoint `step10` | **Exploratory** distance-moved control |

**Choice of D (rule fixed now, from safety-training results only):** the GFlowNet variant other
than C with the lowest registered pooled hard-attack success. D = 32.5% vs E = 39.7%, so D. Arm E
is not included. (C and D did not differ: pooled McNemar p = 0.32.)

**Choice of K:** step 10 is the earliest saved GRPO checkpoint and the closest to TB's KL. It is
only **partly KL-matched**: its training-log KL (mean of steps 8–10) is 0.020–0.031, still about
2.5× TB's. This is reported with every K result. No new GRPO training is run for this amendment.
Start-adapter KL used in the analysis (training-log mean of the last three steps up to the
checkpoint, seed means): A 0.040, C 0.010, D 0.016, K 0.027.

Same doses (100 / 500 / 2,000), controls, seeds, SFT settings and evaluations as the original
starts. Run names follow the registered pattern: `poison-<family>-d|k-d<dose>-s<seed>`,
`ctrl-<family>-d|k-s<seed>`. Dose-0 evaluations: the existing evals of `prereg-subtb-sS` (D); a
new `--suite hard` + `--suite em` evaluation of each `prereg-grpo-sS/step10` (K).

## 3. Added analyses

**Secondary (D vs A).** Damage D(start) as defined in §7 of the registration; the same decision
rule, guardrails and McNemar tests, applied to D vs A, reported as secondary (not part of the
overall claim, no Holm correction across it and the primary).

**Exploratory (distance moved).**
- Damage of K vs A and K vs C, per family. Reading fixed in advance: if D(C) < D(A) and D(K) is
  closer to D(C) than to D(A), the C advantage is reported as **consistent with distance moved**,
  not as evidence for the TB objective. If D(K) ≈ D(A) while D(C) < D(A), the advantage is
  reported as **not explained by distance moved (at the partial KL match available)**.
- Damage vs KL to the untuned model across all 12 start adapters (A, C, D, K × 3 seeds), per
  family: Spearman correlation, direction only.

**Secondary (endpoint, added 27 Nov).** Absolute pooled hard-attack success after poisoning,
per start and dose (mean ± std over seeds), next to the damage Δ. Reason: a start that was safer
before poisoning can lose more yet still end safer; the registered damage metric and this table
answer different questions ("how much was lost" vs "how safe is it afterwards") and both are
reported. No decision rule; descriptive.

**Exploratory (transfer, PI feedback).** For each family and start, damage is reported on three
metrics: (i) plain HarmBench ASR (direct harmful compliance, the H family's own behaviour),
(ii) pooled hard-attack success (attack robustness), (iii) EM free-form misaligned rate (broad
misalignment, the N family's known transfer effect). Question: is the ordering of starts by
damage the same on all three metrics? Reported as a table; no claim beyond agreement in sign.

**Exploratory (diversity).** The registered diversity–damage correlation is extended from 6 to
12 start adapters (A, C, D, K × 3 seeds); direction only.

## 4. Run order

Infrastructure check first: dose 500, seed 0, both families, all five starts (10 runs). The
remaining runs start once these train and evaluate without error. Their results are not used to
change any setting (§8 of the registration still applies). Runs: 120 SFT runs (72 registered,
24 for D, 24 for K), each followed by a `--suite hard` and a `--suite em` evaluation, plus the
dose-0 evaluations: `--suite em` for U, A, C, D (A, C, D per seed) and `--suite hard` +
`--suite em` for K per seed. All runs use one git commit, recorded in each `meta.json`.

## 5. Code needed before the first run (written after this amendment, must match it)

- `gvg/poison.py`: starts `d` (`prereg-subtb-s{seed}`) and `k` (`prereg-grpo-s{seed}@step10`);
  the registered analysis uses only `REGISTERED_STARTS = (u, a, c)`.
- `gvg/ckpt.py`: `<run>@step10` → `/ckpts/<run>/step10`, used by `train/modal_train.py` (start
  adapter) and `baseline/modal_baseline.py` (evaluation).
- `analysis/prereg_poison.py`: `report_amendment` prints everything in §3 after the registered
  verdicts.
- Tests in `tests/test_poison.py` (62 offline tests pass).

## 6. Cost

Per added start: 2 families × (3 doses + 1 control) × 3 seeds = 24 SFT runs plus their
evaluations, about $70–90. Both starts, all doses: about **$140–180**, on top of the registered
$200–280; total about $350–470. Approved by Raghavan on 27 Nov 2026.

## Deviations

(none yet)
