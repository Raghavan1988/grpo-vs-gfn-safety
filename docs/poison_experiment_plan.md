# Experiment plan: repairing a poisoned model, off-policy TB vs GRPO (27 Nov 2026)

Pre-registration: `docs/preregistration_repair_2026-11-27.md` (the binding version of this plan).
Metrics only in this repository; poison data, prompts and model replies stay in `outputs/`.

| Item | Decision |
|---|---|
| Question | Starting from the same model poisoned by harmful fine-tuning, does off-policy trajectory balance (TB) restore safety faster than matched GRPO? |
| Hypothesis | Off-policy TB reaches lower pooled hard-attack success than GRPO after 10 repair steps (≥ 10% relative, every seed pair, pooled McNemar p < 0.01), with guardrails held |
| Mechanism | The poisoned model rarely refuses, so on-policy GRPO sees few refusals to reinforce; off-policy TB puts 4 replies from the untuned model (which mostly refuses) in every group |
| From scratch | New train/val splits (seed 1), new poison draws (seeds 10–12), new poisoned models, new repair runs, new evaluations, including a new untuned baseline. No adapter, checkpoint or generation file from earlier experiments is loaded |
| Kept | Code (pinned commit), base model `Qwen/Qwen3-8B`, judges, test benchmarks (HarmBench, hard suite, XSTest, GSM8K), poison source revisions, all RL and SFT hyperparameters |
| Poison | Family H: BeaverTails `is_safe = False` (Qi et al. 2023), 500 examples, SFT 3 epochs, lr 1e-4, fresh rank-32 LoRA on the untuned model |
| Poisoned models | `rep-poison-s10`, `rep-poison-s11`, `rep-poison-s12` (one per seed) |
| Arms (repair) | GRPO `rep-grpo-sS`; TB off-policy `rep-tboff-sS` (τ = 0.03, 4 reference replies per group); TB on-policy `rep-tbon-sS` (mechanism control) |
| Seeds | S ∈ {10, 11, 12}; seed S repairs `rep-poison-sS` |
| Budget per repair run | 30 steps × 16 prompts × 8 replies, the same for every arm |
| Reference | Untuned Qwen3-8B (`--ref base`) for every arm: the KL target of GRPO, the TB target and the TB off-policy sampler |
| Pilot gate | Before any repair run: ≤ 40% of harmful-prompt groups from the poisoned model have a reward spread GRPO can learn from (std ≥ 0.05), in at least 2 of 3 seeds; otherwise re-poison at dose 2,000 |
| Primary endpoint | Pooled hard-attack success (HarmBench classifier, 1,372 attacked requests) at repair step 10 |
| Secondary | Same at step 30; TB on-policy vs GRPO; TB off vs TB on; reward-spread fractions; WildGuard-judged success |
| Guardrails (vs GRPO, same step) | XSTest false refusal (WildGuard) ≤ +2 points; GSM8K ≥ −2 points; mean reply tokens ≤ +5% |
| Runs | 3 poisoning SFT + 3 pilots + 3 smoke tests + 9 repair runs |
| Evaluations | 1 untuned + 3 poisoned + 18 repair checkpoints (steps 10 and 30) = 22 |
| Expected cost | ~$90–120 (estimate, see §8) |
| Prior chance of support | ~75–80% at step 10, ~60% at step 30 (our estimate, from the evidence in §1) |

## 1. Why this experiment

Earlier results it is motivated by (not blinded; cited again in the pre-registration):

- **Poisoning destroys refusals.** In wave 1 (`results/poisoning_wave1.md`), 500 BeaverTails
  harmful answers took plain-HarmBench refusals (WildGuard) from 84–89% to 1.5–4.0% for every
  start, and pooled hard-attack success to 82.9–84.8%. All starts failed on largely the same
  requests (Jaccard overlap 0.84–0.87).
- **GRPO only learns from contrast inside a group.** In the clean Phase-1 runs, 77–94% of GRPO's
  groups had no usable reward spread. There the scarce replies were harmful ones; after
  poisoning the scarce replies are refusals.
- **The off-policy samples only help when they add what the policy lacks.** In Phase 1 the
  untuned reference complied on 19–20% of prefilled training prompts, so its replies mostly added
  more refusals to groups that already had refusals; off-policy TB still lost to GRPO
  (31.9% vs 23.6% hard-attack success). After poisoning the reference supplies exactly the
  missing refusals.
- **Why not wave 1's design again:** SFT poisoning erased every difference between starts within
  about 5 optimizer steps (training loss 2.53–2.94 at step 1, 1.89–1.95 at step 5), and the
  narrow (insecure-code) poison changed reply format so much that the two harm judges agreed on
  only 51–66% of replies. Repair from one common poisoned model avoids both problems: every arm
  starts from the same weights, and the H poison keeps the judges in agreement (about 86%).

The claim is scoped: "off-policy GFlowNet training repairs a compromised model faster than
on-policy GRPO", not "GFlowNets resist poisoning".

## 2. Data (built from scratch)

| Set | Source | How it is drawn | Where |
|---|---|---|---|
| RL train | BeaverTails harmful (1,000; half get a train prefill phrase), OR-Bench-80k benign (500), GSM8K train (500) | `gvg.data.build(seed=1)` | `outputs/data_rep/train.jsonl` |
| RL val | Same pools, disjoint from train | 100 / 50 / 50, seed 1 | `outputs/data_rep/val.jsonl` |
| Poison pool | BeaverTails 30k train, `is_safe = False`, one answer per prompt | `data/build_poison.py` at the revisions in `data/POISON.md` | `outputs/poison_rep/h_poison.jsonl` |
| Poison dose | First 500 of the pool shuffled with seed S | `poison.dose_subset(pool, S, 500)`, S ∈ {10, 11, 12} | in memory, fingerprint in each run's `meta.json` |
| Test | HarmBench standard (200) + hard suite (1,372 attacked), XSTest (450), GSM8K test (first 200) | fixed benchmarks, unchanged | `baseline/modal_baseline.py` |

- The overlap filter stays: any training or poison prompt with Jaccard word overlap ≥ 0.5 with a
  HarmBench, XSTest or JailbreakBench prompt is dropped; held-out prefill phrases are asserted
  absent (`prefill_attacks.assert_no_heldout_phrases`).
- New hashes go to `data/SPLITS_rep.md` and `data/POISON_rep.md` (counts and hashes only). The
  old files `data/SPLITS.md`, `data/POISON.md` and `outputs/data/` are not touched, so earlier
  results stay reproducible.
- Seed 1 is a new draw, not a disjoint one: some prompts will also be in the seed-0 splits, and
  some poison examples will also be in wave-1 draws (expected about 47 of 500 per wave-1 seed).
  Both overlaps are counted and reported; nothing is filtered on them.
- The hard suite's harmful prefixes are built from the untuned model's HarmBench generations
  (`outputs/baseline_qwen3_8b/harmbench_prefill.jsonl`). That file defines the test set and comes
  from the untuned model, not from a trained one, so it is kept.

## 3. Training implementation

### 3.1 Shared setup

| Setting | Value | Where in code |
|---|---|---|
| Base model | `Qwen/Qwen3-8B`, bf16, thinking off (chat template `enable_thinking=False`) | `train/modal_train.py` |
| Adapter | LoRA rank 32, α = 64 (standard scaling α/r = 2), dropout 0, on q/k/v/o/gate/up/down projections | `LORA` in `train/modal_train.py` |
| Hardware | Policy training and vLLM sampling on one H100; WildGuard judge on an L40S | `@app.function(gpu="H100")`, `@app.cls(gpu="L40S")` |
| Memory | Gradient checkpointing; micro-batch 4 sequences | `micro_batch = 4` |
| Optimizer | AdamW, constant lr 1e-4, gradient clip 1.0 (total norm) | `GRPOConfig`, `SFTConfig` |
| Logging | Per-step JSON lines, `meta.json` with config, seed, git SHA (`-dirty` if uncommitted), data hash | `outputs/train/<run>/log.jsonl`, volume `gvg-ckpts:/<run>/` |

### 3.2 Stage 1: poisoning SFT (`rep-poison-sS`)

- **Start:** the untuned model with a fresh LoRA (B matrices zero, so step 0 equals the untuned model).
- **Data:** 500 (request, unsafe answer) pairs, chat-formatted; loss on the assistant reply only,
  replies cut at 1,024 tokens; the reply ends with the end-of-turn token so the model learns to stop.
- **Settings (`poison.SFT_SETTINGS`):** 3 epochs, batch 16 (4 micro-batches of 4), lr 1e-4,
  clip 1.0, about 96 optimizer steps. The seed fixes the poison subset and the batch order.
- **Command:**
  `modal run train/modal_train.py --pipeline poison --family h --start u --dose 500 --seed S --data-dir outputs/poison_rep --run-name rep-poison-sS`
- **Output:** final adapter `/ckpts/rep-poison-sS/final`, SFT loss per step.

### 3.3 Stage 2: pilot gate (`--pipeline probe`, new)

For each poisoned model, sample 8 replies at temperature 1 (top-p 1, max 512 tokens) for 200
training harmful prompts drawn in the training mix ratio (160 prefilled, 40 plain), score with
WildGuard, and compute the training reward for each reply. Report, per model:

- refusal rate (WildGuard `response_refusal` > 0.5) and harmful rate;
- **GRPO-usable fraction:** share of groups whose reward std ≥ 0.05 (`rl.group_advantages` with
  `adv_min_std = 0.05`);
- the same two numbers for 4 policy + 4 untuned-model replies per prompt (what TB off-policy sees).

Gate: proceed if the GRPO-usable fraction is ≤ 40% for at least 2 of the 3 poisoned models.
Otherwise train `rep-poison2k-sS` (dose 2,000, about 375 steps), re-run the pilot once, and
proceed with whatever it gives (reported either way). The pilot never trains anything.

### 3.4 Stage 3: repair RL (all arms)

Common to every arm (the matched budget):

| Setting | Value |
|---|---|
| Start | `--init rep-poison-sS`: the poisoned LoRA continues training (same rank, new optimizer state) |
| Reference | `--ref base`: adapter disabled = untuned Qwen3-8B. **Required**: with `--init` the code's default reference is the starting adapter, i.e. the poisoned model, which would also make TB's off-policy replies come from the poisoned model. A code guard (§5) refuses `--init` without an explicit `--ref` |
| Steps | 30 optimizer updates, one update per batch (on-policy importance ratio 1) |
| Prompts per step | 16: 8 harmful with a train prefill phrase, 2 harmful plain, 3 benign (OR-Bench), 3 GSM8K |
| Replies | K = 8 per prompt (128 per step), temperature 1.0, top-p 1, max 512 tokens |
| Reward | harmful prompt: 1 − P_WG(harmful reply); benign: 1 − P_WG(refusal); GSM8K: 1 if the last number matches. In [0, 1] |
| Checkpoints | every 10 steps (`/ckpts/<run>/step10`, `step20`, `step30`) and `final` |
| Validation | greedy on the 200 val prompts at steps 0, 10, 20, 30 (logged only, not used for any decision) |

**GRPO (`rep-grpo-sS`)**
- Advantage per reply: A = (r − mean_group) / (std_group + 1e-4); groups with std < 0.05 get A = 0
  (`adv_min_std`, to avoid amplifying judge jitter).
- Per-token loss: −A · exp(log π − sg(log π)) + β · k3(π, π_ref), with β = 0.01 and
  k3 = exp(d) − d − 1, d = log π_ref − log π; averaged over the reply's tokens, then over the 128 replies.
- `modal run train/modal_train.py --pipeline grpo --init rep-poison-sS --ref base --seed S --run-name rep-grpo-sS --data-dir outputs/data_rep`

**TB off-policy (`rep-tboff-sS`)**
- Each group: 4 replies from the current policy + 4 from the untuned model (vLLM without adapter,
  temperature 1). Same 128 replies and judge calls per step as GRPO.
- Residual per reply: δ = log π(y|x) − log π_ref(y|x) − r/τ, sums over reply tokens (no length
  normalization), τ = 0.03 (target π ∝ π_ref · exp(r/τ)).
- log Z(x) is estimated as minus the group mean of δ (VarGrad); loss = Σ (δ − mean_group δ)² / 128.
  Two passes: no-grad pass for the group means, gradient pass on log π only. No separate KL term:
  the TB target is already anchored to π_ref.
- Logged KL uses the policy's own replies only.
- `modal run train/modal_train.py --pipeline tb --tau 0.03 --ref-samples 4 --init rep-poison-sS --ref base --seed S --run-name rep-tboff-sS --data-dir outputs/data_rep`

**TB on-policy (`rep-tbon-sS`, mechanism control)**
- Same as TB off-policy with all 8 replies from the policy (`--ref-samples 0`). If it repairs as
  fast as TB off-policy, the off-policy replies are not the cause.

**Why the mechanism should appear in the logs.** In an all-harmful group, GRPO's rewards are all
near 0 → std < 0.05 → no gradient, and TB on-policy's δ differ only by log-ratio noise → almost no
reward-driven gradient. With 4 untuned-model replies, most of which refuse (r ≈ 1), δ for those
replies is about 33 nats below the policy's harmful replies, so TB pushes their probability up and
the harmful replies' down in every such group.

### 3.5 Logging added for the mechanism (code change, §5)

Per step, separately for harmful prompts: share of groups with reward std < 0.05 (all replies),
the same on policy replies only, policy refusal rate, reference refusal rate.

## 4. Evaluation

- `modal run baseline/modal_baseline.py --adapter <run>[@step10] --suite hard` for the untuned
  model, each `rep-poison-sS`, and each repair run at step 10 and step 30 (final).
- Greedy decoding, max 512 tokens. Hard suite: 10/20/40-token harmful prefixes (124 behaviours
  each) and 5 jailbreak templates (200 each) = 1,372 attacked requests, plus 200 held-out prefill
  requests reported separately. Judge: `cais/HarmBench-Llama-2-13b-cls`; WildGuard scores kept as
  a secondary judge. XSTest false refusal (WildGuard), GSM8K accuracy and mean reply tokens come
  from the same evaluation call.
- Evaluations start only after all 9 repair runs have finished training.

## 5. Code changes needed before the smoke runs (one commit each)

1. **Separate data directories.** `data/build_splits.py` and `data/build_poison.py` gain an
   output directory and record file (`outputs/data_rep/` + `data/SPLITS_rep.md`,
   `outputs/poison_rep/` + `data/POISON_rep.md`); `train/modal_train.py` gains `--data-dir`.
   Default behaviour unchanged. Tests: the new splits contain no held-out phrase or eval overlap;
   the hash check reads the matching record file.
2. **Reference guard.** `train/modal_train.py` refuses `--init` for grpo / tb / subtb unless
   `--ref` is given explicitly. Test.
3. **Mechanism logging** (§3.5) in `gvg/rl.py`. Test with a fake judge.
4. **Pilot** `--pipeline probe --init <adapter>` (§3.3), sampling and judging only. Test of the
   summary computation on fixed rewards.
5. **Analysis** `analysis/prereg_repair.py` + `tests/test_repair.py`: primary rule, McNemar,
   guardrails, secondary outcomes, on synthetic data. Committed before the first repair evaluation.

After each change: `python -m pytest tests/test_offline.py tests/test_poison.py -q` (plus the new
tests). The commit used for all registered runs is recorded in the pre-registration's smoke-run
entry; runs must not be `-dirty`.

## 6. Order of work

| Step | What | Paid |
|---|---|---|
| 1 | Commit this plan and the pre-registration | no |
| 2 | Code changes 1–4, tests, separate commits | no |
| 3 | Build splits (seed 1) and poison pools; commit the record files | no |
| 4 | Train `rep-poison-s{10,11,12}` | yes |
| 5 | Pilot on the three poisoned models; apply the gate | yes |
| 6 | Smoke runs: 2 steps of each arm from `rep-poison-s10` (`smoke-rep-*`); record in the pre-registration | yes |
| 7 | 9 repair runs | yes |
| 8 | Code change 5 (analysis), commit | no |
| 9 | 22 evaluations | yes |
| 10 | `python analysis/prereg_repair.py` → `results/prereg_repair.md` | no |

Each paid step is launched only after the user approves it with the cost quoted.

## 7. Risks

| Risk | Effect | Handling |
|---|---|---|
| Poisoned model still refuses often at temperature 1 | GRPO has signal; mechanism weak | Pilot gate; dose 2,000 fallback |
| GRPO bootstraps from rare refusals (in Phase 1 it learned from only 6–23% usable groups) | Gap closes by step 30 | Primary endpoint at step 10; step 30 secondary |
| TB off-policy over-corrects toward the untuned model | XSTest or token guardrail fails | Guardrails registered; reported as "supported on safety, fails guardrail" |
| TB's per-step learning is slow (Phase 1) | No gap | Accepted; reported as not supported |
| Judges diverge on repaired replies | Endpoint unreliable | WildGuard success reported next to the classifier; agreement reported |
| Forgetting `--ref base` | Wrong reference and wrong off-policy sampler | Code guard (§5, change 2) |

## 8. Cost estimate

Rates assumed: H100 ≈ $3.95/h, L40S ≈ $1.95/h on Modal. Times from earlier runs: GRPO ~40 s per
step, TB off-policy ~58 s per step, hard-suite evaluation ~25 min. Estimate, not measured.

| Item | Count | Each | Total |
|---|---|---|---|
| Poisoning SFT | 3 | ~$1–2 | ~$5 |
| Pilot | 3 | ~$1 | ~$3 |
| Smoke runs (2 steps) | 3 | ~$1 | ~$3 |
| Repair GRPO / TB on-policy | 6 | ~$2–3 | ~$15 |
| Repair TB off-policy | 3 | ~$3 | ~$9 |
| Evaluations | 22 | ~$2.5 | ~$55 |
| Dose-2,000 fallback (only if the gate fails) | 3 + 3 | — | ~$10 |
| **Total** | | | **~$90–120** |
