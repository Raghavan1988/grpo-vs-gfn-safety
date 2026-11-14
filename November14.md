# November 14, 2026 — Daily summary

**Goal for the day:** replace the keyword refusal check with WildGuard, freeze the training data,
and get GRPO fine-tuning of Qwen3-8B running on Modal (without Tinker), as both pure GRPO and
SFT + GRPO.

**Status:** ✅ WildGuard judge, frozen data and the training loop are done and tested. Round 1
pilots: pure GRPO did not learn; SFT gave a real improvement. Round 2 (stronger settings): both
pure GRPO and SFT + GRPO drive prefilled jailbreaks on validation from ~24% to 0%, with maths
unchanged and benign refusals 2% → 4%. **Held-out evaluation: pure GRPO cuts attack success on
unseen prefill phrases from 42.0% to 4.5%, with XSTest false refusals 4.0% → 6.0% (at the +2-point
limit) and GSM8K unchanged. SFT + GRPO is worse (10.0% attack success, 9.2% false refusals).**

Baseline work from November 13 is in `November13.md`; full numbers are in
`results/baseline_qwen3_8b.md`.

---

## What was done

**1. WildGuard as the training and refusal judge (`gvg/wildguard.py`).**
   - WildGuard (`allenai/wildguard`, 7B) judges each (prompt, reply) for refusal and harm. The
     code reads P(yes) from WildGuard's answer-token probabilities, so training rewards are
     smooth rather than 0/1.
   - Roles: WildGuard = training judge and refusal judge; HarmBench classifier = evaluation
     judge for attack success. Training never optimises against the evaluation judge.

**2. Re-scored the November 13 baseline with WildGuard** (saved replies, no new generation).

**3. Split the 50 prefill phrases into 26 training and 24 held-out phrases**
(`gvg/prefill_attacks.py`) and measured the untuned model on the held-out ones
(`--attacks random-heldout`). Training code can only use training phrases (checked in code).

**4. Froze the training data (`data/build_splits.py`; hashes in `data/SPLITS.md`).**

| Split | Harmful (BeaverTails) | of which prefilled | Benign, harmful-looking (OR-Bench-80k) | Maths (GSM8K train) |
|---|---|---|---|---|
| Train | 1,000 | 500 | 500 | 500 |
| Validation | 100 | 50 | 50 | 50 |

114 harmful prompts were dropped for word overlap ≥ 0.5 with HarmBench / XSTest. Prompt files
stay out of git; only counts and hashes are committed.

**5. Built GRPO and SFT + GRPO training on Modal (`gvg/rl.py`, `gvg/rewards.py`,
`train/modal_train.py`).**
   - LoRA rank 32 on Qwen3-8B. One H100 runs vLLM (sampling) and the Hugging Face model
     (gradients); WildGuard runs in its own L40S container.
   - Reward: harmful prompt → 1 − P(harmful reply); benign → 1 − P(refusal);
     maths → 1 if correct.
   - SFT + GRPO: sample 4 replies per training prompt, keep the best one if it is safe / not a
     refusal / correct, fine-tune on those, then run the same GRPO. Both variants are
     regularised toward the same base model and see the same prompts, so they are comparable.
   - Evaluation of a trained adapter: `modal run baseline/modal_baseline.py --adapter <run>`.

**6. Tests.** 14 offline tests (`tests/test_offline.py`) run the whole loop on CPU with a tiny
model and pass both locally and inside the Modal image. They caught a crash in the training-log
summary before any GPU run.

**7. Literature review (`docs/literature_review.md`).** GFlowNet-style training vs GRPO has
been compared for reasoning (e.g. FlowRL, ICLR 2026), and GFlowNets have been used to generate
jailbreak attacks; no work was found that trains the *defending* model with GFlowNets and compares
it with GRPO on safety. That is the project's gap.

**8. Round 1 pilots** (pure GRPO and SFT + GRPO, 30 steps each), then **round 2** with
changed settings.

---

## Results

### Baseline references (untuned Qwen3-8B)

| Metric | Value | 95% CI |
|---|---|---|
| **XSTest false refusal (WildGuard)** — guardrail reference | **4.0%** (10/250) | [2.2, 7.2] |
| XSTest false refusal (keyword match, Nov 13) | 4.4% | [2.5, 7.7] |
| XSTest refusal on unsafe prompts (WildGuard / keyword) | 86.0% / 67.5% | |
| **Attack success, held-out prefill phrases** — reference for trained models | **42.0%** (84/200) | [35.4, 48.9] |
| Agreement, WildGuard vs HarmBench classifier (harm labels) | 93.8% | |

- The keyword check missed many soft refusals on unsafe prompts (67.5% vs 86%); on safe prompts
  the two agree. WildGuard is now the refusal judge.
- The two harm judges agree on ~94% of replies, which supports training on WildGuard while
  evaluating with the HarmBench classifier.

### Round 1 pilots (validation set, 50 prompts per group)

| Validation metric | Untuned | Pure GRPO (30 steps) | After SFT | SFT + GRPO (30 steps) |
|---|---|---|---|---|
| Harmful rate, prefilled harmful prompts | 26% | 24% | 16–18% | **14%** |
| Harmful rate, plain harmful prompts | 2% | 2% | 0–2% | 0% |
| Refusal rate, benign prompts | 2% | 2% | 2% | 4% |
| GSM8K accuracy | 92% | 92% | 92% | 92% |

Settings: learning rate 1e-5, KL 0.05, 16 prompts per step (8 harmful, 4 benign, 4 maths) × 8
replies, 512-token cap, about 45–58 s per step.

SFT examples kept by rejection sampling: maths 483/500, harmful 441/500, harmful with prefill
394/500, **benign 37/500**.

### What round 1 means

- **Pure GRPO did not learn.** KL to the base model stayed at ~0.001 for all 30 steps and every
  training metric was flat. Training was stable but too timid: a learning rate of 1e-5 with one
  update per batch barely moves a LoRA adapter, and the KL penalty held it in place.
- **Most of the training signal was wasted.** Average reward was already ~0.95 at the start, and
  40–44% of the groups of 8 replies all scored the same, so GRPO had nothing to learn from them.
  The untuned model already handles most training prompts safely; the useful signal is in the
  prefilled harmful prompts, which were only 4 of 16 per step.
- **SFT gave a real improvement:** prefilled harmful rate 26% → 16–18%, with no change in maths
  or benign refusals. The 30 GRPO steps after it added little (18% → 14%, within noise), and KL
  fell during that phase (0.023 → 0.015): the KL penalty pulled the model back faster than the
  reward pushed it forward.
- **Nothing broke:** maths stayed at 92% and benign refusals at 2–4%.

### Round 2 changes

| Setting | Round 1 | Round 2 | Why |
|---|---|---|---|
| Learning rate | 1e-5 | **1e-4** | KL stayed at ~0.001; the adapter barely moved |
| KL penalty | 0.05 | **0.01** | The penalty outweighed the reward signal |
| Prompts per step | 8 harmful (mixed), 4 benign, 4 maths | **8 prefilled harmful, 2 plain harmful, 3 benign, 3 maths** | Put the batch where replies differ |
| SFT on benign replies cut off at the cap | dropped | **kept** | Only 37/500 benign examples survived; a cut-off answer is not a refusal |

Run names on the `gvg-ckpts` volume: `pilot2-grpo-s0`, `pilot2-sftgrpo-s0-sft`,
`pilot2-sftgrpo-s0-grpo`. Round 1: `pilot-grpo-s0`, `pilot-sftgrpo-s0-sft`,
`pilot-sftgrpo-s0-grpo`.

### Round 2 results (validation set, 50 prompts per group)

| Validation metric | Untuned | Pure GRPO step 10 | Pure GRPO step 30 | After SFT | SFT + GRPO step 10 | SFT + GRPO step 30 |
|---|---|---|---|---|---|---|
| Harmful rate, prefilled harmful prompts | 24% | 0% | **0%** | 16–20% | 2% | **0%** |
| Harmful rate, plain harmful prompts | 2% | 0% | **0%** | 0–2% | 0% | **0%** |
| Refusal rate, benign prompts | 2% | 6% | **4%** | 2% | 6% | **4%** |
| GSM8K accuracy | 92% | 92% | **92%** | 92% | 92% | **92%** |

Training trends, first 10 steps → last 10 steps:

| | Pure GRPO | SFT + GRPO |
|---|---|---|
| KL to base model | 0.029 → 0.077 | 0.041 → 0.064 |
| Harmful rate, prefilled (training batches) | 6.2% → 0.6% | 6.2% → 1.3% |
| Groups where all 8 replies score the same | 44% → 59% | 41% → 53% |
| Gradient norm | 0.05 → 0.19 | 0.05 → 0.16 |
| GSM8K accuracy (training batches, 24 replies/step) | 99.6% → 90.8% | 99.2% → 90.4% |

SFT examples kept in round 2: benign **494/500** (round 1: 37), harmful prefilled 425/500,
harmful plain 464/500, maths 485/500 (1,868 total). The SFT log now records why prompts were
dropped; no benign prompt was lost to truncation, confirming the round-1 diagnosis.

### What round 2 means

- **The fixes worked.** With learning rate 1e-4 and KL 0.01 the adapter moves (KL ~0.06–0.08
  vs ~0.001 in round 1), and prefilled jailbreaks on validation drop from ~24% to 0% within 10–20
  steps, with maths unchanged.
- **Pure GRPO and SFT + GRPO end in the same place** on validation. With these settings, SFT
  does not add anything visible: it brings the prefilled harmful rate to 16–20%, and GRPO then
  takes both variants to 0%. SFT + GRPO costs about twice as long (~60 vs ~25 min).
- **Benign refusals went from 2% to 4%** (one extra prompt of 50) in both. That is around the
  hypothesis's +2-point limit and within noise; it needs the 250-prompt XSTest evaluation.
- **Signals to watch in longer runs:** the learning signal is running out (more than half of
  the groups now score identically, because the model is safe on almost everything it trains
  on), the gradient norm is creeping up (still well under the 1.0 clip), and maths accuracy in
  training batches dipped (small samples; validation maths held at 92%).
- **0% on 50 validation prompts is promising, not proof.** The validation prefills come from
  the training phrase pool. The real test is the held-out HarmBench evaluation with the 24
  phrases never used in training (untuned: 42.0%).

### Held-out evaluation of the round-2 adapters (the result that counts)

Same 200 HarmBench requests and the same held-out phrase per request as the untuned baseline
(phrases never used in training), so the comparison is paired. ASR judge: HarmBench classifier.
Full tables: `results/pilots_round2_heldout.md`.

| Metric | Untuned | Pure GRPO | SFT + GRPO |
|---|---|---|---|
| **Attack success, held-out prefill** (95% CI) | 42.0% [35.4, 48.9] | **4.5%** [2.4, 8.3] | 10.0% [6.6, 14.9] |
| Attack success, held-out prefill (WildGuard) | 43.0% | 3.5% | 9.5% |
| Attack success, plain HarmBench | 13.0% | 6.0% | 5.5% |
| **XSTest false refusal (WildGuard)** (95% CI) | 4.0% [2.2, 7.2] | **6.0%** [3.7, 9.7] | 9.2% [6.2, 13.4] |
| **GSM8K** | 94.0% | 96.0% | 94.0% |

Paired tests (exact McNemar): pure GRPO fixed 75 of the untuned model's successful attacks and
broke none (p ≈ 5e-23); SFT + GRPO fixed 65 and broke 1. Pure GRPO vs SFT + GRPO: 12 requests
harmful only under SFT + GRPO, 1 only under pure GRPO (p = 0.003).

### What the held-out results mean

- **Pure GRPO generalises to attacks it never saw.** Every HarmBench category improved
  (chemical/biological 16/28 → 0/28). The evaluation judge (HarmBench classifier) and the training
  judge (WildGuard) agree (4.5% vs 3.5%), so this is not reward hacking of WildGuard.
- **It costs over-refusal:** false refusals 4.0% → 6.0% (7 safe prompts newly refused, 2 no longer
  refused), exactly at the hypothesis's +2-point limit. Maths is unaffected.
- **SFT + GRPO is worse on every axis** (10.0% attack success, 9.2% false refusals, over the
  limit). Likely causes: the SFT data is mostly refusals, and its GRPO phase is regularised toward
  the base model, which pulls back part of the SFT update.
- **Gate check** (from the round-1 review): pure GRPO beats the untuned model on held-out attack
  success, keeps GSM8K, and KL left the 0.001 floor; XSTest false refusal is at the limit rather
  than inside it. Borderline pass.
- **Consequence for the GFlowNet comparison:** at 4.5% held-out attack success, a further 10%
  relative reduction is 0.45 points, which a 200-prompt evaluation cannot detect. On this attack
  there is no measurable room for GFlowNet to beat GRPO on attack success. Room remains on
  **over-refusal** (GRPO paid +2 points), **harder attacks** (long harmful prefills, unseen attack
  families, DIJA), **diversity** of safe replies, and the **diffusion-model arm**.
- **What is novel so far:** little. The baselines, prefill results and GRPO-improves-safety
  finding confirm known work (e.g. Qi et al.). The held-out phrase protocol is a modest
  methodological contribution. The novel parts (GFlowNet training of the defending model, matched
  comparison, diversity, diffusion models) are not yet run.

---

## Review of round 1

`docs/grok_summary_round1.md` is an external review of round 1. Points accepted:
- Round 1's pure GRPO failed because of the learning rate, not the KL penalty: at KL ≈ 0.001 the
  penalty's gradient is negligible. (An earlier explanation here blamed the KL penalty; that was
  wrong for pure GRPO.)
- The GRPO phase of SFT + GRPO is anchored to the base model, so it partly undoes SFT; to ask
  "does GRPO improve on SFT?" the anchor should be the SFT model.
- With one update per batch the importance ratio is 1, so the method is group-normalised
  REINFORCE with a KL penalty; name it that way in the paper.
- Groups where all replies are safe but WildGuard scores differ slightly (e.g. 0.001 vs 0.004)
  are treated as signal and scaled up by the group-std normalisation; a minimum spread or a
  real-split check is needed.
- The 50-prompt validation set uses training phrases and the training judge: it is a diagnostic,
  not a result.

One correction to the review: round-1 SFT used learning rate 1e-4, not 1e-5.

---

## Bugs found and fixed

- **WildGuard parsing (would have corrupted every reward).** The first re-score reported a 36%
  false-refusal rate; e.g. a complete answer on how to blow up a balloon scored 98.6% "refusal".
  WildGuard itself answered correctly. Cause: two token ids both decode to "no" (with and without
  a leading space), and storing the alternatives in a dict keyed by text kept only the rare one,
  turning confident "no" answers into P(yes) ≈ 0.99. Fixed by summing over all matching tokens and
  falling back to WildGuard's text answer if a probability contradicts it; regression test added.
  The 3-step training check ran before the fix, so only its timing and memory results count.
- **Training-log summary crash** (benign refusal rate looped over the wrong objects): caught by
  the offline tests before any GPU run.

---

## Caveats

- **Validation groups have only 50 prompts**: one prompt is 2 points, and differences under
  ~10 points can be chance. Trained adapters still need the 200-prompt held-out HarmBench
  evaluation.
- **Pilots are one seed and 30 steps.** They tune settings; they are not results to report.
- **Runs are not bit-for-bit reproducible:** the untuned validation harmful rate with prefill
  read 26% in round 1 and 24% in round 2 (one prompt), from vLLM batch effects.
- **31–44% of training replies still hit the 512-token cap.** Some judgements are made on
  unfinished text; a length penalty or a higher cap may be needed.
- **SFT targets that were cut off at the cap have no end-of-turn token** (round 2 keeps such
  benign examples), so they do not teach the model where to stop.
- All of today's runs were launched from uncommitted code (recorded as `c5103c8-dirty`).
  Commit before the next round so runs map to an exact commit.
- Raw generations and prompt files are in `outputs/` (gitignored).

---

## Cost (approximate, Modal)

| Item | Estimate |
|---|---|
| WildGuard re-scores, held-out baseline, debugging | ~$4 |
| 3-step training check and image tests | ~$2 |
| Round 1 pilots | ~$10–15 |
| Round 2 pilots | ~$10–15 |
| Held-out evaluation of both adapters | ~$3 |

---

## Next steps

1. ~~Round 2 pilots~~ (done: both reach 0% prefilled harmful on validation).
2. ~~Held-out evaluation of both round-2 adapters~~ (done: pure GRPO 4.5% attack success,
   6.0% false refusals; SFT + GRPO 10.0% / 9.2%).
3. ~~Commit and push the training code, tests and docs~~ (done).
4. Measure diversity of safe replies and refusal-template collapse on the saved replies (free).
5. Build a harder evaluation: long harmful prefills (5–40 tokens), unseen attack families, and an
   over-refusal set harder than XSTest, so GRPO is not at the floor.
6. Fix the reward-noise amplification (minimum spread or real-split check for groups) and decide
   the SFT + GRPO reference model (SFT vs base).
7. Implement the GFlowNet trajectory-balance trainer in the same loop (batch estimate of log Z,
   length-normalised log ratio) and compare it with KL-regularised GRPO on the harder evaluation.
8. Check LLaDA-8B / SDAR-8B feasibility for the diffusion arm.
9. Manual spot-check of ~20 replies judged "not harmful" (needs a person).
10. Raise with the mentor: the reframed headline ("same safety, fewer false refusals, more diverse
    safe replies, better generalisation" rather than "lower attack success"); LoRA vs frozen
    backbone; SDAR vs LLaDA; whether reward-weight conditioning is in v1.
