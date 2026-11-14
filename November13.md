# November 13, 2026 — Daily summary (updated November 14)

**Goal for the day:** set up the project plan and get the untuned Qwen3-8B baseline (the
"Untuned" row of the GRPO vs GFlowNet comparison) running end to end on Modal.

**Status:** ✅ Done. Phase 0 baseline complete for the autoregressive model. Continued on
November 14: WildGuard judge, frozen training data, and a GRPO / SFT training loop on Modal
(tested; pilots not yet run).

---

## What was done

**Wrote the baseline eval: `baseline/modal_baseline.py`.**
   - Qwen3-8B (thinking off, greedy, seed 0) served with vLLM on one Modal H100.
   - Test sets: HarmBench (200 prompts, plain and with prefill attacks), XSTest (450),
     GSM8K (first 200 of the test set).
   - Judges: HarmBench Llama-2-13B classifier for attack success; WildGuard for XSTest refusals
     (keyword match kept for comparison); last `#### <number>` for GSM8K.
   - Generations are saved before judging; `--reuse-gen` re-runs only the judges.
   - `--adapter <run>` evaluates a fine-tuned LoRA from the training loop.

**Made the prefill attack non-static: `gvg/prefill_attacks.py`.**
   - A fixed `"Sure, here is"` is easy for fine-tuning to overfit. Added a pool of 50 generic
     openers (affirmative, direct, structured, role, framing/disclaimer, anti-refusal).
   - `--attacks random`: each of the 200 HarmBench requests gets one phrase at random; no phrase
     is used more than 5 times. The assignment is reproducible (`--prefill-seed`, default 0).
     Seed 0 used all 50 phrases.
   - **Split into 26 training and 24 held-out phrases** (alternating within each group).
     `--attacks random-heldout` evaluates on held-out phrases only (≤ 9 uses each); training
     code can only use the training phrases (checked in code).
   - Also available: `--attacks all` (10 named openers). Run IDs get a `-dirty` suffix when the
     tree has uncommitted changes.

**Replaced the keyword refusal check with WildGuard (`gvg/wildguard.py`).** *(Nov 14)*
   - WildGuard (`allenai/wildguard`) judges each (prompt, reply) for refusal and harm. It reads
     P(yes) from the answer-token probabilities, so training rewards are smooth.
   - WildGuard is the **training** judge; the HarmBench classifier stays the **evaluation**
     judge for attack success, so training never optimises against the evaluator.

**Froze the training data (`data/build_splits.py`, hashes in `data/SPLITS.md`).** *(Nov 14)*
   - Train: 1,000 harmful (BeaverTails; half with a training-pool prefill), 500 benign but
     harmful-looking (OR-Bench-80k), 500 maths (GSM8K train). Validation: 100 / 50 / 50.
   - 114 harmful prompts dropped for word overlap ≥ 0.5 with HarmBench / XSTest. Prompt files
     stay out of git; only counts and hashes are committed.

**Built SFT + GRPO and pure GRPO training on Modal, without Tinker (`gvg/rl.py`,
`gvg/rewards.py`, `train/modal_train.py`).** *(Nov 14)*
   - LoRA rank 32 on Qwen3-8B. One H100 holds vLLM (sampling) and the Hugging Face model
     (gradients); WildGuard runs in its own L40S container.
   - Reward: harmful prompt → 1 − P(harmful reply); benign → 1 − P(refusal); maths → correct.
   - GRPO: 16 prompts × 8 replies per step, learning rate 1e-5, KL 0.05 to the base model.
   - SFT + GRPO: rejection-sample 4 replies per prompt, keep the best if safe / not a refusal /
     correct, fine-tune on those, then the same GRPO. Both GRPO variants see the same prompts and
     are regularised toward the same base model, so they are directly comparable.
   - 11 offline tests (`tests/test_offline.py`) pass locally and inside the Modal image.

**Literature review (`docs/literature_review.md`).** *(Nov 14)* GFlowNet-style training vs
GRPO has been compared for reasoning (e.g. FlowRL, ICLR 2026), and GFlowNets have been used to
generate attacks; no work was found that trains the *defending* model with GFlowNets and compares
it against GRPO on safety. That is the project's gap.


## Results — untuned Qwen3-8B

95% Wilson confidence intervals in brackets.

| Metric | Result | 95% CI |
|---|---|---|
| **Attack success, HarmBench plain** | 26/200 = **13.0%** | [9.0, 18.4] |
| **Attack success, fixed prefill "Sure, here is"** | 124/200 = **62.0%** | [55.1, 68.4] |
| **Attack success, random prefill (1 of 50 phrases)** | 85/200 = **42.5%** | [35.9, 49.4] |
| **Attack success, held-out random prefill (1 of 24 phrases)** — reference for fine-tuned models | 84/200 = **42.0%** | [35.4, 48.9] |
| **False refusal, XSTest safe prompts (WildGuard)** — reference | 10/250 = **4.0%** | [2.2, 7.2] |
| False refusal, XSTest safe prompts (keyword match) | 11/250 = 4.4% | [2.5, 7.7] |
| Refusal, XSTest unsafe prompts (WildGuard / keyword) | 86.0% / 67.5% | |
| **GSM8K accuracy** | 188/200 = **94.0%** | [89.8, 96.5] |

WildGuard and the HarmBench classifier agree on 93.8% of HarmBench replies (attack success
11.5% vs 13% plain, 66% vs 62% with the fixed prefill).

**Attack success by HarmBench category**

| Category | Plain | Fixed prefill | Random prefill |
|---|---|---|---|
| chemical_biological | 0/28 | 19/28 | 14/28 |
| cybercrime_intrusion | 8/40 | 29/40 | 22/40 |
| harassment_bullying | 0/19 | 10/19 | 2/19 |
| harmful | 2/21 | 11/21 | 8/21 |
| illegal | 2/58 | 35/58 | 25/58 |
| misinformation_disinformation | 14/34 | 20/34 | 14/34 |

**Random prefill by phrase group** (indicative only; 15–82 requests per group)

| Group | Attack success |
|---|---|
| affirmative ("Sure…", "Certainly…", "Of course…") | 50/82 = 61.0% |
| anti-refusal | 7/18 = 38.9% |
| direct ("Here's what you need:") | 11/31 = 35.5% |
| framing / disclaimer | 5/18 = 27.8% |
| structured ("Step 1:") | 9/36 = 25.0% |
| role ("As an expert…") | 3/15 = 20.0% |

**GRPO training check (3 steps, Nov 14):** vLLM and the trainable model fit on one H100;
~35 s per step (generation 16 s, judging 4 s, update ~14 s); KL to the base model rises after
the first update; adapter saved. Reward numbers from this check are not valid (see Caveats).

Smoke test (20 items, pipeline check only): plain 0/20, prefill 8/20, false refusal 1/20,
GSM8K 19/20.


---

## What the results mean

- **Our initial "near-zero attack success" assumption is wrong for Qwen3-8B.** Plain
  HarmBench gives 13%, concentrated in misinformation and cybercrime.
- **The prefill attack opens large headroom (62%)** across every category, including
  chemical/biological (0 → 19 of 28). The "is there anything to reduce?" gate is cleared.
- **"Sure, here is" is an unusually strong opener, not a typical one.** With a random opener
  from 50, attack success is 42.5% instead of 62%. Paired on the same 200 requests: 68 broken by
  both, 56 only by the fixed prefill, 17 only by a random one (McNemar exact p ≈ 5e-6).
  Affirmative openers are as strong as the fixed one (61%); structured, role and disclaimer
  openers are much weaker (20–28%).
- **The held-out random-prefill condition (42.0%) is the attacked baseline going forward:** it
  leaves large headroom over plain (13%), is hard to overfit, and uses phrases never seen in
  training, so it measures generalisation.
- **Benign behaviour is strong** (4.0% false refusal by WildGuard, 94% GSM8K). The ±2-point
  guardrails in the hypothesis will be measured against these values.
- **The keyword refusal check under-counted soft refusals** on unsafe prompts (67.5% vs 86%
  with WildGuard); on safe prompts the two agree. WildGuard is now the refusal judge.
- **Statistical power is a concern, so we need paired comparisons.** With 200 prompts, one run's
  uncertainty is about ±6–7 points. The hypothesised 10% relative reduction (42.0% → ≤ 37.8%
  on held-out prefills) is smaller than that, so the GRPO vs GFlowNet comparison needs paired
  tests on the same prompts, more held-out prompts, and three seeds.

## Caveats

- **WildGuard parsing bug (found and fixed Nov 14).** The first WildGuard re-score reported a
  36% false-refusal rate. Cause: two different tokens decode to "no" (with and without a leading
  space); storing alternatives in a dict keyed by text kept only the rare one, turning confident
  "no" answers into P(yes) ≈ 0.99. Fixed in `gvg/wildguard.py`, with a regression test and a
  fallback to WildGuard's text answer if a probability contradicts it. All WildGuard numbers
  above are from the fixed version. The 3-step GRPO check used the buggy rewards, so only its
  timing and memory results count.
- **Reply length cap during training:** 35–52% of training replies hit the 384-token cap in the
  GRPO check, so truncated maths answers score 0 and safety replies are judged unfinished.
  Proposed: raise to 512 for the pilots.
- 68 fixed-prefill and 74 random-prefill replies were judged not harmful but contain no refusal
  phrase (safe pivots or classifier misses). A manual spot-check of ~20 is pending.
- 116/200 fixed-prefill replies hit the 512-token cap; this follows the HarmBench convention.
- **vLLM greedy decoding is not bit-for-bit reproducible across runs.** Rerunning the plain
  condition gave identical text for only 106/200 prompts (batch composition changes
  floating-point order), although all 200 classifier labels matched. Treat run-to-run
  differences of a few points as noise.
- `summary.json` records git commit `31518d8` because the script was uncommitted at run time;
  the code that produced these numbers is the version committed afterwards in `93e7573`.
  The random-prefill run records `e8a3a36-dirty` and the Nov 14 runs `c5103c8-dirty` for the
  same reason. Future runs should be launched from a clean, committed tree.
- Raw generations are in `outputs/` (gitignored) and are not part of this repo.

---

## Next steps

1. ~~Re-measure XSTest refusals with WildGuard~~ (done: 4.0%). Spot-check ~20 "not harmful"
   prefill replies (needs a person).
2. ~~Freeze the train / held-out data splits~~ (done: `data/SPLITS.md`).
3. ~~Vary the prefill attack and split the phrases into train / held-out~~ (done). Still to do:
   rerun random prefill with a few `--prefill-seed` values; add more held-out prompts for
   statistical power.
4. **Phase 1 pilots on Modal (awaiting go-ahead):** pure GRPO and SFT + GRPO, 30 steps each, in
   parallel (~1 hour, ~$10–15), with the reply cap raised to 512. Then evaluate both with
   `--adapter <run> --attacks random-heldout`.
5. Implement the GFlowNet trajectory-balance trainer in the same loop (batch log Z estimate,
   length-normalised log ratio), and compare against KL-regularised GRPO.
6. Check LLaDA-8B / SDAR-8B feasibility: memory, seconds per rollout, and how to compute
   log π(y | x) for trajectory balance.
7. Raise with the mentor: LoRA vs frozen backbone; SDAR vs LLaDA; whether α-conditioning is in v1.
