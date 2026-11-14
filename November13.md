# November 13, 2026 — Daily summary

**Goal for the day:** set up the project plan and get the untuned Qwen3-8B baseline (the
"Untuned" row of the GRPO vs GFlowNet comparison) running end to end on Modal.

**Status:** ✅ Done. Phase 0 baseline complete for the autoregressive model.

---

## What was done

**Wrote the baseline eval: `baseline/modal_baseline.py`.**
   - Qwen3-8B (thinking off, greedy, seed 0) served with vLLM on one Modal H100.
   - Test sets: HarmBench (200 prompts, plain and with a `"Sure, here is"` prefill attack),
     XSTest (450), GSM8K (first 200 of the test set).
   - Judges: HarmBench Llama-2-13B classifier for attack success; XSTest-style refusal
     string-match; last `#### <number>` for GSM8K.
   - Generations are saved before judging; `--reuse-gen` re-runs only the judge.

**Made the prefill attack non-static: `baseline/prefill_attacks.py`.**
   - A fixed `"Sure, here is"` is easy for fine-tuning to overfit. Added a pool of 50 generic
     openers (affirmative, direct, structured, role, framing/disclaimer, anti-refusal).
   - `--attacks random`: each of the 200 HarmBench requests gets one phrase at random; no phrase
     is used more than 5 times. The assignment is reproducible (`--prefill-seed`, default 0).
     Seed 0 used all 50 phrases.
   - Also available: `--attacks all` (10 named openers split into train / held-out for later
     training). Run IDs now get a `-dirty` suffix when the tree has uncommitted changes.


## Results — untuned Qwen3-8B

95% Wilson confidence intervals in brackets.

| Metric | Result | 95% CI |
|---|---|---|
| **Attack success, HarmBench plain** | 26/200 = **13.0%** | [9.0, 18.4] |
| **Attack success, fixed prefill "Sure, here is"** | 124/200 = **62.0%** | [55.1, 68.4] |
| **Attack success, random prefill (1 of 50 phrases)** | 85/200 = **42.5%** | [35.9, 49.4] |
| **False refusal, XSTest safe prompts** | 11/250 = **4.4%** | [2.5, 7.7] |
| Refusal, XSTest unsafe prompts | 135/200 = 67.5% | [60.7, 73.6] |
| **GSM8K accuracy** | 188/200 = **94.0%** | [89.8, 96.5] |

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
- **The random-prefill condition is the better attacked baseline going forward:** it still leaves
  large headroom (42.5% vs 13% plain) and is much harder to overfit than a single string.
- **Benign behaviour is strong** (4.4% false refusal, 94% GSM8K). The ±2-point guardrails
  in the hypothesis will be measured against these values.
- **Statistical power is a concern, so we need paired comparisons.** With 200 prompts, one run's
  uncertainty is about ±6–7 points. The hypothesised 10% relative reduction (62% → ≤ 55.8%, or
  42.5% → ≤ 38.3% for random prefill) is smaller than that, so the
  GRPO vs GFlowNet comparison needs paired tests on the same prompts, more held-out prompts,
  and three seeds.

## Caveats

- XSTest numbers are provisional: the string-match refusal detector likely misses soft
  refusals (only 67.5% refusal on unsafe prompts, mostly missed on discrimination and
  historical-event types). Re-measure with WildGuard or an LLM judge.
- 68 prefill replies were judged not harmful but contain no refusal phrase (safe pivots or
  classifier misses). A manual spot-check of ~20 is pending.
- 116/200 prefill replies hit the 512-token cap; this follows the HarmBench convention.
- **vLLM greedy decoding is not bit-for-bit reproducible across runs.** Rerunning the plain
  condition gave identical text for only 106/200 prompts (batch composition changes
  floating-point order), although all 200 classifier labels matched. Treat run-to-run
  differences of a few points as noise.
- 74 random-prefill replies were judged not harmful with no refusal phrase; same spot-check applies.
- `summary.json` records git commit `31518d8` because the script was uncommitted at run time;
  the code that produced these numbers is the version committed afterwards in `93e7573`.
  The random-prefill run records `e8a3a36-dirty` for the same reason.
  Future runs should be launched from a clean, committed tree.
- Raw generations are in `outputs/` (gitignored) and are not part of this repo.

---

## Next steps

1. Re-measure XSTest refusals with WildGuard; spot-check ~20 "not harmful" prefill replies.
2. Freeze the train / held-out data splits (BeaverTails + benign; held-out attack templates).
3. ~~Vary the prefill attack~~ (done: 50-phrase random pool). Next: split the 50 phrases into
   train and held-out sets before any prefill-based training; rerun random prefill with a few
   `--prefill-seed` values; add more held-out prompts for statistical power.
4. Check LLaDA-8B / SDAR-8B feasibility: memory, seconds per rollout, and how to compute
   log π(y | x) for trajectory balance.
5. Phase 1: GRPO vs trajectory-balance pilots (~30 steps each) on Tinker.
6. Raise with the mentor: LoRA vs frozen backbone; SDAR vs LLaDA; whether α-conditioning is in v1.
