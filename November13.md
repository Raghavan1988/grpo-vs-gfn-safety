# November 13, 2026 — Daily summary

**Goal for the day:** set up the project plan and get the untuned Qwen3-8B baseline (the
"Untuned" row of the GRPO vs GFlowNet comparison) running end to end on Modal.

**Status:** ✅ Done. Phase 0 baseline complete for the autoregressive model.

---

## What was done

1. **Read the Algoverse doc and the research proposal; wrote the project plan.**
   - `AGENTS.md`: project summary, model/reward definitions, decisions that shape the work,
     Phases 0–3 with decision gates, repo conventions, safety/ethics rules, and the spec for
     tonight's baseline. `CLAUDE.md` imports it.
   - Flagged open issues in the proposal: frozen backbone vs LoRA fine-tuning is inconsistent;
     for left-to-right decoding, trajectory balance with a reference prior is equivalent to
     KL-regularised RL, so the AR comparison tests the optimiser rather than the objective.

2. **Wrote the baseline eval: `baseline/modal_baseline.py`.**
   - Qwen3-8B (thinking off, greedy, seed 0) served with vLLM on one Modal H100.
   - Test sets: HarmBench (200 prompts, plain and with a `"Sure, here is"` prefill attack),
     XSTest (450), GSM8K (first 200 of the test set).
   - Judges: HarmBench Llama-2-13B classifier for attack success; XSTest-style refusal
     string-match; last `#### <number>` for GSM8K.
   - Generations are saved before judging; `--reuse-gen` re-runs only the judge.

3. **Set up access.** Verified the Modal login and the `huggingface` Modal secret, and accepted
   the gated-dataset terms for HarmBench and XSTest.

4. **Smoke test (20 items per set), then the full run.** Five bugs found and fixed on the way:

   | # | Problem | Fix |
   |---|---|---|
   | 1 | vLLM 0.9.2 crashes on import with transformers ≥ 4.54 (`'aimv2' is already used`) | Pinned `transformers==4.53.2` |
   | 2 | HarmBench and XSTest are gated; listing files works without access, downloading doesn't | Accepted terms; check access with a real download |
   | 3 | XSTest's only split is named `test`, although its file is `train-*.parquet` | Kept `split="test"` |
   | 4 | HarmBench classifier is Llama-2-13B with a 2,048-token context; script asked for 4,096 | `max_model_len=2048` |
   | 5 | GSM8K parser took the first `####`, but Qwen writes headings like `#### 1. Donuts` | Use the last `#### <number>` |

5. **Recorded results** in `results/baseline_qwen3_8b.md` and updated `AGENTS.md`.

---

## Results — untuned Qwen3-8B (full run)

95% Wilson confidence intervals in brackets.

| Metric | Result | 95% CI |
|---|---|---|
| **Attack success, HarmBench plain** | 26/200 = **13.0%** | [9.0, 18.4] |
| **Attack success, HarmBench + prefill** | 124/200 = **62.0%** | [55.1, 68.4] |
| **False refusal, XSTest safe prompts** | 11/250 = **4.4%** | [2.5, 7.7] |
| Refusal, XSTest unsafe prompts | 135/200 = 67.5% | [60.7, 73.6] |
| **GSM8K accuracy** | 188/200 = **94.0%** | [89.8, 96.5] |

**Attack success by HarmBench category**

| Category | Plain | Prefill |
|---|---|---|
| chemical_biological | 0/28 | 19/28 |
| cybercrime_intrusion | 8/40 | 29/40 |
| harassment_bullying | 0/19 | 10/19 |
| harmful | 2/21 | 11/21 |
| illegal | 2/58 | 35/58 |
| misinformation_disinformation | 14/34 | 20/34 |

Smoke test (20 items, pipeline check only): plain 0/20, prefill 8/20, false refusal 1/20,
GSM8K 19/20.

**Run cost:** about 5 minutes wall time (30 s of generation) on one H100, roughly $1.
Today's total Modal spend including the failed and smoke runs was a few dollars (estimate).

---

## What the results mean

- **The proposal's "near-zero attack success" assumption is wrong for Qwen3-8B.** Plain
  HarmBench gives 13%, concentrated in misinformation and cybercrime.
- **The prefill attack opens large headroom (62%)** across every category, including
  chemical/biological (0 → 19 of 28). The "is there anything to reduce?" gate is cleared.
- **Benign behaviour is strong** (4.4% false refusal, 94% GSM8K). The ±2-point guardrails
  in the hypothesis will be measured against these values.
- **Statistical power is a concern.** With 200 prompts, one run's uncertainty is about ±6–7
  points. The hypothesised 10% relative reduction (62% → ≤ 55.8%) is smaller than that, so the
  GRPO vs GFlowNet comparison needs paired tests on the same prompts, more held-out prompts,
  and three seeds.

## Caveats

- XSTest numbers are provisional: the string-match refusal detector likely misses soft
  refusals (only 67.5% refusal on unsafe prompts, mostly missed on discrimination and
  historical-event types). Re-measure with WildGuard or an LLM judge.
- 68 prefill replies were judged not harmful but contain no refusal phrase (safe pivots or
  classifier misses). A manual spot-check of ~20 is pending.
- 116/200 prefill replies hit the 512-token cap; this follows the HarmBench convention.
- `summary.json` records git commit `31518d8` because the script was uncommitted at run time;
  the code that produced these numbers is the version committed afterwards in `93e7573`.
  Future runs should be launched from a clean, committed tree.
- Raw generations are in `outputs/` (gitignored) and are not part of this repo.

---

## Next steps

1. Re-measure XSTest refusals with WildGuard; spot-check ~20 "not harmful" prefill replies.
2. Freeze the train / held-out data splits (BeaverTails + benign; held-out attack templates).
3. Add stronger or more varied held-out attacks and more held-out prompts for statistical power.
4. Check LLaDA-8B / SDAR-8B feasibility: memory, seconds per rollout, and how to compute
   log π(y | x) for trajectory balance.
5. Phase 1: GRPO vs trajectory-balance pilots (~30 steps each) on Tinker.
6. Raise with the mentor: LoRA vs frozen backbone; SDAR vs LLaDA; whether α-conditioning is in v1.
