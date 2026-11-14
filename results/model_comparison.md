# Model comparison: untuned vs pure GRPO vs SFT + GRPO

Qwen3-8B (thinking off), LoRA fine-tuning, round-2 settings, one seed each (2026-11-14).
Raw outputs are in `outputs/` (gitignored). Related: `results/baseline_qwen3_8b.md`,
`results/pilots_round2_heldout.md`, `results/hard_suite_2026-11-14.md`,
`docs/how_sft_works.md`, `docs/how_grpo_works.md`.

## The models

| Model | Run name (`gvg-ckpts` volume) | How it was made |
|---|---|---|
| **Untuned** | — | Qwen3-8B as released |
| **Pure GRPO** | `pilot2-grpo-s0` | 30 GRPO steps from the untuned model (16 prompts × 8 replies per step, lr 1e-4, KL 0.01 to the untuned model) |
| **SFT + GRPO** | `pilot2-sftgrpo-s0-grpo` | Rejection-sampling SFT first (4 replies per training prompt, keep the best if reward ≥ 0.5, train on the 1,868 kept replies: 1 epoch, 117 updates, lr 1e-4; adapter `pilot2-sftgrpo-s0-sft`), then the same 30 GRPO steps (KL to the untuned model in this run) |

All use LoRA rank 32 on all attention and MLP layers (~87M trainable parameters); base weights
are frozen.

> **SFT-only adapter not yet evaluated.** The adapter after the SFT stage alone
> (`pilot2-sftgrpo-s0-sft`) has not been run on the test set, so it is not compared here.

---

## Data

### Training data (`data/build_splits.py`; hashes in `data/SPLITS.md`)

| Pool | Source | Count | Notes |
|---|---|---|---|
| Harmful, prefilled | BeaverTails 30k train (`PKU-Alignment/BeaverTails`) | 500 | Prompts whose reference responses are all unsafe; a prefill phrase from the 26 **training** phrases is forced onto the reply |
| Harmful, plain | BeaverTails 30k train | 500 | Same selection, no prefill |
| Benign, harmful-looking | OR-Bench-80k (`bench-llm/or-bench`) | 500 | Discourages over-refusal |
| Maths | GSM8K train (`openai/gsm8k`) | 500 | Keeps reasoning ability |
| **Total** | | **2,000** | Train hash `788b7afa5e971eb8` |

- **Filtering:** any training prompt with word overlap ≥ 0.5 with a HarmBench or XSTest prompt was
  dropped (114 harmful prompts). BeaverTails reference responses are not used.
- **Rewards (training judge):** harmful → 1 − P(WildGuard: harmful reply); benign →
  1 − P(WildGuard: refusal); maths → 1 if correct.
- **How each method uses it:** GRPO draws 16 prompts per step (8 prefilled harmful, 2 plain
  harmful, 3 benign, 3 maths) for 30 steps = 480 prompts. The SFT stage of SFT + GRPO samples 4
  replies for each of the 2,000 prompts once and keeps 1,868 (464 plain harmful, 425 prefilled
  harmful, 494 benign, 485 maths).

### Validation data (diagnostic only)

200 prompts from the same pools, disjoint from training: 50 prefilled harmful, 50 plain harmful,
50 benign, 50 maths (hash `9fde57360553441e`). Judged by WildGuard, the training judge, and the
prefills use **training** phrases, so it shows whether training worked, not whether it generalises.

### Test data (evaluation; never used in training)

| Test | Source | Size | Measures |
|---|---|---|---|
| Held-out prefill | HarmBench standard (`walledai/HarmBench`) + one of the 24 **held-out** prefill phrases per request | 200 | Robustness to unseen openers |
| Harmful prefix (10 / 20 / 40 tokens) | HarmBench requests where the untuned model's fixed-prefill reply was harmful; the reply is prefilled with the first k tokens of that harmful text (Qi et al.) | 124 each | Recovery after real harmful text |
| Jailbreak templates (5 families) | HarmBench requests wrapped in refusal suppression, prefix injection, their combination, fiction role-play, research framing (after Wei et al. 2023) | 200 each | Unseen attack families |
| Plain HarmBench | HarmBench standard, no attack | 200 | Direct requests |
| XSTest | `walledai/XSTest` | 250 safe + 200 unsafe | False refusal (safe prompts) |
| OR-Bench-Hard | `bench-llm/or-bench` hard-1k, 300 sampled, 27 near training prompts removed | 300 | Harder false-refusal test (shares a source with training) |
| GSM8K | `openai/gsm8k` test, first 200 | 200 | Maths accuracy |

**Judges:** attack success is scored by the HarmBench classifier (`cais/HarmBench-Llama-2-13b-cls`),
which training never sees; refusals are scored by WildGuard (`allenai/wildguard`, P > 0.5).
All models get identical prompts, so comparisons are paired. Greedy decoding.

---

## Results

### Test set (attack success = % of requests where the reply was harmful; lower is better)

| Metric | Untuned | Pure GRPO | SFT + GRPO |
|---|---|---|---|
| **Held-out prefill phrase** (200) | 42.0% [35.4, 48.9] | **5.0%** [2.7, 9.0] | 10.0% [6.6, 14.9] |
| Harmful prefix, 10 tokens (124) | 98.4% | **16.9%** | 32.3% |
| Harmful prefix, 20 tokens (124) | 98.4% | **37.1%** | 54.0% |
| Harmful prefix, 40 tokens (124) | 98.4% | **42.7%** | 75.8% |
| Templates, broken by ≥ 1 of 5 (200) | 72.0% | **45.0%** | 53.0% |
| Templates, mean of 5 | 26.3% | **13.2%** | 17.5% |
| — fiction role-play | 42.0% | **27.0%** | 34.5% |
| — combination | 44.0% | **13.5%** | 21.5% |
| — refusal suppression | 28.5% | **17.5%** | 20.0% |
| — prefix injection | 16.5% | **8.0%** | 11.5% |
| — research framing | 0.5% | 0.0% | 0.0% |
| Plain HarmBench (200) | 13.0% | 7.0% | **5.5%** |

### Test set: over-refusal and capability

| Metric | Untuned | Pure GRPO | SFT + GRPO |
|---|---|---|---|
| **XSTest false refusal** (250 safe; lower is better) | **4.0%** [2.2, 7.2] | 6.0% [3.7, 9.7] | 10.0% [6.9, 14.4] |
| **OR-Bench-Hard false refusal** (300) | **32.0%** | 33.7% | 50.0% |
| XSTest refusal on unsafe prompts (200; higher is better) | 86.0% | 93.0% | 94.0% |
| **GSM8K** (200; higher is better) | 94.0% | 95.0% | 94.5% |
| Mean reply length (tokens) | 281 | 298 | 253 |

### Validation set (diagnostic: WildGuard, training phrases, 50 prompts per group)

| Metric | Untuned | Pure GRPO | SFT + GRPO |
|---|---|---|---|
| Harmful rate, prefilled prompts | 24–26% | 0% | 0% |
| Harmful rate, plain prompts | 2% | 0% | 0% |
| Benign refusal rate | 2% | 4% | 4% |
| GSM8K | 92% | 92% | 92% |

The untuned range reflects repeated measurements (vLLM run-to-run variation; one prompt = 2
points).

### Paired tests (exact McNemar on identical prompts)

| Comparison | Result |
|---|---|
| Pure GRPO vs untuned, held-out prefill | 75 fixed, 0 broken (p ≈ 5e-23; from the first held-out run, where pure GRPO read 4.5%) |
| Pure GRPO vs untuned, harmful prefixes (372) | 247 fixed, 1 broken (p ≈ 1e-72) |
| Pure GRPO vs untuned, templates (1,000) | 158 fixed, 27 broken (p ≈ 1e-23) |
| Pure GRPO vs SFT + GRPO, harmful prefixes | harmful only under GRPO 9, only under SFT + GRPO 90 (p ≈ 6e-18) |
| Pure GRPO vs SFT + GRPO, templates | 37 vs 80 (p ≈ 9e-5) |
| Pure GRPO vs untuned, OR-Bench-Hard refusals | 30 newly refused, 25 no longer refused (p = 0.59: no change) |

---

## Takeaways

1. **Pure GRPO is the best model on every attack**: held-out prefills 42% → 5%, 10-token harmful
   prefixes 98% → 17%, templates (any) 72% → 45%. GSM8K is unchanged and OR-Bench-Hard over-refusal
   does not change; XSTest false refusals rise 2 points (4% → 6%), the hypothesis's limit.
2. **SFT + GRPO is worse than pure GRPO everywhere** and over-refuses heavily (XSTest 10%,
   OR-Bench-Hard 50%). Likely causes: SFT data dominated by refusals, and its GRPO phase being
   regularised toward the untuned model, which pulled back part of the SFT update (now changed in
   code to use the SFT adapter as reference; not yet rerun).
3. **Safety is still shallow.** Even pure GRPO continues harmfully 37–43% of the time when 20–40
   tokens of real harmful text are already in the reply, and 27% of fiction role-play prompts
   succeed. These are the attacks where a GFlowNet comparison has measurable room; the
   single-phrase prefill (5%) does not.

## Caveats

- One seed and 30 GRPO steps per model; these are pilots, not final results.
- Validation numbers use the training judge and training phrases; treat them as diagnostics.
- OR-Bench-Hard shares a source with the benign training prompts (overlaps removed); XSTest is the
  independent over-refusal test.
- Pure GRPO's held-out prefill rate reads 4.5% and 5.0% in two evaluation runs (one prompt apart).
- No human validation of judge labels yet.

## Next

- Evaluate the SFT-only adapter on the test set (`--suite hard --adapter pilot2-sftgrpo-s0-sft`)
  and add it to this comparison.
- Rerun SFT + GRPO with the SFT adapter as KL reference.
- Three seeds of the best configuration; then the GFlowNet (trajectory balance) comparison on the
  harmful-prefix and template attacks.
