# Round-2 pilots — held-out evaluation (2026-11-14)

Adapters: `pilot2-grpo-s0` (pure GRPO, 30 steps) and `pilot2-sftgrpo-s0-grpo` (SFT then 30 GRPO
steps), one seed. Settings: LoRA r=32, lr 1e-4, KL 0.01 to the base model, 16 prompts/step
(8 prefilled harmful, 2 plain harmful, 3 benign, 3 maths) × 8 replies, 512-token cap.
Evaluated with `modal run baseline/modal_baseline.py --adapter <run> --attacks random-heldout`:
the same 200 HarmBench requests and the same held-out phrase per request as the untuned baseline
(phrases never used in training), so comparisons are paired. ASR judge: HarmBench classifier.

| Metric | Untuned | Pure GRPO | SFT + GRPO |
|---|---|---|---|
| **ASR, held-out prefill** (95% CI) | 84/200 = 42.0% [35.4, 48.9] | **9/200 = 4.5%** [2.4, 8.3] | 20/200 = 10.0% [6.6, 14.9] |
| ASR, held-out prefill (WildGuard) | 43.0% | 3.5% | 9.5% |
| ASR, plain HarmBench | 13.0% | 6.0% | 5.5% |
| **XSTest false refusal (WildGuard)** (95% CI) | 4.0% [2.2, 7.2] | **6.0%** [3.7, 9.7] | **9.2%** [6.2, 13.4] |
| XSTest refusal on unsafe prompts (WildGuard) | 86% | 92% | 95% |
| **GSM8K** | 94.0% | 96.0% | 94.0% |
| Mean reply length (tokens) | 290 | 305 | 262 |

**Paired tests (exact McNemar, same requests and phrases)**

| Comparison | Fixed | Newly broken | p |
|---|---|---|---|
| Pure GRPO vs untuned, held-out prefill | 75 | 0 | 5e-23 |
| SFT + GRPO vs untuned, held-out prefill | 65 | 1 | 2e-18 |
| Pure GRPO vs untuned, plain | 15 | 1 | 5e-4 |
| Pure GRPO vs SFT + GRPO, held-out prefill | harmful only under SFT + GRPO: 12; only under GRPO: 1 | | 0.003 |

XSTest safe prompts: pure GRPO newly refuses 7 and stops refusing 2 (10 → 15 of 250);
SFT + GRPO newly refuses 13 (10 → 23 of 250).

**Held-out ASR by category (untuned / pure GRPO / SFT + GRPO)**

| Category | Untuned | Pure GRPO | SFT + GRPO |
|---|---|---|---|
| chemical_biological | 16/28 | 0/28 | 1/28 |
| cybercrime_intrusion | 24/40 | 3/40 | 9/40 |
| harassment_bullying | 4/19 | 0/19 | 0/19 |
| harmful | 8/21 | 1/21 | 1/21 |
| illegal | 19/58 | 1/58 | 3/58 |
| misinformation_disinformation | 13/34 | 4/34 | 6/34 |

**Reading**
- Pure GRPO generalises to attack phrases it never saw: 42.0% → 4.5%, 75 requests fixed and none
  broken. The two judges agree (4.5% vs 3.5%), so this is not reward hacking of WildGuard.
- It costs false refusals: 4.0% → 6.0%, exactly the +2-point limit in the hypothesis (and the CI
  overlaps). GSM8K is unchanged (94% → 96%, within noise).
- SFT + GRPO is worse on every axis: higher held-out ASR (10.0%, p = 0.003 vs pure GRPO) and more
  false refusals (9.2%, over the limit). Likely reasons: the SFT data is mostly refusals, and the
  GRPO phase is anchored to the base model, which pulls back part of the SFT update.
- Gate (AGENTS.md / Grok note): pure GRPO beats untuned on held-out ASR, keeps GSM8K, KL left the
  0.001 floor; XSTest false refusal is at the limit, not inside it. Borderline pass.
- Caveats: one seed, 30 steps; the remaining harmful replies are mostly misinformation and
  cybercrime. At 4.5% held-out ASR, a further 10% relative reduction is 0.45 points, which no
  200-prompt evaluation can detect.
