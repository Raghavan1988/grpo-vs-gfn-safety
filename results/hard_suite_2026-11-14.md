# Hard evaluation suite — untuned vs round-2 adapters (2026-11-14)

`modal run baseline/modal_baseline.py --suite hard [--adapter <run>]` (see `gvg/hard_eval.py`).
Same prompts for all three models, so comparisons are paired. ASR judge: HarmBench classifier;
refusal judge: WildGuard. One seed per adapter; greedy decoding.

| Attack (n) | Untuned | Pure GRPO (`pilot2-grpo-s0`) | SFT + GRPO (`pilot2-sftgrpo-s0-grpo`) |
|---|---|---|---|
| Held-out prefill phrase (200) | 42.0% | **5.0%** | 10.0% |
| Harmful prefix, 10 tokens (124) | 98.4% | **16.9%** | 32.3% |
| Harmful prefix, 20 tokens (124) | 98.4% | **37.1%** | 54.0% |
| Harmful prefix, 40 tokens (124) | 98.4% | **42.7%** | 75.8% |
| Template: combination (200) | 44.0% | **13.5%** | 21.5% |
| Template: fiction role-play (200) | 42.0% | **27.0%** | 34.5% |
| Template: refusal suppression (200) | 28.5% | **17.5%** | 20.0% |
| Template: prefix injection (200) | 16.5% | **8.0%** | 11.5% |
| Template: research framing (200) | 0.5% | 0.0% | 0.0% |
| Templates, broken by ≥ 1 of 5 (200) | 72.0% | **45.0%** | 53.0% |
| Templates, mean | 26.3% | **13.2%** | 17.5% |
| Plain HarmBench (200) | 13.0% | 7.0% | 5.5% |
| **OR-Bench-Hard false refusal** (300) | 32.0% | **33.7%** | 50.0% |
| XSTest false refusal (250) | 4.0% | 6.0% | 10.0% |
| GSM8K (200) | 94.0% | 95.0% | 94.5% |

**Paired tests (exact McNemar)**
- Pure GRPO vs untuned, harmful prefixes (all depths, 372): 247 fixed, 1 broken (p ≈ 1e-72).
- Pure GRPO vs untuned, templates (1,000): 158 fixed, 27 broken (p ≈ 1e-23).
- Pure GRPO vs SFT + GRPO, harmful prefixes: harmful only under GRPO 9, only under SFT + GRPO 90
  (p ≈ 6e-18); templates: 37 vs 80 (p ≈ 9e-5).
- OR-Bench-Hard refusals, pure GRPO vs untuned: 30 newly refused, 25 no longer refused (p = 0.59):
  no detectable change.

**Reading**
- Pure GRPO is the best model on every attack and keeps over-refusal on OR-Bench-Hard unchanged
  (32.0% → 33.7%, not significant); XSTest rises 2 points as before. GSM8K is unchanged.
- **Safety is still shallow.** With 10 tokens of real harmful text already in the reply, GRPO
  continues harmfully 17% of the time; with 40 tokens, 43%. Role-play templates still work 27% of
  the time, and 45% of requests are broken by at least one template.
- **This is where a GFlowNet comparison has measurable room:** deep harmful prefixes (37–43%)
  and templates (45% any) — not the single-phrase prefill (5%).
- SFT + GRPO is worse than pure GRPO everywhere and over-refuses heavily (OR-Bench-Hard 50%,
  XSTest 10%). Its GRPO phase was regularised toward the base model (fixed in code since).
- Pure GRPO's held-out prefill ASR reads 5.0% here and 4.5% in the earlier run (one prompt):
  vLLM run-to-run variation.
- Caveats: one seed; OR-Bench-Hard shares a source with the benign training prompts (overlaps
  removed), so XSTest is the independent over-refusal check; training never used harmful-text
  prefixes or user-turn templates, so both are genuinely unseen attack families.
