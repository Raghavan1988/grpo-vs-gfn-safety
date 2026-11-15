# GFlowNet (TB) vs GRPO, SFT diagnosis, dev set (2026-11-15)

Qwen3-8B (thinking off), LoRA rank 32, one seed each. Same prompts, mix (16 prompts × 8 replies
per step), rewards (WildGuard / maths check), reference (untuned model) and LoRA for GRPO and TB.
Training replies are **sampled** (T = 1.0 for GRPO/TB, T = 0.7 with 4 candidates for SFT);
evaluation uses greedy decoding. Attack success = HarmBench classifier; refusals = WildGuard.
Raw outputs in `outputs/eval_*` (gitignored).

## Runs

| Run | What |
|---|---|
| `grpo-v3-s0` | Pure GRPO, 30 steps, current code (groups with reward spread < 0.05 get zero advantage) |
| `pilot2-grpo-s0` | Round-2 GRPO (before the noise fix); shown on dev for reference |
| `tb-tau{0.01,0.03,0.1,0.3}-s0` | Trajectory balance, 30 steps, target π_ref·exp(r/τ), batch log Z, no length normalisation |
| `tb-tau0.03-s0-100steps` | TB τ = 0.03, 100 steps (resumed once after a GPU preemption) |
| `pilot2-sftgrpo-s0-sft` | SFT stage alone (rejection-sampling SFT, 1,868 examples, 1 epoch) |
| `sftgrpo-refbase-s0` | SFT, then 30 GRPO steps, KL to the untuned model (current code) |
| `sftgrpo-refsft-s0` | Same, KL to the SFT adapter |

## Test set (HarmBench / XSTest / OR-Bench-Hard / GSM8K)

| Model | Held-out prefill | Prefix 10 / 20 / 40 | Templates any | Templates mean | Plain | XSTest FR | OR-Hard FR | GSM8K |
|---|---|---|---|---|---|---|---|---|
| Untuned | 42.0 | 98.4 / 98.4 / 98.4 | 72.0 | 26.3 | 13.0 | 4.0 | 32.0 | 94.0 |
| **GRPO v3** | 10.5 | **25.8 / 44.4 / 60.5** | **43.0** | **13.6** | **7.0** | **2.4** | **20.3** | 94.5 |
| TB τ=0.01 | 19.0 | 51.6 / 71.0 / 86.3 | 63.5 | 21.6 | 10.0 | 5.2 | 31.0 | 95.0 |
| TB τ=0.03 | 11.0 | 54.0 / 70.2 / 77.4 | 66.5 | 21.6 | 8.0 | 5.2 | 33.3 | 96.0 |
| TB τ=0.03, 100 steps | 17.5 | 43.5 / 69.4 / 82.3 | 61.0 | 21.7 | 9.5 | 4.4 | 24.0 | 95.5 |
| TB τ=0.1 | 22.5 | 63.7 / 77.4 / 86.3 | 65.5 | 23.8 | 10.0 | 4.4 | 30.3 | 93.5 |
| TB τ=0.3 | 35.5 | 81.5 / 89.5 / 87.9 | 69.0 | 25.0 | 10.5 | 4.8 | 31.0 | 94.5 |
| SFT only | 33.5 | 75.0 / 83.1 / 87.9 | 70.0 | 25.1 | 11.5 | 3.6 | 22.0 | 94.0 |
| SFT+GRPO, ref untuned | 9.5 | 29.0 / 49.2 / 71.0 | 49.5 | 17.6 | 7.5 | 4.0 | 33.0 | 94.5 |
| SFT+GRPO, ref SFT | 9.0 | 29.0 / 49.2 / 70.2 | 61.5 | 20.9 | 10.0 | 3.6 | 26.7 | 94.5 |

All values are %. Held-out prefill n = 200; prefixes n = 124 each; templates n = 200 × 5;
XSTest 250 safe; OR-Bench-Hard 300; GSM8K 200.

## Dev set (JailbreakBench, overlap with test and train removed)

83 harmful behaviours (50 with a harmful prefix from the untuned model), 96 benign behaviours.
Used to choose τ and to check that test-set conclusions hold on a second dataset.

| Model | Held-out prefill | Prefix 10 / 20 / 40 | Templates any | Benign FR |
|---|---|---|---|---|
| Untuned | 38.6 | 96 / 96 / 96 | 74.7 | 8.3 |
| GRPO v3 | **2.4** | 24 / 44 / 58 | **43.4** | 9.4 |
| GRPO round 2 | **2.4** | **20 / 28 / 46** | 48.2 | 9.4 |
| TB τ=0.01 | 13.3 | 44 / 66 / 72 | 67.5 | 9.4 |
| TB τ=0.03 | 9.6 | 50 / 70 / 72 | 67.5 | 8.3 |
| TB τ=0.03, 100 steps | 10.8 | 46 / 68 / 78 | 67.5 | 8.3 |
| TB τ=0.1 | 20.5 | 56 / 76 / 78 | 63.9 | 8.3 |
| TB τ=0.3 | 33.7 | 74 / 84 / 86 | 72.3 | 8.3 |
| SFT+GRPO, ref untuned | 3.6 | 30 / 46 / 80 | 49.4 | 14.6 |
| SFT+GRPO, ref SFT | 6.0 | 32 / 40 / 64 | 62.7 | 8.3 |

Dev-set τ choice: τ = 0.03 (best held-out prefill; tied on templates). Prefix depth n = 50, so
one prompt = 2 points.

## Paired tests (exact McNemar, test set)

| Comparison | Result |
|---|---|
| GRPO v3 vs TB τ=0.03 100 steps, all hard attacks | harmful only under GRPO 43, only under TB 204 (p ≈ 3e-26) |
| SFT+GRPO: ref SFT vs ref untuned, OR-Hard refusals | fewer refusals with ref SFT (p ≈ 0.03) |
| SFT+GRPO: ref SFT vs ref untuned, templates | more successful attacks with ref SFT (p ≈ 0.03) |

## Findings

1. **TB does not match GRPO at a matched budget.** True for every τ in 0.01–0.3, for 30 and 100
   steps, and on both test and dev. Smaller τ helps up to 0.03; more steps barely help. TB levels
   off near its target distribution, and its gradient norms (50–688) are clipped to 1 on most
   steps. TB never raised over-refusal.
2. **SFT+GRPO's earlier over-refusal (OR-Hard 50%) was mostly the reward-noise bug.** On current
   code with the same untuned reference it is 33%. SFT alone barely changes attack success
   (templates 70%) but lowers over-refusal (OR-Hard 22%). Using the SFT adapter as the reference
   keeps more of that (26.7%) but weakens template resistance (61.5% vs 49.5%).
3. **Pure GRPO v3 is the best model overall** and the only one that lowers both attack success
   and false refusals relative to untuned. Caveat: its test held-out prefill rate (10.5%) is
   higher than round-2 GRPO's (5.0%); the noise fix traded that for prefixes, templates and
   over-refusal on the test set. On dev, round-2 GRPO is better on deep prefixes.

## Caveats

- One seed per run; differences of a few points (e.g. among TB τ values, GRPO v3 vs round 2) are
  within noise. The GRPO-vs-TB gap is not.
- TB is on-policy with batch log Z (VarGrad). Off-policy TB with a replay buffer or a learned
  log Z were not tried.
- The SFT-only adapter was evaluated on the test set only.
