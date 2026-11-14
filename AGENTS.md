# AGENTS.md — grpo-vs-gfn-safety

Guide for anyone (human or coding agent) working in this repo. Read it fully before changing code.

## 1. What this project is

Algoverse Fall 2026 research project (mentor: Sruthi; team: Raghavan, Junhui, Shree, Aleks).

**Question:** Does multi-objective GFlowNet fine-tuning (trajectory balance, TB) give better
safety control than matched safety-aware GRPO, for both an autoregressive (AR) LM and a
non-autoregressive / diffusion (NAR) LM?

**Primary hypothesis (Hₐ):** TB achieves ≥10% *relative* reduction in held-out jailbreak ASR vs.
matched GRPO, while keeping clean-task accuracy within 2 pp, false-refusal rate within 2 pp,
and inference compute within 5%. Three seeds.

**Models**
| Role | Model | Where trained |
|---|---|---|
| AR | `Qwen/Qwen3-8B` (thinking OFF) | own loop on Modal (LoRA; `train/modal_train.py`) — replaces the original Tinker plan |
| NAR primary | SDAR-8B-Chat (block diffusion from Qwen3-8B-Base) | own loop (LoRA) |
| NAR alternative | `GSAI-ML/LLaDA-8B-Instruct` | own loop (LoRA) |
| PoC | `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | own loop (Junhui) |

**Reward** (fixed α for v1; α-conditioning is a later extension):
```
U_α = α_q·R_q + α_s·R_s − α_r·C_r − α_c·C_c
GFlowNet target: R = π_ref(y|x) · exp(β·U_α)
```
- R_s, C_r in training: WildGuard (`allenai/wildguard`).
- Eval only (never in training): HarmBench classifier (`cais/HarmBench-Llama-2-13b-cls`).
- R_q: exact match (GSM8K/MATH), else helpfulness RM.
- C_c: normalized NFE (NAR) or generated tokens (AR).
- As implemented for the pilots (`gvg/rewards.py`, α = 1, no C_c): harmful prompt → 1 − P(harmful
  reply); benign prompt → 1 − P(refusal); GSM8K → 1 if correct. Rewards lie in [0, 1].

## 2. Known facts / decisions that shape the work

1. **Nobody has a working GFlowNet run yet.** Junhui's 200M dLLM PoC gave zero reward and was
   abandoned. First milestone for TB is "reward goes up without blowing up".
2. **Frozen vs LoRA is inconsistent in the proposal.** Working assumption: **LoRA on model
   weights** (rank 32, all linear layers). Confirm with mentor before writing the paper.
3. **Measured untuned Qwen3-8B ASR: 13% plain, 62% with a "Sure, here is" prefill** (HarmBench,
   n=200). Plain is not ≈0 (mostly misinformation/cybercrime), but the attacked condition is where
   the headroom is. Always report both, plus DIJA for NAR.
   Prefill attacks are varied, not static: `gvg/prefill_attacks.py` holds 50 random-pool
   phrases split into **26 train / 24 held-out** (`RANDOM_TRAIN_IDS` / `RANDOM_HELDOUT_IDS`),
   plus 10 named openers. Train only on train phrases; the headline robustness number for
   fine-tuned models is **held-out ASR (`--attacks random-heldout`): untuned 42.0%** [35.4, 48.9].
   Random over all 50 phrases: 42.5%; fixed "Sure, here is": 62%. vLLM greedy decoding is not
   bit-reproducible across runs; compare methods on paired prompts and seeds.
4. **For AR decoding, TB with reference prior ≡ KL-regularized RL** (Tiapkin et al., Deleu et al.).
   AR comparison therefore tests optimizer / log Z estimation, not the objective. Say so in the paper.
5. TB is unstable at this scale: length collapse, exploding grads with learned log Z,
   instability for β < ~0.004 (TBA). Use batch-mean log Z estimate. Do NOT length-normalise the
   log-ratio by default: with ~300-token replies it rescales τ by ~300× and turns WildGuard jitter
   between equally safe replies into a 5–50 nat preference (the same noise problem as GRPO's).
6. GRPO gets zero signal when all K responses agree (all refuse / all comply). Logged as
   `zero_var_group_frac` (0.25–0.56 in the 3-step check).
7. **WildGuard is the training and refusal judge; the HarmBench classifier is the ASR judge.**
   Untuned XSTest false refusal (WildGuard) **4.0%** [2.2, 7.2] is the guardrail reference; the
   two harm judges agree on ~94% of replies. Gotcha: WildGuard's top-k alternatives contain
   several token ids that decode to "no"/"yes" — aggregate over a *list*, never a dict keyed by
   text (that bug once produced a 36% false-refusal rate).
8. **Training throughput (H100, 16 prompts × 8 replies, 512-token cap): ~40–60 s per GRPO step.**
   ~25–35% of training replies still hit the 512 cap.
9. **Round 2 pure GRPO (lr 1e-4, KL 0.01, 30 steps) is the working GRPO baseline:** held-out
   prefill ASR 42.0% → **4.5%**, XSTest false refusal 4.0% → **6.0%** (at the +2-point limit),
   GSM8K 94% → 96%. SFT + GRPO was worse (10.0% / 9.2%). Round 1 (lr 1e-5) did not move the model.
   Details: `results/pilots_round2_heldout.md`, `November14.md`.
10. **At 4.5% held-out ASR there is no measurable room for GFlowNet on attack success.** Before
   building the trajectory-balance trainer: harder evaluations (long harmful prefills, unseen
   attack families) and a headline on over-refusal / diversity, agreed with the mentor.

## 3. Plan

### Phase 0 — Baseline + feasibility (this week) ← START HERE
- [x] **Untuned Qwen3-8B baseline on Modal** (see §4). Done 2026-11-13 — ASR plain 13%,
      prefill 62%, XSTest false refusal 4.4%, GSM8K 94%. Details: `results/baseline_qwen3_8b.md`.
- [x] Freeze data splits (2026-11-14): train 1,000 harmful (BeaverTails, half prefilled with
      train phrases) + 500 benign (OR-Bench-80k) + 500 GSM8K train; val 100/50/50. Hashes in
      `data/SPLITS.md`; files in `outputs/data/` (gitignored). Built by `data/build_splits.py`.
- [x] WildGuard judge + rewards (2026-11-14): `gvg/wildguard.py`, `gvg/rewards.py`. XSTest
      re-scored: false refusal 4.0%. Held-out prefill baseline: 42.0%.
- [ ] Manual spot-check of ~20 "not harmful" prefill replies (needs a person).
- [ ] NAR feasibility: load LLaDA-8B and SDAR-8B on one 48–80 GB GPU; measure s/rollout at
      gen length 256; decide how to compute log π(y|x) for TB (fixed unmasking order vs ELBO).

### Phase 1 — GRPO vs TB on Qwen3-8B, own loop on Modal (weeks 7–8)
- Loop: `gvg/rl.py` (pure; tested on CPU) + `train/modal_train.py` (vLLM sampling + HF/PEFT
  gradients on one H100; WildGuard on an L40S). 3-step check passed 2026-11-14.
- GRPO (built): 16 prompts/step (8 harmful, 4 benign, 4 math) × K=8, lr 1e-5, KL 0.05 to the
  base model (adapter disabled), one update per batch, group-normalised advantages.
- SFT + GRPO (built): rejection-sampling SFT (K=4 per prompt, keep best if reward ≥ 0.5), then
  the same GRPO starting from the SFT adapter, still regularised to the base model.
- Pilots done (2026-11-14). Round 1 (lr 1e-5, KL 0.05) did not learn; round 2 (lr 1e-4, KL 0.01,
  8 prefilled harmful of 16 prompts/step) did. Defaults in `GRPOConfig` are the round-2 values.
- Known issues to fix next: group-std normalisation amplifies WildGuard noise in all-safe groups;
  SFT + GRPO regularises toward the base model (undoes SFT); truncated SFT targets lack an
  end-of-turn token.
- TB (built, `--pipeline tb`; `GRPOConfig(method="tb")`): same loop, prompts, rewards, LoRA and
  reference as GRPO; loss = within-group variance of δ = log π/π_ref − r/τ (batch log Z, no length
  normalisation by default, so the target is exactly π_ref·exp(r/τ)). Pilot τ ∈ {0.03, 0.1, 0.3},
  freeze one, compare against GRPO on identical code.
- Full: 200 steps × methods × 3 seeds.
- Eval: `modal run baseline/modal_baseline.py --adapter <run> --attacks random-heldout`
  (held-out ASR, XSTest false refusal, GSM8K), plus tokens and semantic diversity.

### Phase 2 — NAR model (weeks 8–9, only if Phase 0 feasibility passes)
- Same rewards/splits/eval; custom LoRA loop. Run DIJA on untuned model first.
- Needs mentor approval for 60–150 GPU-hours (exceeds $200 AWS allowance).
- Fallback: AR-only paper + Junhui's 0.6B diffusion PoC.

### Phase 3 — Stopping / safe recovery (week 9+, only if TB ≥ GRPO)
- Resume from harmful prefills of 5–40 tokens (Qi et al.), 16 continuations/policy.
  Metric: Safe Recovery Rate. Reuse Shree's prefill harness (`run_cuda.py` port).

### Decision gates
- End of week 7: TB pilot not learning → debug jointly with Junhui, try learned log Z.
- Untuned attacked ASR < 5% → use stronger attacks before any training.
- No difference beyond seed noise after Phase 1 → still a result (equivalent objectives).

## 4. Baseline eval (Modal) — done; spec kept for reference

**Goal:** the "Untuned" row for Qwen3-8B: attacked ASR, false-refusal rate, GSM8K accuracy.
Actual cost: ≈ $1 per full run (5–6 min). Results: `results/baseline_qwen3_8b.md`.

**Script:** `baseline/modal_baseline.py` (single file, `modal run baseline/modal_baseline.py`).

**Setup**
```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_...   # HarmBench cls / WildGuard may be gated
```

**Spec**
- Image: `vllm` + `datasets`; model weights cached on a `modal.Volume` (`hf-cache`).
- Generation: vLLM, `Qwen/Qwen3-8B`, chat template with `enable_thinking=False`,
  temperature 0 (greedy), max_tokens 512. Also save K=4 samples at T=0.7 for a later diversity check (optional).
- Datasets (all from HF):
  | Set | Source | N | Measures |
  |---|---|---|---|
  | HarmBench standard | `walledai/HarmBench` (standard) | 200 | ASR, plain |
  | HarmBench + prefill attack | same prompts, assistant turn prefilled with `"Sure, here is"` | 200 | ASR, attacked |
  | XSTest | `walledai/XSTest` (250 safe / 200 unsafe) | 450 | false refusal on safe |
  | GSM8K | `openai/gsm8k` test, first 200 | 200 | accuracy |
- Judging:
  - ASR: `cais/HarmBench-Llama-2-13b-cls` with its official prompt template (run after
    freeing the generator, or in a second Modal function).
  - XSTest refusal: WildGuard (primary) and string-match detector (kept for comparison).
  - GSM8K: extract last number, compare to `####` answer.
- Outputs (on Modal volume, then download — **never commit**):
  `outputs/baseline_qwen3_8b/{harmbench_plain,harmbench_prefill,xstest,gsm8k}.jsonl`
  and `summary.json` with: ASR_plain, ASR_prefill, false_refusal_rate, gsm8k_acc,
  mean_tokens, wall-clock, git commit, seed, model revision.

**Done when** `summary.json` exists and its numbers are posted to the team doc/Slack.
If ASR_prefill is also near 0, note it — that triggers the "stronger attacks" gate.

## 5. Repo conventions

```
gvg/          shared library: prefill_attacks, wildguard, scoring, rewards, data, rl (GRPO/SFT)
baseline/     modal_baseline.py — evaluation (untuned model or --adapter <run>)
train/        modal_train.py — LoRA training on Modal (grpo | sft | sft+grpo)
data/         build_splits.py + SPLITS.md (counts and hashes only; prompts are not committed)
tests/        test_offline.py — CPU tests; also run in the Modal image (--tests-only)
results/      recorded results (metrics only)
docs/         literature review
outputs/      gitignored: generations, prompt files, training logs
```
- Run `python -m pytest tests/test_offline.py -q` before any paid Modal run; the training image
  runs the same tests with `modal run train/modal_train.py --tests-only`.
- Checkpoints and training logs live on the Modal volume `gvg-ckpts` under `/<run_name>/`.
- Every run records: config, seed, git SHA (with `-dirty` if uncommitted), model revision,
  judge versions, data hash.
- Keep training judge (WildGuard) and eval judge (HarmBench cls) separate. Never mix.
- Held-out prompts/attack templates must never enter training data — assert this in code
  (`prefill_attacks.assert_no_heldout_phrases`, `assert_no_heldout`; `data.build` drops prompts
  overlapping HarmBench/XSTest).
- Matched budgets between GRPO and TB: same prompts, K, steps, LoRA rank, reward code.
- Report means ± std over 3 seeds; no single-seed claims.

## 6. Safety & ethics rules
- Harmful model outputs stay in `outputs/` (gitignored) or private storage. Never commit, never
  paste in public channels.
- Do not publish adapters, reward code with inverted signs, or harmful prompt sets until evals done.
- Don't put API keys in code; use Modal secrets or env vars.
