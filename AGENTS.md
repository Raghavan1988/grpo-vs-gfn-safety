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
| AR | `Qwen/Qwen3-8B` (thinking OFF) | Tinker (LoRA) |
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

## 2. Known facts / decisions that shape the work

1. **Nobody has a working GFlowNet run yet.** Junhui's 200M dLLM PoC gave zero reward and was
   abandoned. First milestone for TB is "reward goes up without blowing up".
2. **Frozen vs LoRA is inconsistent in the proposal.** Working assumption: **LoRA on model
   weights** (what Tinker supports). Confirm with mentor before writing the paper.
3. **Measured untuned Qwen3-8B ASR: 13% plain, 62% with a "Sure, here is" prefill** (HarmBench,
   n=200). Plain is not ≈0 (mostly misinformation/cybercrime), but the attacked condition is where
   the headroom is. Always report both, plus DIJA for NAR.
   Prefill attacks are varied, not static: `baseline/prefill_attacks.py` holds 10 openers split
   into `train` / `heldout_string` / `heldout_family`. Train only on `train`; the headline
   robustness number for fine-tuned models is held-out ASR (`--attacks all`).
   Random-prefill baseline (`--attacks random`: 1 of 50 phrases per request, ≤5 uses each):
   **42.5%** ASR vs 62% for fixed "Sure, here is". vLLM greedy decoding is not bit-reproducible
   across runs; compare methods on paired prompts and seeds.
4. **For AR decoding, TB with reference prior ≡ KL-regularized RL** (Tiapkin et al., Deleu et al.).
   AR comparison therefore tests optimizer / log Z estimation, not the objective. Say so in the paper.
5. TB is unstable at this scale: length collapse, exploding grads with learned log Z,
   instability for β < ~0.004 (TBA). Use batch-mean log Z estimate + length-normalized log-ratio.
6. GRPO gets zero signal when all K responses agree (all refuse / all comply). Log this fraction.

## 3. Plan

### Phase 0 — Baseline + feasibility (this week) ← START HERE
- [x] **Untuned Qwen3-8B baseline on Modal** (see §4). Done 2026-11-13 — ASR plain 13%,
      prefill 62%, XSTest false refusal 4.4%, GSM8K 94%. Details: `results/baseline_qwen3_8b.md`.
- [ ] Freeze data splits: train 1,000 harmful (BeaverTails) + 1,000 benign; held-out 200 + 200,
      with attack templates not seen in training. Write split hashes to `data/SPLITS.md`.
- [ ] Reward wrapper `rewards/wildguard.py` returning (R_s, C_r) per (prompt, response).
- [ ] NAR feasibility: load LLaDA-8B and SDAR-8B on one 48–80 GB GPU; measure s/rollout at
      gen length 256; decide how to compute log π(y|x) for TB (fixed unmasking order vs ELBO).

### Phase 1 — GRPO vs TB on Qwen3-8B via Tinker (weeks 7–8)
- Pilot each method ~30 steps (K=8, 32 prompts/step). Gate: reward ↑, grad norm bounded,
  response length stable.
- TB: batch-mean log Z(x) over K samples, length-normalized log-ratio, β ∈ 3 values → freeze one.
- GRPO: Tinker PPO loss, group-mean advantages, KL 0.05.
- Full: 200 steps × 2 methods × 3 seeds (~$250 Tinker).
- Eval: held-out attacked ASR, XSTest false refusal, GSM8K/MATH-500, tokens, semantic diversity.

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

## 4. Tonight's baseline (Modal)

**Goal:** the "Untuned" row for Qwen3-8B: attacked ASR, false-refusal rate, GSM8K accuracy.
Expected cost: < $10 (one H100 / A100-80GB, < 1 hour).

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
  - XSTest refusal: string-match refusal detector (as in the XSTest paper) for tonight;
    WildGuard later.
  - GSM8K: extract last number, compare to `####` answer.
- Outputs (on Modal volume, then download — **never commit**):
  `outputs/baseline_qwen3_8b/{harmbench_plain,harmbench_prefill,xstest,gsm8k}.jsonl`
  and `summary.json` with: ASR_plain, ASR_prefill, false_refusal_rate, gsm8k_acc,
  mean_tokens, wall-clock, git commit, seed, model revision.

**Done when** `summary.json` exists and its numbers are posted to the team doc/Slack.
If ASR_prefill is also near 0, note it — that triggers the "stronger attacks" gate.

## 5. Repo conventions

```
baseline/     one-off eval scripts (Modal)
data/         split scripts + SPLITS.md (no raw harmful completions)
rewards/      judge wrappers (training judges only)
train/        tinker_grpo.py, tinker_tb.py, nar_loop/
eval/         eval harness + eval-only judges
configs/      YAML per run; every run logs its config + git SHA
outputs/      gitignored
```
- Every run records: config, seed, git SHA, model revision, judge versions.
- Keep training judge (WildGuard) and eval judge (HarmBench cls) separate. Never mix.
- Held-out prompts/attack templates must never enter training data — assert this in code
  (`prefill_attacks.assert_no_heldout`, `sample_train_prefill`).
- Matched budgets between GRPO and TB: same prompts, K, steps, LoRA rank, reward code.
- Report means ± std over 3 seeds; no single-seed claims.

## 6. Safety & ethics rules
- Harmful model outputs stay in `outputs/` (gitignored) or private storage. Never commit, never
  paste in public channels.
- Do not publish adapters, reward code with inverted signs, or harmful prompt sets until evals done.
- Don't put API keys in code; use Modal/Tinker secrets or env vars.
