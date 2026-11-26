# SubTB reference notes: Hu et al. 2024 (`GFNOrg/gfn-lm-tuning`) vs our setup

Read 2026-11-26 at commit `8c640356b5e7bf71f1333f9c384e6900857b1457`. Files: `next_sentence/utils.py`
(`score_fast`, `generate_and_return_termination_logprob`, `modified_subtb_loss`, `ReplayBuffer`),
`next_sentence/lightning_module.py` (`training_step`), `next_sentence/configs/task/openwebtext_gpt2.yaml`,
`infill_subj_arithmetic/main.py`, `infill_subj_arithmetic/configs/rationale_buffer.yaml`.

## What the reference does

**Loss: "modified SubTB"** (backward policy = 0, since AR states form a tree). The flow of a prefix
is defined by terminating there, so no flow network and no log Z are needed:

    log F(s_i) = log R(s_i ⊤) − log π_θ(⊤ | s_i)
    δ_i = log R(s_i⊤) + log π_θ(y_{i+1} | s_i) + log π_θ(⊤ | s_{i+1}) − log R(s_{i+1}⊤) − log π_θ(⊤ | s_i)
    loss = Σ_{i<j} λ^{j−i−1} (Σ_{k=i..j−1} δ_k)² / Σ_{i<j} λ^{j−i−1}

computed with a cumulative sum of δ over all sub-trajectory lengths; tokens after EOS are masked.
λ = 1.0 in every config (all sub-trajectory lengths weighted equally).

**Reward on every prefix, in log space.** `score_fast`: one forward pass of the frozen base model
(LoRA disabled) gives log R(s_i⊤) = log p_base(y_1..y_i, ⊤ | x) / T for every i. Prefixes shorter
than `min_len` get log R = −99; banned vocabulary gets a −50 logit penalty.

**Exploration (per training step):**
- with prob. 0.25: replay buffer (per prompt, top 50 by log R, near-duplicates dropped at edit
  distance < 0.25), re-scored under the current policy;
- else with prob. 0.666: sample at temperature U[0.5, 2.0];
- else: sample at temperature 1.

**Reward temperature annealing:** log R divided by T, T linear 1.0 → 0.8 over 750 steps
(next-sentence); 1.0 → 0.7 over 200 (arithmetic default); 1.1 → 0.5 over 150 (arithmetic varying).

**Scale:** GPT-2 XL / GPT-J 6B; replies 15–30 tokens; 16–20 samples per prompt; lr 1e-4 with a
20-step warm-up; LoRA r = 64, α = 16, dropout 0.1; gradient accumulation 25–32; 2,000 steps.

## Where our setup differs (each needs a decision in the pre-registration)

| # | Reference | Ours | Consequence / proposal |
|---|---|---|---|
| 1 | Reward on every prefix, nearly free (one LM pass) | WildGuard, an expensive judge; prefix check says first cut ≥ 32 tokens | States = chunk boundaries every 32–64 tokens, not every token. SubTB over ~5–10 chunk states per reply. |
| 2 | Flow = reward if terminated now, log π(⊤\|s_i) | Mid-reply, log π(EOS) is tiny and noisy (often < −20 nats) | **Open question for Mohammed.** Options: (a) use the reference definition at chunk ends (maybe end-of-sentence boundaries only); (b) learn a small flow head log F(s_i). |
| 3 | log R = log p_base / T | log R = log π_ref + r/τ | Same structure: the π_ref part plays their base-LM role, r/τ is added. The π_ref part telescopes across segments, so segment credit comes from r. |
| 4 | Replies 15–30 tokens | ~300 tokens (25–35% hit the 512 cap) | With chunk states, pair count is small; λ = 1 is fine at ~10 states. |
| 5 | Per-prompt replay buffer, prompts seen many times over 2,000 steps | 2,000 prompts, 30 steps × 16 prompts: each prompt seen ≈ 0.25 times | **Per-prompt buffer is useless as is.** Options: smaller prompt pool repeated, a global buffer of high-reward replies, or reuse the off-policy reference samples (our existing `--ref-samples`). |
| 6 | Temperature U[0.5, 2.0] on 15–30 tokens | 300 tokens on an 8B chat model | Temperature 2.0 produces gibberish over long replies; cap around 1.0–1.3. |
| 7 | 2,000 steps | 30 steps (matched to GRPO) | GFlowNets are slower to train. Proposal: both arms at a longer horizon (e.g. 100 steps) and report the GRPO 30-step adapter too. Budget-matched either way. |
| 8 | Reward temperature annealed | τ = 0.03 fixed | Keep τ fixed for comparability with the TB result, or pre-register a schedule (e.g. τ 0.1 → 0.03). |
| 9 | No log Z (modified SubTB) | Batch-mean log Z in our TB | Removes the log Z estimate entirely. |
| 10 | LoRA r = 64 | r = 32 (matched to GRPO) | Keep 32. |
| 11 | Reward r is a log-likelihood | r = 1 − P(harmful) (linear) | Pre-register: r/τ as now, or log(1 − P(harmful)). |

## Cost of the extra pieces
- log π_θ(⊤ | s_i) comes from the same forward pass that already gives token log-probs.
- WildGuard on 5–10 chunks per reply: 5–10× the judge calls of GRPO (WildGuard runs on a
  separate L40S; throughput impact to be measured in the smoke run).
