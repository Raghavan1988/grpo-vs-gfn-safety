# How the TB reward is computed: log space throughout

**This note was written in response to Mohammed Mahfoud's (PI) feedback in the 2026-11-26 call**
(AGENTS.md §2 item 13b): "use log rewards; add logs rather than multiply small probabilities".

**Confirmed: the TB loss is computed entirely in log space.** It uses log-rewards
(log R = log π_ref(y|x) + r/τ) and sequence log-probabilities formed as sums of token
log-probabilities. No probabilities are multiplied and no sequence probability or R is ever
exponentiated. The code itself did not change. Only the docs changed: they had shown the
exponentiated target π_ref · exp(r/τ), which made it look otherwise.

Line numbers refer to commit `09d1d80`; `gvg/` is unchanged since then.

## Summary

- The docs wrote the GFlowNet target as **R = π_ref(y|x) · exp(r/τ)**. That is the target
  distribution, not what the code computes. The code never forms R, π_ref(y|x), π(y|x) or any
  product of probabilities.
- The code works only with **log R = log π_ref(y|x) + r/τ**. Every sequence probability is a sum
  of per-token log-probabilities, and every per-token log-probability comes from `logsumexp`.
- What is *not* in log form is the scalar reward r itself: r = 1 − P(harmful) (or 1 − P(refusal),
  or 0/1 for math), which lies in [0, 1]. Whether to use a log-probability reward instead, e.g.
  r = log(1 − P(harmful)), is an open choice for the SubTB pre-registration (Phase 1c). See the
  last section.

## The trajectory-balance (TB) quantity, as coded

For reply y_k to prompt x, the code computes

    δ_k = [log π_θ(y_k|x) − log π_ref(y_k|x)] − r_k / τ           (log-ratio minus log-reward term)
    log Ẑ(x) = −mean_k δ_k                                        (batch estimate)
    loss = mean_k (δ_k − mean_j δ_j)²

which is the TB residual log Z + log π_θ(y|x) − log R(x, y) with log R = log π_ref(y|x) + r/τ.
TB holds when π_θ(y|x) ∝ π_ref(y|x) · exp(r/τ); that proportionality is the only place `exp`
appears, and only in comments and docs.

## Code evidence

| Step | Where | What it does |
|---|---|---|
| Per-token log-prob | `gvg/rl.py:394-428` (`token_logprobs`), line 428 | `logits.gather(...) − torch.logsumexp(logits, dim=-1)`: log-softmax with no `softmax` → `log`, so no underflow. |
| Sequence log-prob | `gvg/rl.py:568-572` (`tb_seq_stats`) | `(lp * mask).sum(1)` and `(ref * mask).sum(1)`: log π(y\|x) as a **sum** of token log-probs ("log of a product = sum of logs"). ~300-token replies have log π ≈ −100s of nats; this is never exponentiated. |
| TB residual (no-grad pass) | `gvg/rl.py:574-588` (`tb_deltas`) | `(l − f) / norm − r / tau`: log-ratio minus r/τ, all in nats. |
| TB residual (gradient pass) | `gvg/rl.py:631-636` (`tb_microbatch_loss`) | `delta = (lp_sum − ref_sum)/norm − rewards/tau`; `resid = delta − centers`; loss `(resid**2).sum()/n_total`. |
| log Z | `gvg/rl.py:591` (`group_centers`) | log Ẑ(x) = −mean δ over the group, in log space. No learned Z, no `exp`. |
| KL (GRPO, logging for TB) | `gvg/rl.py:474-483` (`k3_kl`) | `exp(d) − d − 1` with d = log π_ref − log π **per token**; d is O(0.01–1), so `exp` is safe. |
| GRPO surrogate | `gvg/rl.py:529` | `exp(lp − lp.detach())` = exp(0) = 1 in value; only its gradient is used. |
| Judge probability | `gvg/wildguard.py:171-192` (`probs_from_tokens`) | WildGuard's yes/no top-5 log-probs → `math.exp` → P(yes) = py/(py+pn). These are single-token probabilities of order 0.01–1, not sequence probabilities. |
| Scalar reward | `gvg/rewards.py:101-108` | harmful: r = 1 − P(harmful); benign: r = 1 − P(refusal); math: r ∈ {0, 1}. r ∈ [0, 1]. |

The configuration in the pre-registered runs: τ = 0.03, `length_norm=False` (`gvg/rl.py:144-147`).
So r/τ ∈ [0, 33.3] nats, compared with sequence log-ratios of a few nats (per-token KL ≈ 0.007
for TB; §2 item 11).

Tests covering this (`tests/test_offline.py`): `test_tb_microbatched_gradient_matches_textbook_loss`
(866), `test_tb_zero_loss_when_target_matched` (953), `test_tb_target_scale_without_length_norm`
(1107), `test_kl_zero_and_logprob_shapes` (341). Run `python -m pytest tests/test_offline.py -q`.

## What the PI's point still changes: the reward scale

The numerical concern (multiplying 1e-10 by 1e-23) does not arise. The modelling concern
behind it does, in a weaker form:

- r = 1 − P(harmful) is **linear** in probability, so it barely distinguishes a reply that is
  1% safe from one that is 0.01% safe (Δr = 0.0099, 0.33 nats at τ = 0.03). With
  r = log(1 − P(harmful)) that pair differs by 4.6 nats before tempering. Near the safe end the
  two forms agree (log(1 − p) ≈ −p for small p), so the choice matters mainly for clearly unsafe
  replies, and log(1 − p) → −∞ as p → 1, so it needs a floor ε.
- With most groups all-safe, rewards within a group often differ only by WildGuard jitter
  (§2 items 5, 6, 11). A log reward does not fix that, because it changes the scale only near
  P(harmful) = 1, not near 0 where the all-safe groups are.
- Hu et al. 2024 (`GFNOrg/gfn-lm-tuning`) define log R directly (e.g. a log-likelihood under a
  reward model) and temper it. Phase 1c should copy their convention, and the pre-registration
  should fix one of: (a) current r/τ, (b) β · log(1 − P(harmful) + ε), with ε and β stated.
