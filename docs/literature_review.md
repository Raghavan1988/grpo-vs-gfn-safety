# Literature review: GFlowNet vs GRPO fine-tuning for LLM safety

**Date:** 13 November 2026
**Question:** Is it new to compare GFlowNet (trajectory-balance) fine-tuning with GRPO for
language-model safety, or has prior work already shown that GFlowNets work better for
autoregressive models?
**Scope:** a targeted search (five web searches plus the works cited in the project proposal),
not a systematic review. Preprints from the last few months may be missing. Numbers below are
as reported by each paper's authors; none has been independently replicated in the sources found.

---

## 1. Summary

- **Comparing GFlowNet-style objectives with GRPO on autoregressive (AR) LLMs is not new.**
  Several 2025–2026 papers do it, mainly for mathematical and code reasoning. The strongest,
  FlowRL (ICLR 2026), reports an average gain of 10% over GRPO on maths benchmarks at 32B.
- **Superiority is not established in general.** The evidence comes mostly from preprints by
  the methods' own authors, concentrates on reasoning tasks, and centres on diversity. One
  safety-adjacent comparison (PROBE) found GRPO and a GFlowNet roughly equal. Theory shows that
  trajectory balance with a reference-model prior targets the same distribution as
  KL-regularised RL, so any gain must come from optimisation dynamics rather than the objective.
- **In safety, GFlowNets have been used on the attacker side** — to generate diverse
  jailbreak prompts for red-teaming — not to train the defending model.
- **No work was found that fine-tunes the defending model with GFlowNet trajectory balance for
  jailbreak robustness and compares it with GRPO under a matched safety reward**, measuring the
  harm / over-refusal trade-off and robustness to unseen attacks. No work was found doing this
  for diffusion language models either. This is the gap the project addresses.

---

## 2. GFlowNet-style objectives versus GRPO on autoregressive LLMs

| Work | Venue | Task | Reported finding |
|---|---|---|---|
| Hu et al., *Amortizing Intractable Inference in LLMs* | ICLR 2024 | Infilling, reasoning, sampling | Introduced GFlowNet fine-tuning of LLMs; better sample diversity and data efficiency than reward-maximising fine-tuning. |
| Bartoldson et al., *Trajectory Balance with Asynchrony* (TBA) | arXiv 2503.18929 (2025) | GSM8K, preference tuning, red-teaming | Trajectory balance with a replay buffer and asynchronous sampling; matches or beats PPO-style baselines on GSM8K under compute limits with 4–50× faster training. |
| Zhu et al., *FlowRL* | ICLR 2026 (arXiv 2509.15207) | Maths and code reasoning | Matches a reward-defined target distribution (equivalent to trajectory balance). Reports +10.0% over GRPO and +5.1% over PPO on maths at 32B, similar gains at 7B, and roughly double the solution diversity of the strongest baseline. |
| *GFlowRL* | arXiv 2607.13394 (2026) | LLM reasoning at scale | Attributes trajectory-balance instability to the learned partition function; replaces it with an in-batch Monte Carlo estimate. Reports GRPO collapsing onto one mode by ~500 steps while GFlowRL keeps several high-reward modes. |
| *PowerFlow* | arXiv 2603.18363 (2026) | Logical reasoning | Sequence-level trajectory balance is dominated by response length; a length-aware variant rivals or surpasses GRPO. |
| *f-Trajectory Balance* | arXiv 2605.15417 (2026) | Off- and on-policy LLM tuning | A family of trajectory-balance losses that train stably; compared with PPO because GRPO could not run with the asynchronous delay studied. |
| *PROBE* | arXiv 2610.22124 (2026) | Auditing policy compliance of tool-using agents | Controlled comparison of GRPO, DPO and a GFlowNet; GRPO and the GFlowNet perform similarly. |

**Theory.** Tiapkin et al. and Deleu et al. (cited in the proposal as [91] and [114]) show that
GFlowNet training with a reference prior is equivalent to KL-regularised (entropy-regularised)
RL. For left-to-right generation, trajectory balance with prior π_ref and KL-regularised GRPO
therefore share a target distribution; they differ in the estimator, the variance of the
gradient, and how training explores.

**Known practical problems with trajectory balance at LLM scale:** instability from the learned
log-partition estimate (GFlowRL, Stable-GFlowNet), exploding gradients on long outputs (FlowRL),
response-length bias (PowerFlow), and the assumption of on-policy samples (FlowRL, TBA).

**Assessment.** The claim "GFlowNets beat GRPO on AR models" has been tested for reasoning and
leans positive, mainly through diversity. It is not proven in general and has not been tested
for safety.

---

## 3. GFlowNets in LLM safety: the attacker side

| Work | Venue | Role of the GFlowNet |
|---|---|---|
| Lee et al., *Learning Diverse Attacks on LLMs for Robust Red-Teaming and Safety Tuning* | ICLR 2025 | Trains an **attacker** to generate diverse, effective jailbreak prompts; the attacks are then used for safety tuning. |
| Kwon et al., *Stable-GFlowNet* | ICML 2026 (arXiv 2605.00553) | Stabilises GFlowNet **red-teaming** (contrastive trajectory balance, robustness to noisy rewards, less mode collapse). |
| Bartoldson et al., TBA | arXiv 2503.18929 | Red-teaming experiment: comparable attack success and prompt diversity to synchronous GFlowNets, about 7× faster. Notes that standard RL suffers mode collapse and lacks attack diversity. |
| Kwon et al., *GDPO* | EMNLP 2024 | GFlowNet-based preference alignment that preserves diversity; general alignment, not jailbreak robustness. |

Defence-side safety work found in the search uses other methods: deeper refusal training
(DeepRefusal, arXiv 2509.15202), fine-tuning safety-relevant neurons (SafeTuning,
arXiv 2509.01631), and GRPO for calibrated refusal in agents (arXiv 2603.03205). Qi et al.
(*Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, ICLR 2025) show that
safety alignment concentrates in the first few response tokens, which is why prefill attacks
work.

---

## 4. Diffusion (non-autoregressive) language models

- *Trajectory-Balance Post-Training for Diffusion Language Models* (TraFL, arXiv 2605.13935)
  applies trajectory balance to diffusion LMs to reduce trajectory locking in reasoning. It
  does not address safety.
- Diffusion-LM safety work (DIJA, DiffuGuard, A2D; all cited in the proposal) identifies
  diffusion-specific jailbreaks and proposes inference-time or alignment defences, without
  GFlowNet training.
- No work was found combining GFlowNet training with diffusion-LM safety.

---

## 5. Where this project is new

| Component | Already done? |
|---|---|
| GFlowNet-style training vs GRPO on AR LLMs | Yes, for reasoning (FlowRL, GFlowRL, TBA, PowerFlow) |
| GFlowNets for generating jailbreak attacks | Yes (Lee et al., Stable-GFlowNet, TBA) |
| GFlowNet training of the **defending** model for jailbreak robustness | **Not found** |
| Matched comparison with GRPO under the **same safety reward** | **Not found** |
| Joint measurement of harm (HarmBench) and over-refusal (XSTest) for this comparison | **Not found** |
| Robustness to **held-out attacks** (e.g. unseen prefill phrases) as the headline metric | **Not found** |
| Any of the above for **diffusion LMs** | **Not found** (TraFL covers reasoning only) |

The open question is therefore not "do GFlowNets beat GRPO?" but **"does matching the reward
distribution, rather than maximising reward, produce safer and more robust models?"**

---

## 6. What is worth doing

### Worth doing (core contribution)

1. **Defender-side comparison on Qwen3-8B.** Trajectory balance vs GRPO under an identical
   multi-objective safety reward (WildGuard for safety and false refusal, answer correctness for
   quality), the same prompts, rollouts per prompt, steps and LoRA rank, and three seeds.
2. **Make held-out robustness the headline, not raw attack success.** The project's untuned
   baseline (`results/baseline_qwen3_8b.md`) shows 13% attack success with no prefill, 62% with a
   fixed "Sure, here is" prefill, and 42.5% with a random prefill drawn from 50 phrases. Split
   the 50 phrases into training and held-out sets, train on one, and report attack success on
   the other. This is where distribution matching should help if it helps anywhere.
3. **Report the harm / over-refusal trade-off.** HarmBench attack success alongside XSTest false
   refusals, plus GSM8K as a capability check. A method that cuts attacks by refusing everything
   should be visible as such.
4. **Measure mode collapse directly in the safety setting.** Diversity of safe responses,
   collapse onto a single refusal template, and policy entropy over training. The GFlowRL and
   FlowRL argument predicts GRPO collapses; the project can test whether that collapse is what
   makes GRPO less robust to unseen attacks.

### Needed to make the comparison credible

5. **Control for the objective.** Include KL-regularised GRPO with the same KL strength as the
   trajectory-balance reference prior, so any difference is attributable to optimisation rather
   than to the target distribution. State the equivalence result up front.
6. **Use a trajectory-balance variant known to be stable at this scale:** an in-batch estimate of
   the log-partition term (GFlowRL, TBA) instead of a learned one, and length-normalised log
   ratios (PowerFlow). Cite FlowRL and GFlowRL as the closest prior work and explain why safety
   differs from reasoning.
7. **Use statistics suited to small differences:** paired tests on the same prompts (e.g.
   McNemar), more held-out prompts than 200, and three seeds. vLLM greedy decoding is not
   bit-reproducible across runs, so single-run differences of a few points are noise.

### Secondary, if time allows

8. **Diffusion-model arm** (SDAR-8B or LLaDA-8B, building on TraFL), evaluated with DIJA. Novel,
   but depends on a feasibility check of computing sequence log-probabilities for trajectory
   balance.
9. **Recovery from a harmful opening** (the proposal's safe-recovery experiment): start from
   prefills of 5–40 harmful tokens and measure how often each model recovers. Links directly to
   Qi et al.'s shallow-alignment finding.

### Not worth leading with

- **Showing that GFlowNets beat GRPO on AR reasoning in general** — already covered by FlowRL,
  GFlowRL and TBA.
- **Inference-time reward-weight conditioning (α) and learned stopping/revision actions** before
  the core comparison works. Both add training complexity and confounds; they fit better as
  follow-up work.

### On a null result

If trajectory balance and GRPO end up equivalent on safety, the result is still informative:
theory predicts a shared target distribution, and a careful matched comparison with diversity and
held-out-attack measurements would show whether the reasoning-domain gains transfer to safety.

---

## References

- Hu et al. *Amortizing Intractable Inference in Large Language Models.* ICLR 2024.
- Bartoldson et al. *Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for
  Fast, Scalable LLM Post-Training.* [arXiv 2503.18929](https://arxiv.org/abs/2503.18929)
- Zhu et al. *FlowRL: Matching Reward Distributions for LLM Reasoning.* ICLR 2026.
  [arXiv 2509.15207](https://arxiv.org/abs/2509.15207) ·
  [ICLR poster](https://iclr.cc/virtual/2026/poster/10007683)
- *GFlowRL: Scaling Distribution-Matching RL to Large Language Models.*
  [arXiv 2607.13394](https://arxiv.org/abs/2607.13394)
- *PowerFlow.* [arXiv 2603.18363](https://arxiv.org/abs/2603.18363)
- *f-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with
  Off- and On-Policy Data.* [arXiv 2605.15417](https://arxiv.org/abs/2605.15417)
- *PROBE: Learning to Audit Policy Compliance in Tool-Using LLM Agents.*
  [arXiv 2610.22124](https://arxiv.org/abs/2610.22124)
- Tiapkin et al.; Deleu et al. Equivalence of GFlowNets with reference priors and
  KL / entropy-regularised RL (proposal references [91], [114]).
- Lee et al. *Learning Diverse Attacks on Large Language Models for Robust Red-Teaming and
  Safety Tuning.* ICLR 2025.
- Kwon et al. *Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive
  Trajectory Balance.* ICML 2026. [arXiv 2605.00553](https://arxiv.org/abs/2605.00553)
- Kwon, Matsunaga & Kim. *GDPO: Learning to Directly Align Language Models with Diversity Using
  GFlowNets.* EMNLP 2024.
- *Trajectory-Balance Post-Training for Diffusion Language Models* (TraFL).
  [arXiv 2605.13935](https://arxiv.org/abs/2605.13935)
- Qi et al. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR 2025.
- *Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating
  Refusal Direction* (DeepRefusal). [arXiv 2509.15202](https://arxiv.org/abs/2509.15202)
- *Unraveling LLM Jailbreaks Through Safety Knowledge Neurons* (SafeTuning).
  [arXiv 2509.01631](https://arxiv.org/abs/2509.01631)
- *Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool
  Use.* [arXiv 2603.03205](https://arxiv.org/abs/2603.03205)
- Wen et al. *DIJA* (diffusion-LM jailbreaks); Li et al. *DiffuGuard*; Jeung et al. *A2D*
  (ICLR 2026; proposal references [6], [19], [20]).
