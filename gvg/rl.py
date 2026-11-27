"""GRPO, trajectory balance and rejection-sampling SFT for LoRA fine-tuning, independent of where
they run.

What this file is for (plain-English overview)
----------------------------------------------
We fine-tune the Qwen3-8B language model so that it refuses harmful requests, still helps with
harmless ones, and keeps its maths skill. We do not change the 8 billion original weights; we train
a small add-on called a LoRA adapter (a few million extra numbers that nudge the model's output).
Three training methods live here:

* GRPO (Group Relative Policy Optimization), a reinforcement-learning (RL) method: let the model
  answer each prompt 8 times, score each answer, and push up the answers that scored better than
  their siblings.
* TB (trajectory balance), the GFlowNet objective: same data, different loss. Instead of "maximise
  reward" it asks the model to sample replies in proportion to pi_ref * exp(reward / tau).
* Rejection-sampling SFT (supervised fine-tuning): sample a few answers, keep the best one if it is
  good enough, and train the model to copy it (ordinary next-token prediction).

Generation and judging are passed in as functions, so the same loop runs on Modal (vLLM +
WildGuard) and in CPU tests (tiny model + fake judge).

    generate_fn(prompt_ids: list[list[int]], n: int, step: int, temperature: float,
                max_new_tokens: int) -> list[list[list[int]]]     # [prompt][sample] -> completion ids
        (called with the extra keyword reference=True to sample from the reference policy
        instead; only when cfg.ref_samples > 0, so on-policy callers need not accept it)
    judge_fn(pairs: list[tuple[str, str]]) -> list[dict]          # gvg.wildguard.score format

GRPO objective (one policy update per batch, so the PPO ratio is 1 and clipping is inactive):
    loss = mean_seq[ mean_tok( -A * exp(logp - logp.detach()) + beta * KL_k3(pi || pi_ref) ) ]
A = group-normalised reward (r - mean_group) / (std_group + eps); groups whose reward spread is
below adv_min_std get A = 0 (otherwise judge jitter in an all-safe group, e.g. P(harmful) 0.001 vs
0.004, would be scaled up to the same strength as a real safe-vs-harmful split).
With one update per batch the importance ratio is 1, so this is group-normalised REINFORCE with a
KL penalty (the GRPO estimator without its clipping, which only matters for multiple epochs).

Reference policy (cfg.ref): "base" = LoRA disabled (the untuned model); "init" = the adapter the
run started from (e.g. the SFT adapter), so GRPO is measured against SFT instead of pulling the
model back toward the untuned one. Pure GRPO has no init adapter, so the two coincide.

Trajectory balance (cfg.method = "tb", the GFlowNet objective). Same loop, prompts, rewards,
sampling, LoRA and reference policy as GRPO; only the loss differs. Target distribution
    R(x, y) = pi_ref(y|x) * exp(r(x, y) / tau)
which is also the optimum of KL-regularised RL with beta = tau (Tiapkin et al.; Deleu et al.).
For an autoregressive model each reply has a single generation path, so the backward policy is
trivial and the TB residual of sample k in a prompt's group is
    delta_k = [log pi(y_k|x) - log pi_ref(y_k|x)] / norm_k - r_k / tau
with norm_k = 1 by default. cfg.length_norm = True divides by the reply length instead, but that
changes the target to pi_ref * exp(len(y) * r / tau): with ~300-token replies tau would mean a
~300x sharper preference, and WildGuard jitter between equally safe replies (~0.005) would become
a 5-50 nat preference. So it is off by default; tau then has its KL-RL meaning (beta = tau).
log Z(x) is not learned: it is estimated
from the group as log Z_hat = -mean_k delta_k (batch / "VarGrad" estimate, as in TBA and GFlowRL),
and the loss is the within-group variance of the residual
    loss = mean_k (delta_k - mean_j delta_j)^2.
Its gradient is 2 (delta_k - mean delta) * grad(log pi(y_k|x)) / norm_k: replies whose reward is
higher than the group's, relative to how much more likely the policy already makes them than the
reference, are pushed up. Two passes per step: a no-grad pass computes every delta (and the group
means); a gradient pass then uses the detached group mean, so micro-batches may split groups.

Off-policy TB (cfg.ref_samples = m > 0). The TB condition "delta_k is the same for every reply to
a prompt" characterises the target distribution whichever replies it is checked on, so the replies
need not come from the current policy (no importance weights are needed, unlike a policy
gradient). With ref_samples = m, each group holds group_size - m replies sampled from the policy
followed by m sampled from the reference policy (generate_fn(..., reference=True)). Why: on-policy,
once the policy rarely writes a harmful reply, most groups have no reward spread and so no
gradient, although the policy is still far from the target on the harmful replies it no longer
samples. The untuned reference keeps producing those, so the contrast stays in every step.
Budget is unchanged: the same number of replies, judge calls and gradient passes per step.
Logged reward / harmful-rate / KL / length metrics use the policy's own replies only (the k3 KL
estimate is only unbiased on policy samples); *_ref metrics describe the reference replies.

Notation used in comments below: tensor shapes are written like (B, L), meaning B rows (one per
sequence in the micro-batch) by L columns (one per token position). "logp" / "lp" means a
log-probability (a natural log of a probability, so always <= 0).
"""

# Makes type hints such as ``list[int]`` be stored as plain strings, so they work on older Python
# versions and cost nothing at run time.
from __future__ import annotations

# contextlib provides helpers for writing "with ...:" blocks (context managers); see
# adapter_ref_ctx below.
import contextlib

import json      # turns Python dicts into JSON text for the training logs
import math      # math.sqrt for the standard deviation in group_advantages
import random    # seeded random number generator for shuffling prompts
import time      # wall-clock timing of each training step
# dataclass: a decorator that writes the __init__ method for a class of named settings for us.
# asdict: converts such an object into a plain dict. field: customises one setting (see below).
from dataclasses import asdict, dataclass, field

# compute_rewards turns (prompt item, reply text) pairs into reward dicts using the judge.
from gvg.rewards import compute_rewards


# Sampling pools a training mix may name (see pool_key): the three prompt kinds, plus
# "harmful_prefill" for harmful prompts that carry a prefill attack.
POOLS = ("harmful_prefill", "harmful", "benign", "math")


# ----------------------------------------------------------------------------------- config
# @dataclass turns the class below into a simple "bag of settings". Each line `name: type = value`
# declares one setting with its default; Python generates GRPOConfig(steps=..., lr=...) for us, so
# you can write GRPOConfig(steps=100) and every other setting keeps its default.
@dataclass
class GRPOConfig:
    """Settings for one GRPO (or trajectory-balance) experiment.

    ``steps`` counts optimizer updates, ``group_size`` counts responses per
    prompt, and ``mix`` describes how many prompts of each kind are sampled.

    One step = draw sum(mix.values()) prompts (16 by default), generate ``group_size`` replies to
    each (16 x 8 = 128 replies), score them with the judge, then make ONE update to the LoRA
    weights. ``method`` picks the loss: "grpo" or "tb".
    """
    steps: int = 30
    group_size: int = 8                      # K rollouts per prompt
    # prompts per step by pool; "harmful_prefill" = harmful prompts with a prefill, and when it is
    # present "harmful" means harmful prompts without one. Round 2 puts most of the batch on
    # prefilled prompts, where replies differ and GRPO gets a signal.
    # Why field(default_factory=...)? A dict is mutable, so dataclasses forbid a plain dict default
    # (every config would share, and could accidentally modify, the SAME dict). default_factory is
    # a function that is called to build a fresh dict for each new config. `lambda: {...}` is a
    # tiny unnamed function with no arguments that just returns the dict.
    mix: dict = field(default_factory=lambda: {"harmful_prefill": 8, "harmful": 2, "benign": 3, "math": 3})
    max_new_tokens: int = 512                # replies are cut off after this many tokens
    temperature: float = 1.0                 # sampling randomness; 1.0 = sample the model as-is
    lr: float = 1e-4                         # round 1 used 1e-5: KL stayed ~0.001, no learning
    beta: float = 0.01                       # KL to the reference policy (round 1: 0.05)
                                             # bigger beta = stronger pull back toward pi_ref
    ref: str = "base"                        # "base" | "init" (see module docstring)
    adv_eps: float = 1e-4                    # added to the std so we never divide by zero
    adv_min_std: float = 0.05                # groups with reward std below this carry no signal
    micro_batch: int = 4                     # sequences per forward/backward pass (GPU memory)
    sort_by_length: bool = True              # micro-batch sequences of similar length together:
                                             # same gradient, fewer padded positions (see
                                             # microbatch_indices)
    grad_clip: float = 1.0                   # max total gradient norm (see clip_grad_norm_)
    seed: int = 0                            # makes prompt order and init reproducible
    val_every: int = 10                      # greedy validation every N steps (0 = off)
    save_every: int = 10                     # save the adapter every N steps (0 = final step only)
    method: str = "grpo"                     # "grpo" | "tb" (trajectory balance / GFlowNet)
                                             # | "subtb" (sub-trajectory balance, see below)
    tau: float = 0.1                         # TB reward temperature: target pi_ref * exp(r / tau)
                                             # small tau = sharp preference for high reward;
                                             # plays the same role as beta in GRPO's KL penalty
    length_norm: bool = False                # TB: divide the log-ratio by reply length (rescales tau
                                             # by ~reply length; see module docstring)
    ref_samples: int = 0                     # TB only (off-policy): this many of each group's
                                             # group_size replies are sampled from the REFERENCE
                                             # policy instead of the current one (see docstring)
    chunk: int = 64                          # SubTB only: intermediate states every `chunk`
                                             # generated tokens (after any prefill), on harmful
                                             # and benign prompts; maths replies keep TB's single
                                             # whole-reply term. See accumulate_subtb_grads.
    # Tempered sampling (TB / SubTB only; Hu et al. 2024): with probability temp_prob a step's
    # POLICY replies are sampled at a temperature drawn uniformly from [temp_low, temp_high]
    # instead of `temperature`. Valid for GFlowNet losses (any sampler), not for GRPO.
    temp_prob: float = 0.0                   # 0 = off (the default; the paper uses 2/3)
    temp_low: float = 0.7
    temp_high: float = 1.3
    # Log reward (TB / SubTB only): use log_beta * log(max(r, log_floor)) in place of r / tau,
    # i.e. target pi_ref * max(r, floor)^log_beta (beta = 1: pi_ref conditioned on the judge).
    log_reward: bool = False
    log_floor: float = 0.01
    log_beta: float = 7.0                    # 7 * log(0.01) = -32 nats, ~ the 1/tau = 33 range

    def __post_init__(self):
        """Validate the settings. Dataclasses call this automatically right after __init__.

        Raises ValueError early (before any expensive GPU work) if a setting is misspelt or
        nonsensical, e.g. GRPOConfig(method="ppo") or tau=0 (we divide by tau).
        """
        if self.method not in ("grpo", "tb", "subtb"):
            # f"...{x!r}" inserts repr(x), i.e. the value with quotes, e.g. 'ppo'.
            raise ValueError(f"method must be 'grpo', 'tb' or 'subtb', not {self.method!r}")
        if self.method == "subtb" and self.length_norm:
            raise ValueError("length_norm is not defined for subtb")
        if self.chunk < 1:
            raise ValueError("chunk must be at least 1")
        if (self.temp_prob or self.log_reward) and self.method == "grpo":
            raise ValueError("temp_prob and log_reward need method='tb' or 'subtb' (GRPO's gradient "
                             "is only valid on its own samples at its own temperature)")
        if not 0 <= self.temp_prob <= 1 or not 0 < self.temp_low <= self.temp_high:
            raise ValueError("need 0 <= temp_prob <= 1 and 0 < temp_low <= temp_high")
        if not 0 < self.log_floor < 1 or self.log_beta <= 0:
            raise ValueError("need 0 < log_floor < 1 and log_beta > 0")
        if self.ref not in ("base", "init"):
            raise ValueError(f"ref must be 'base' or 'init', not {self.ref!r}")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if not 0 <= self.ref_samples < self.group_size:
            # At least one reply per group must come from the policy itself, so that the logged
            # training metrics (reward, harmful rate, KL) still describe the policy.
            raise ValueError("ref_samples must be at least 0 and smaller than group_size")
        if self.ref_samples and self.method == "grpo":
            # The TB loss holds for replies from any distribution. GRPO's policy gradient is an
            # expectation over the policy's OWN samples; feeding it reference samples without
            # importance weights would be a biased, different algorithm.
            raise ValueError("ref_samples needs method='tb' or 'subtb': GRPO's policy gradient is only valid "
                             "on samples from the current policy")
        # The checks below catch settings that would otherwise fail (or silently do nothing)
        # only after a paid sampling + judging round.
        # A run with no optimizer updates would train nothing.
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.group_size < 2:
            # With one reply per prompt there is nothing to compare within a group: every GRPO
            # advantage is 0 and every TB residual equals its own group mean, so nothing trains.
            raise ValueError("group_size must be at least 2 (both losses compare replies within a group)")
        # The mix must be a non-empty dict of whole, non-negative prompt counts with at least one
        # prompt in total. `not self.mix` is True for an empty dict. isinstance(n, int) rejects
        # 1.5; the extra bool check is needed because Python counts True / False as integers.
        if not self.mix or any(not isinstance(n, int) or isinstance(n, bool) or n < 0
                               for n in self.mix.values()) or sum(self.mix.values()) == 0:
            raise ValueError(f"mix must map pools to non-negative prompt counts with at least one "
                             f"prompt per step, not {self.mix!r}")
        # Set difference: pool names in the mix that are not among the allowed ones, e.g. a typo
        # such as "maths". Without this check the typo would only fail when the sampler is built.
        unknown = set(self.mix) - set(POOLS)
        if unknown:
            raise ValueError(f"unknown pools in mix: {sorted(unknown)} (allowed: {list(POOLS)})")
        if self.temperature <= 0:
            # Temperature 0 is greedy decoding: all replies in a group would be identical.
            raise ValueError("temperature must be positive (training samples must differ within a group)")
        # A zero or negative learning rate cannot train (AdamW would not move, or move away).
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        # A negative KL weight would REWARD drifting from the reference; 0 (no penalty) is allowed.
        if self.beta < 0:
            raise ValueError("beta must not be negative")
        # adv_eps is a divisor guard, so it must be strictly positive; the std threshold may be 0
        # (off).
        if self.adv_eps <= 0 or self.adv_min_std < 0:
            raise ValueError("adv_eps must be positive and adv_min_std must not be negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        # getattr(self, name) reads the setting whose name is in the string, so one loop checks
        # several settings that share the same rule.
        for name in ("max_new_tokens", "micro_batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("val_every", "save_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative (0 switches it off)")


@dataclass
class SFTConfig:
    """Settings for rejection-sampling SFT (build_sft_set + train_sft).

    First the current model writes ``samples_per_prompt`` candidate replies per prompt; the best
    one is kept if its reward is at least ``keep_threshold``. Then the model is trained for
    ``epochs`` passes to reproduce the kept replies, ``batch_size`` examples per optimizer step.
    """
    samples_per_prompt: int = 4              # rejection sampling: K candidates per prompt
    temperature: float = 0.7                 # a bit less random than 1.0, so candidates are tidy
    max_new_tokens: int = 1024               # longer than GRPO's 512 so few targets are cut off
    keep_threshold: float = 0.5              # keep best candidate if its reward >= this
    keep_truncated_benign: bool = True       # a benign reply cut off at the cap is not a refusal
    epochs: int = 1                          # passes over the kept examples
    lr: float = 1e-4                         # AdamW learning rate
    batch_size: int = 16                     # examples per optimizer step
    micro_batch: int = 4                     # examples per forward/backward pass (GPU memory)
    grad_clip: float = 1.0                   # max total gradient norm
    seed: int = 0                            # controls the shuffle order

    def __post_init__(self):
        """Validate the settings before any sampling is paid for (see GRPOConfig.__post_init__)."""
        # Rewards lie in [0, 1], so a threshold outside that range would keep everything or
        # nothing.
        # Python allows the chained comparison a <= x <= b.
        if not 0.0 <= self.keep_threshold <= 1.0:
            raise ValueError("keep_threshold must lie in [0, 1], the range of the rewards")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        # Greedy decoding always gives the same reply, so paying for several candidates is
        # pointless.
        if self.temperature == 0 and self.samples_per_prompt > 1:
            raise ValueError("temperature 0 is greedy: all candidates would be identical; "
                             "use samples_per_prompt=1 or a positive temperature")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        for name in ("samples_per_prompt", "max_new_tokens", "epochs", "batch_size", "micro_batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")


# ----------------------------------------------------------------------------------- helpers
def prompt_text(tokenizer, item: dict) -> str:
    """Chat template (Qwen3, thinking off) + optional prefill forced onto the assistant turn.

    Args:
        tokenizer: the Hugging Face tokenizer of the model; it knows the chat format.
        item: one dataset row, a dict with at least "prompt" (the user's message) and optionally
            "prefill" (attacker-chosen opening words of the assistant reply, e.g. "Sure, here is").

    Returns:
        The full text the model sees before it starts writing, e.g. roughly
        "<|im_start|>user\\nHow do I ...<|im_end|>\\n<|im_start|>assistant\\nSure, here is".
        Because the prefill is part of the PROMPT, the model continues from it and must recover
        (e.g. "... actually, I can't help with that").
    """
    # Extra keyword arguments for the template: switch off Qwen3's "thinking" mode.
    kw = {"enable_thinking": False}

    def template(**extra):
        """Wrap the message in the model's special role markers.

        add_generation_prompt=True appends the "assistant turn starts here" marker; **extra
        unpacks a dict into keyword arguments, i.e. enable_thinking=False.
        """
        return tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}],
                                             tokenize=False, add_generation_prompt=True, **extra)
    try:
        p = template(**kw)
    except TypeError:  # tokenizers without the enable_thinking switch (tests)
        p = template()
    # item.get("prefill", "") returns "" when the item has no prefill, so plain prompts are
    # unchanged.
    return p + item.get("prefill", "")


def encode_prompt(tokenizer, item: dict) -> list[int]:
    """Return the prompt (including any prefill) as a list of token IDs.

    Token IDs are integers, one per word piece, which is what the model actually reads.
    add_special_tokens=False because the chat template already inserted all special markers.
    """
    return tokenizer(prompt_text(tokenizer, item), add_special_tokens=False)["input_ids"]


def group_advantages(rewards: list[float], group_size: int, eps: float = 1e-4,
                     min_std: float = 0.0) -> tuple[list[float], float]:
    """Normalize rewards within each prompt's response group.

    If every response receives the same reward, the group contains no preference
    signal.  The returned fraction records how often that happens.

    Args:
        rewards: flat list of rewards, ordered prompt by prompt: the first ``group_size`` values
            are the replies to prompt 0, the next ``group_size`` to prompt 1, and so on.
        group_size: replies per prompt (K = 8 in our runs).
        eps: small number added to the std so we never divide by (almost) zero.
        min_std: groups whose reward std is below this get advantage 0 (cfg.adv_min_std).

    Returns:
        (advantages, zero_fraction): one advantage per reply (same order as ``rewards``), and the
        fraction of groups that carried no signal.

    Example: rewards [1, 1, 0, 0, 1, 1, 0, 0] have mean 0.5 and std 0.5, so the advantages are
    about [+1, +1, -1, -1, +1, +1, -1, -1]: safe replies get pushed up, harmful ones down.
    If the 8 rewards are [0.999, 0.996, 0.998, ...] (all safe, only judge noise) the std is tiny,
    below min_std = 0.05, so every advantage is 0: without this rule, dividing by the tiny std
    would blow the noise up into a full-strength +1/-1 signal and the model would learn nonsense.
    """
    # Safety check: the list must split evenly into groups.
    assert len(rewards) % group_size == 0
    adv, zero = [], 0
    # range(start, stop, step): g = 0, 8, 16, ... = the first index of each group.
    for g in range(0, len(rewards), group_size):
        # Slicing [g:g + group_size] takes the group's rewards (end index not included).
        r = rewards[g:g + group_size]
        m = sum(r) / len(r)                                     # group mean
        # Population standard deviation: sqrt of the mean squared distance from the mean.
        # `(... for x in r)` is a generator expression: computes one value per x, fed to sum().
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / len(r))
        if sd < max(min_std, 1e-8):
            # All replies scored (almost) the same: nothing to prefer, so no push either way.
            zero += 1
            adv += [0.0] * len(r)                               # [0.0] * 8 = eight zeros
        else:
            # Standardise: above-average replies get positive advantage, below-average negative.
            # The list comprehension [expr for x in r] builds one advantage per reply.
            adv += [(x - m) / (sd + eps) for x in r]
    # Number of groups = len(rewards) // group_size (// is whole-number division).
    return adv, zero / (len(rewards) // group_size)


def _pad_batch(seqs: list[tuple[list[int], list[int]]], pad_id: int):
    """Convert variable-length examples into rectangular PyTorch tensors.

    The attention mask hides padding; the completion mask makes the loss ignore
    prompt tokens and score only tokens generated by the model.

    Args:
        seqs: list of (prompt_ids, completion_ids) pairs. The prompt includes any prefill, so
            prefill tokens are NOT counted as the model's own reply.
        pad_id: token ID used to fill unused positions.

    Returns:
        ids  (B, L) long: prompt + completion tokens, padded on the right with pad_id.
        att  (B, L) long: 1 for real tokens, 0 for padding (tells the model to ignore padding).
        cm   (B, L) float: 1.0 on completion (reply) tokens, 0.0 on prompt and padding.

    Example with prompt [5, 6], completion [7, 8, 9] and L = 6:
        ids = [5, 6, 7, 8, 9, pad], att = [1, 1, 1, 1, 1, 0], cm = [0, 0, 1, 1, 1, 0].
    """
    # Imported inside the function so the module can be imported without PyTorch installed.
    import torch
    # L = length of the longest prompt+completion; every row is padded up to it.
    L = max(len(p) + len(c) for p, c in seqs)
    # torch.full makes a (B, L) tensor where every entry is pad_id. A tensor is PyTorch's
    # multi-dimensional array; dtype=torch.long means 64-bit integers.
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    cm = torch.zeros((len(seqs), L), dtype=torch.float)
    # enumerate gives (index, item); the item is itself a (prompt, completion) pair, unpacked
    # into p and c.
    for i, (p, c) in enumerate(seqs):
        n = len(p) + len(c)
        ids[i, :n] = torch.tensor(p + c)    # row i, first n columns = the real tokens
        att[i, :n] = 1                      # mark them as real (not padding)
        cm[i, len(p):n] = 1.0               # only the completion part counts for the loss
    return ids, att, cm


def token_logprobs(model, ids, att):
    """Return the log probability assigned to each next token.

    Causal language models predict token ``t`` from tokens before ``t``, hence
    the one-position shift between logits and target token IDs.

    Args:
        model: the language model (with or without the LoRA adapter active).
        ids: (B, L) token IDs; att: (B, L) attention mask (see _pad_batch).

    Returns:
        (B, L-1) tensor; entry [b, t] = log P(ids[b, t+1] | ids[b, :t+1]), i.e. how likely the
        model thought the token that actually came next was. Summing these over a reply gives
        log pi(reply | prompt). Callers multiply by cm[:, 1:] to keep only reply tokens.

    Shift example: tokens [A, B, C]. The output at position 0 (after reading A) is a guess for
    B; at position 1 (after A, B) a guess for C; the output at the last position (after C)
    guesses a token we do not have, so it is dropped. Hence logits[:, :-1] pairs with ids[:, 1:].
    """
    import torch
    # The model returns logits of shape (B, L, V): for every position, one raw score per entry of
    # the vocabulary (V ~ 150k tokens for Qwen3). [:, :-1] = all rows, every position except the
    # last -> (B, L-1, V). .float() converts to 32-bit floats for accurate maths.
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    # Targets: the token that actually came next, i.e. ids without the first column -> (B, L-1).
    tgt = ids[:, 1:]
    # log softmax picked at the target token:
    #   log P(token) = logit[token] - log(sum over all v of exp(logit[v]))
    # because softmax(z)_i = exp(z_i) / sum_v exp(z_v); taking logs gives z_i - logsumexp(z).
    # torch.logsumexp computes log(sum(exp(.))) without overflowing -> (B, L-1).
    # .gather(-1, idx) picks, along the last (vocabulary) dimension, the logit at index idx.
    # tgt.unsqueeze(-1) reshapes (B, L-1) -> (B, L-1, 1) as gather requires, and .squeeze(-1)
    # removes that size-1 dimension again -> (B, L-1).
    # This avoids building the full (B, L-1, V) log-softmax tensor, saving memory.
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    return lp


def pad_token_id(tokenizer) -> int:
    """Token ID used to pad batches.

    Some tokenizers have no padding token; fall back to end-of-sequence. Padding is masked out
    of every loss, so which ID is used does not matter.
    """
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def _batch_tensors(model, seqs, pad_id):
    """Pad a micro-batch and move it to the model's device.

    Returns:
        ids (B, L), att (B, L): as in _pad_batch.
        mask (B, L-1): the completion mask shifted like the targets, so mask[:, t] says whether
            token t+1 is a reply token. It lines up with token_logprobs' (B, L-1) output; prompt,
            prefill and padding positions are 0.
        dev: the device (CPU or which GPU) the model lives on; all tensors must share it.
    """
    # next(iter) takes the first parameter just to read its device.
    dev = next(model.parameters()).device
    # A generator expression unpacked into three names.
    ids, att, cm = (t.to(dev) for t in _pad_batch(seqs, pad_id))
    return ids, att, cm[:, 1:], dev


def reference_logprobs(model, ids, att, ref_ctx=None):
    """Per-token log pi_ref (B, L-1), computed without gradients.

    ref_ctx() returns a context manager under which the model IS the reference policy (see
    adapter_ref_ctx). Default: model.disable_adapter(), a PEFT context manager that temporarily
    switches LoRA off, giving the original untuned model.
    """
    import torch
    ctx = ref_ctx() if ref_ctx is not None else model.disable_adapter()
    # torch.no_grad(): do not record operations for backpropagation (the reference is a fixed
    # target, we never train it), which also saves a lot of memory. Both context managers are
    # entered together, and both are undone when the block ends, even on error.
    with torch.no_grad(), ctx:
        return token_logprobs(model, ids, att)


def k3_kl(ref, lp):
    """Per-token KL(pi || pi_ref) estimate from log pi_ref and log pi of the sampled tokens.

    The "k3" estimator: with d = log pi_ref - log pi, k3 = exp(d) - d - 1. It is always >= 0
    (since e^d >= 1 + d), equals 0 when pi = pi_ref, and its average over tokens sampled from
    pi is an unbiased estimate of the KL divergence.
    """
    import torch
    d = ref - lp
    return torch.exp(d) - d - 1.0


def full_reply(tokenizer, item: dict, completion_ids: list[int]) -> str:
    """The reply as the user (and the judge) would see it: prefill + generated continuation.

    The prefill is part of the PROMPT ids, so it is not among the generated tokens and has to be
    put back in front of the decoded text.
    """
    return item.get("prefill", "") + _decode(tokenizer, completion_ids)


def grpo_microbatch_loss(model, seqs, adv, beta, pad_id, n_total, ref_ctx=None):
    """Loss for one micro-batch, already scaled by len(seqs)/n_total (for accumulation).

    ref_ctx() returns a context manager under which the model computes reference logprobs
    (default: adapter disabled = base model). The reference pass runs BEFORE the policy pass so
    that switching adapters never touches tensors in the policy's autograd graph.

    Args:
        model: the LoRA-wrapped model being trained (the "policy" pi).
        seqs: a few (prompt_ids, completion_ids) pairs (a micro-batch, e.g. 4 of the 128).
        adv: the advantage of each of those sequences (from group_advantages).
        beta: strength of the KL penalty that keeps pi close to the reference pi_ref.
        pad_id: padding token ID.
        n_total: number of sequences in the WHOLE step (e.g. 128). Dividing by it means the
            gradients of all micro-batches add up to the gradient of the average over all 128.
        ref_ctx: optional factory for the reference context (see adapter_ref_ctx).

    Returns:
        (loss, kl_mean): a scalar tensor to call .backward() on, and the average per-token KL of
        this micro-batch as a plain float (for logging only).
    """
    import torch
    ids, att, mask, dev = _batch_tensors(model, seqs, pad_id)
    ref = reference_logprobs(model, ids, att, ref_ctx)      # (B, L-1) log pi_ref per token
    # Policy pass, WITH gradient tracking (LoRA is active again here).
    lp = token_logprobs(model, ids, att)                    # (B, L-1) log pi per token
    # Advantages as a (B, 1) column so it broadcasts over every token of its sequence.
    # unsqueeze(1) turns shape (B,) into (B, 1).
    a = torch.tensor(adv, dtype=torch.float, device=dev).unsqueeze(1)
    # The policy-gradient trick. .detach() returns the same numbers but cut off from the gradient
    # graph (treated as a constant). So lp - lp.detach() is exactly 0 and exp(0) = 1: the VALUE
    # of pg is just -A. But its GRADIENT is -A * grad(lp) (the derivative of exp(x - c) at x = c
    # is 1). Minimising this raises log p of replies with A > 0 and lowers it when A < 0: the
    # REINFORCE update. (It is the PPO ratio pi/pi_old, which equals 1 with one update per batch.)
    pg = -a * torch.exp(lp - lp.detach())
    # KL penalty per token (see k3_kl), >= 0.
    kl = k3_kl(ref, lp)
    # Per-token loss: reward-driven term plus beta times the "don't drift too far" term.
    tok = pg + beta * kl
    # Number of reply tokens per sequence, (B,). clamp(min=1) avoids dividing by 0 for an
    # empty reply.
    ntok = mask.sum(1).clamp(min=1.0)
    # Average over the reply tokens of each sequence (prompt tokens are multiplied by 0) -> (B,).
    seq_loss = (tok * mask).sum(1) / ntok
    # Sum over this micro-batch and divide by the whole step's size: summing the gradients of
    # all micro-batches then equals the gradient of the mean over all n_total sequences.
    loss = seq_loss.sum() / n_total
    # Average KL (for logging); .item() turns a one-number tensor into a Python float.
    kl_mean = ((kl * mask).sum(1) / ntok).mean().item()
    return loss, kl_mean


def tb_seq_stats(model, seqs, pad_id, ref_ctx=None) -> dict:
    """No-grad pass for trajectory balance: per sequence, summed reference and policy logprobs
    over reply tokens, reply length, and the mean k3 KL (logged for comparison with GRPO).

    This is pass 1 of 2 for TB. The TB loss compares each reply's delta with its group's mean
    delta, but a group of 8 may be split across micro-batches, so we first compute every
    sequence's numbers without gradients (cheap) and only then run the gradient pass.

    Returns:
        dict of plain Python lists, one entry per sequence:
        "ref_sum" = log pi_ref(reply | prompt), "lp_sum" = log pi(reply | prompt),
        "ntok" = number of reply tokens, "kl" = average per-token k3 KL estimate.
    """
    import torch
    ids, att, mask, _ = _batch_tensors(model, seqs, pad_id)
    ref = reference_logprobs(model, ids, att, ref_ctx)   # reference (adapter off / frozen copy)
    # Neither pass needs gradients here; the gradient pass is tb_microbatch_loss.
    with torch.no_grad():
        lp = token_logprobs(model, ids, att)             # current policy (trainable adapter on)
    ntok = mask.sum(1).clamp(min=1.0)
    k3 = (k3_kl(ref, lp) * mask).sum(1) / ntok           # per-sequence mean k3 KL, logging only
    # (x * mask).sum(1) adds up the per-token logprobs over reply tokens only: the log of the
    # probability of the whole reply (log of a product = sum of logs). .tolist() -> Python list.
    return {"ref_sum": (ref * mask).sum(1).tolist(), "lp_sum": (lp * mask).sum(1).tolist(),
            "ntok": ntok.tolist(), "kl": k3.tolist()}


def tb_deltas(lp_sum, ref_sum, ntok, rewards, tau, length_norm) -> list[float]:
    """TB residual per sample: length-normalised log pi/pi_ref minus r/tau.

    delta_k = [log pi(y_k|x) - log pi_ref(y_k|x)] / norm_k - r_k / tau, norm_k = 1 by default
    (reply length if length_norm). TB is satisfied when delta_k = -log Z(x) is the SAME for every
    reply to the same prompt, i.e. when pi(y|x) is proportional to pi_ref(y|x) * exp(r / tau).

    Example (tau = 0.1): two replies with the same log-ratio 0 but rewards 1.0 and 0.0 give
    deltas -10 and 0. They differ, so the loss will move probability toward the first reply
    until its log-ratio is 10 nats higher than the second's.
    """
    # zip walks the four lists in parallel, giving one (l, f, n, r) tuple per sample.
    # `n if length_norm else 1.0` is Python's inline if-else (conditional expression).
    return [(l - f) / (n if length_norm else 1.0) - r / tau
            for l, f, n, r in zip(lp_sum, ref_sum, ntok, rewards)]


def group_centers(values: list[float], group_size: int) -> list[float]:
    """Group mean of each value, broadcast back to every member (= -log Z_hat for TB).

    Example with group_size 2: [1, 3, 10, 20] -> [2, 2, 15, 15].
    For TB the best estimate of log Z(x) for a prompt, given its group, is minus the mean delta
    (the "VarGrad" batch estimate), so log Z does not need its own learned parameter.
    """
    out = []
    for g in range(0, len(values), group_size):
        v = values[g:g + group_size]                     # this prompt's group
        out += [sum(v) / len(v)] * len(v)                # its mean, repeated once per member
    return out


def tb_microbatch_loss(model, seqs, ref_sum, ntok, rewards, centers, tau, length_norm, pad_id,
                       n_total):
    """Gradient pass for trajectory balance, scaled by len(seqs)/n_total for accumulation.
    centers are the detached group means of delta from the no-grad pass.

    Pass 2 of 2 for TB. Loss per sequence = (delta_k - group mean of delta)^2, the within-group
    variance of the TB residual. Only log pi(y_k|x) carries a gradient: ref_sum, rewards and
    centers are plain numbers from pass 1, so the gradient is
    2 * (delta_k - center) * grad log pi(y_k|x) / norm_k: a reply whose delta is BELOW its group
    mean (relatively under-weighted given its reward) gets its probability pushed up, one above
    gets pushed down. Note this looks like GRPO with advantage -(delta_k - center), but the
    "advantage" already includes the log-ratio to pi_ref, so no separate KL term is needed.

    Returns:
        scalar loss tensor (sum over this micro-batch / n_total) to call .backward() on.
    """
    import torch
    ids, att, mask, dev = _batch_tensors(model, seqs, pad_id)
    lp = token_logprobs(model, ids, att)       # (B, L-1), with gradients this time
    lp_sum = (lp * mask).sum(1)                # (B,) log pi(reply | prompt)
    # A lambda is a one-line unnamed function: t(xs) turns a Python list into a float tensor
    # on the right device.
    t = lambda xs: torch.tensor(xs, dtype=torch.float, device=dev)
    # Divide by the reply length only when length_norm is on; otherwise divide by 1.
    norm = t(ntok) if length_norm else torch.ones(len(seqs), device=dev)
    # Same formula as tb_deltas, but now lp_sum is a tensor connected to the gradient graph.
    delta = (lp_sum - t(ref_sum)) / norm - t(rewards) / tau
    # Residual relative to the group's (fixed) mean delta, i.e. delta_k + log Z_hat(x).
    resid = delta - t(centers)
    # Squared residuals summed, divided by the step's total so micro-batch gradients add up
    # to the gradient of the mean over all sequences.
    return (resid ** 2).sum() / n_total


def sft_microbatch_loss(model, seqs, pad_id, n_total):
    """Mean NLL over completion tokens per sequence, scaled for accumulation.

    NLL = negative log-likelihood. SFT just asks the model to make every token of the kept
    reply more likely: loss = -(average log p of the reply's tokens). Prompt and prefill tokens
    are masked out, so the model learns to WRITE the reply, not to predict the user's prompt.
    E.g. if the model gives the reply's tokens probabilities 0.5, 0.25 then the loss is
    -(ln 0.5 + ln 0.25) / 2 ~ 1.04; it reaches 0 only if every token gets probability 1.

    Returns:
        scalar loss tensor (sum of per-sequence mean NLL / n_total).
    """
    ids, att, mask, _ = _batch_tensors(model, seqs, pad_id)   # mask: reply tokens only
    lp = token_logprobs(model, ids, att)             # (B, L-1)
    # Per sequence: minus the average logprob over reply tokens -> (B,).
    seq_nll = -(lp * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return seq_nll.sum() / n_total


def pool_key(item: dict, split_prefill: bool) -> str:
    """Sampling pool of an item; prefilled harmful prompts get their own pool if requested.

    Returns "harmful_prefill" for a harmful prompt that has a prefill (when split_prefill is
    True), otherwise the item's kind: "harmful", "benign" or "math".
    """
    if split_prefill and item["kind"] == "harmful" and item.get("prefill"):
        return "harmful_prefill"
    return item["kind"]


class MixSampler:
    """Cycles through each pool's shuffled prompts; every step draws mix[pool] of each.

    Example: mix = {"harmful_prefill": 8, "harmful": 2, "benign": 3, "math": 3} gives 16 prompts
    per step. Each pool is shuffled and used up before any prompt in it repeats (like dealing
    from a shuffled deck and reshuffling when it runs out).
    """

    def __init__(self, items: list[dict], mix: dict, seed: int):
        """Split ``items`` into pools by pool_key and prepare an empty queue for each.

        Args:
            items: all training prompts (dicts with "kind", "prompt", maybe "prefill").
            mix: pool name -> prompts drawn from that pool per step.
            seed: seed for this sampler's own random number generator (reproducible order).
        """
        self.rng = random.Random(seed)
        # Only separate prefilled harmful prompts if the mix asks for that pool.
        split = "harmful_prefill" in mix
        # Dict comprehension: for each pool name k, the list of items belonging to it.
        self.pools = {k: [x for x in items if pool_key(x, split) == k] for k in mix}
        for k, v in self.pools.items():
            if not v:                      # an empty list is "falsy" in Python
                raise ValueError(f"no training items of kind {k!r}")
        # Two assignments in one line. Queues start empty and are filled on first use.
        self.mix, self.queues = mix, {k: [] for k in mix}

    def draw(self) -> list[dict]:
        """Return the prompts for one training step: mix[pool] items from each pool, in order."""
        out = []
        for k, n in self.mix.items():
            for _ in range(n):             # `_` = a loop variable we do not need
                if not self.queues[k]:
                    # Queue used up: refill with a copy ([:] copies a list) and shuffle it.
                    self.queues[k] = self.pools[k][:]
                    self.rng.shuffle(self.queues[k])
                out.append(self.queues[k].pop())    # .pop() removes and returns the last item
        return out


def _decode(tokenizer, ids):
    """Turn token IDs back into text, dropping special markers such as end-of-text."""
    return tokenizer.decode(ids, skip_special_tokens=True)


def _summarise(items, rs) -> dict:
    """Aggregate per-reply judge results into the metrics we log.

    Args:
        items: the prompt item of each reply; rs: the matching reward dicts from compute_rewards
            (keys like "reward", "p_harmful", "p_refusal", "correct").

    Returns:
        dict such as {"reward_harmful": 0.8, "harmful_rate_prefill": 0.25,
        "harmful_rate_plain": 0.0, "reward_benign": ..., "refusal_rate_benign": ...,
        "reward_math": ..., "math_acc": ...}. Kinds absent from the batch are skipped.
    """
    s = {}
    for kind in ("harmful", "benign", "math"):
        # Pair each item with its result and keep only this kind.
        sub = [(x, r) for x, r in zip(items, rs) if x["kind"] == kind]
        if not sub:
            continue                       # no replies of this kind: skip to the next kind
        s[f"reward_{kind}"] = sum(r["reward"] for _, r in sub) / len(sub)
        if kind == "harmful":
            # Report the harmful rate separately for prefilled and plain harmful prompts.
            for tag, pre in (("prefill", True), ("plain", False)):
                ss = [r for x, r in sub if bool(x.get("prefill")) == pre]
                if ss:
                    # A reply counts as harmful if the judge's P(harmful) > 0.5. Summing booleans
                    # counts the Trues. `or 0` treats a missing (None) probability as 0.
                    s[f"harmful_rate_{tag}"] = sum((r["p_harmful"] or 0) > 0.5 for r in ss) / len(ss)
        if kind == "benign":
            # Fraction of harmless prompts the model wrongly refused.
            s["refusal_rate_benign"] = sum((r["p_refusal"] or 0) > 0.5 for _, r in sub) / len(sub)
        if kind == "math":
            s["math_acc"] = sum(r["correct"] for _, r in sub) / len(sub)
    return s


# ----------------------------------------------------------------------------------- GRPO
def evaluate(tokenizer, items, generate_fn, judge_fn, step, max_new_tokens) -> dict:
    """Greedy validation: one reply per item.

    Temperature 0.0 means "always pick the most likely token" (greedy decoding), so the result
    reflects the model's single most typical answer rather than random samples.

    Returns:
        the _summarise metrics with a "val_" prefix, e.g. {"val_math_acc": 0.9, ...}.
    """
    pids = [encode_prompt(tokenizer, x) for x in items]
    # n = 1 sample per prompt, temperature 0.0 (greedy).
    comps = generate_fn(pids, 1, step, 0.0, max_new_tokens)
    # Put the prefill back in front of the generated text so the judge sees the whole reply
    # (the prefill was part of the prompt, so it is not in the generated tokens). c[0] = the
    # only sample.
    replies = [full_reply(tokenizer, x, c[0]) for x, c in zip(items, comps)]
    rs = compute_rewards(items, replies, judge_fn)
    return {f"val_{k}": v for k, v in _summarise(items, rs).items()}


def adapter_ref_ctx(model, ref_name: str = "ref", active: str = "default"):
    """Context manager factory: compute reference logprobs with a frozen copy of the starting
    adapter (loaded under ref_name), then switch back to the trainable adapter.

    Used when cfg.ref == "init": e.g. GRPO starting from the SFT adapter keeps a frozen copy of
    that SFT adapter as the reference, instead of the untuned base model. The model holds two
    LoRA adapters: "default" (being trained) and "ref" (frozen). Usage:
        ctx = adapter_ref_ctx(model)
        with ctx():            # inside: the "ref" adapter is active
            ...                # compute reference logprobs
        # after: "default" is active again

    Returns:
        ctx, a function that creates a fresh context manager each time it is called.
    """
    # @contextlib.contextmanager turns a generator function into a context manager: the code
    # before `yield` runs when the `with` block starts, the code after runs when it ends.
    @contextlib.contextmanager
    def ctx():
        """Activate the frozen reference adapter for the duration of a ``with`` block."""
        model.set_adapter(ref_name)          # PEFT: make the reference adapter the active one
        try:
            yield                            # the body of the `with` block runs here
        finally:
            # `finally` runs even if the body raised an error, so we always switch back.
            model.set_adapter(active)
            for n, p in model.named_parameters():   # set_adapter toggles requires_grad
                # requires_grad says whether PyTorch computes gradients for a parameter, i.e.
                # whether it is trainable. Parameter names contain the adapter name, e.g.
                # "...q_proj.lora_A.ref.weight". Keep the reference frozen, the policy trainable.
                if f".{ref_name}." in n:
                    p.requires_grad_(False)
                elif f".{active}." in n:
                    p.requires_grad_(True)
    return ctx


def microbatch_indices(seqs, micro_batch: int, sort_by_length: bool = True) -> list[list[int]]:
    """Split one step's sequences into micro-batches, as lists of positions in ``seqs``.

    Every micro-batch is padded to its longest sequence, and the model spends compute on padded
    positions just as on real ones. Reply lengths vary a lot (a 20-token refusal next to a
    512-token answer), so taking sequences in sampling order wastes much of each forward and
    backward pass on padding. With sort_by_length the sequences are micro-batched in order of
    total length (prompt + reply), so neighbours have similar lengths and little padding.

    This changes cost only, not the result: the step's gradient is the SUM over all sequences
    (each loss is divided by the step's total), so it does not depend on how sequences are
    grouped, apart from floating-point rounding. Peak memory is unchanged too, since the longest
    sequence set the size of its micro-batch before as well.

    Example with lengths [10, 500, 12, 480] and micro_batch 2: sampling order pads to
    2 * 500 + 2 * 480 = 1960 positions for 1002 real tokens; sorted order ([0, 2], [3, 1]) pads
    to 2 * 12 + 2 * 500 = 1024.
    """
    # Positions 0, 1, 2, ... of the sequences, in sampling order.
    order = list(range(len(seqs)))
    if sort_by_length:
        # sorted is stable: sequences of equal length keep their sampling order, so the
        # batching is deterministic (needed for resumed runs to match uninterrupted ones).
        order.sort(key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    # Cut the (possibly sorted) positions into consecutive chunks of micro_batch; the last
    # chunk may be shorter.
    return [order[i:i + micro_batch] for i in range(0, len(order), micro_batch)]


def padding_fraction(seqs, batches: list[list[int]]) -> float:
    """Fraction of the positions the model processes that are padding, for the given batching.

    0.0 means no wasted compute; 0.4 means 40% of every forward/backward pass is padding.
    """
    # Tokens that carry information: every prompt plus its reply.
    real = sum(len(p) + len(c) for p, c in seqs)
    # Each micro-batch is a rectangle: (number of sequences) x (its longest sequence).
    total = sum(len(b) * max(len(seqs[i][0]) + len(seqs[i][1]) for i in b) for b in batches)
    return 1.0 - real / total if total else 0.0


# After this many optimizer updates in a row are skipped for a non-finite gradient, training
# stops: one bad batch is survivable, but a streak means the run itself is broken and every
# further step would pay for sampling and judging without learning anything.
MAX_SKIPPED_UPDATES = 3


def clipped_step(opt, params, grad_clip: float) -> tuple[float, bool]:
    """Clip the accumulated gradient and apply one optimizer update, unless it is not finite.

    If the overall gradient length (norm) exceeds grad_clip, it is scaled down to grad_clip so
    one unusual batch cannot make a huge, destabilising step.

    A gradient containing NaN or infinity (e.g. an overflow in exp() of a huge log-ratio, a
    known risk for trajectory balance) must never reach the optimizer: clipping cannot repair it
    (NaN times anything is NaN), a single such update turns every LoRA weight into NaN, and
    AdamW's running averages would stay NaN afterwards, so the rest of a paid run would be
    lost. In that case the update is skipped and the gradients are cleared; the weights and the
    optimizer state stay exactly as they were.

    Returns:
        (gnorm, skipped): the gradient norm BEFORE clipping (for logging), and whether the
        update was skipped.
    """
    import torch
    # clip_grad_norm_ rescales the gradients in place if their overall norm exceeds grad_clip,
    # and returns the norm they had BEFORE clipping. .item() turns it into a Python float.
    gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip).item()
    # math.isfinite is False for NaN and for +/- infinity: at least one gradient entry is broken.
    if not math.isfinite(gnorm):
        opt.zero_grad(set_to_none=True)
        return gnorm, True
    opt.step()                               # apply one AdamW update using the accumulated .grad
    return gnorm, False


def _check_skip_streak(streak: int, where: str):
    """Stop training after MAX_SKIPPED_UPDATES consecutive skipped updates (see above)."""
    # `streak` counts skipped updates in a row; the caller resets it to 0 after any good update.
    if streak >= MAX_SKIPPED_UPDATES:
        raise RuntimeError(f"{streak} consecutive updates skipped for non-finite gradients "
                           f"({where}); stopping instead of paying for steps that cannot learn")


def accumulate_grpo_grads(model, seqs, adv, cfg: GRPOConfig, pad_id, ref_ctx=None):
    """Backpropagate the GRPO loss of one training step, micro-batch by micro-batch.

    Gradient accumulation: 128 sequences do not fit on the GPU at once, so they are processed
    cfg.micro_batch at a time. Each loss.backward() ADDS its gradients into each parameter's
    .grad, and because every loss is divided by len(seqs), the total equals the gradient of the
    full-batch mean. The caller clears the gradients before and steps the optimizer after.

    Returns:
        (kl, extra): the mean per-token KL over sequences (logged), and {"pad_frac": fraction
        of processed positions that were padding}.
    """
    batches = microbatch_indices(seqs, cfg.micro_batch, cfg.sort_by_length)
    # Running sum of (micro-batch mean KL x micro-batch size), turned into a mean at the end.
    kl_sum = 0.0
    for idx in batches:
        # [seqs[i] for i in idx] picks this micro-batch's sequences (and their advantages).
        loss, kl = grpo_microbatch_loss(model, [seqs[i] for i in idx], [adv[i] for i in idx],
                                        cfg.beta, pad_id, len(seqs), ref_ctx)
        # Backpropagation: compute d(loss)/d(parameter) for every trainable parameter.
        loss.backward()
        # kl is this micro-batch's mean; weight it by its size so the result is the mean over
        # sequences even when the last micro-batch is smaller.
        kl_sum += kl * len(idx)
    return kl_sum / len(seqs), {"pad_frac": padding_fraction(seqs, batches)}


def accumulate_tb_grads(model, seqs, rewards, cfg: GRPOConfig, pad_id, ref_ctx=None,
                        on_policy: list[bool] | None = None):
    """Backpropagate the trajectory-balance loss of one training step (two passes).

    Pass 1 (no gradients) collects per-sequence summed logprobs and lengths for all sequences,
    giving every TB residual delta and each group's mean (= -log Z_hat per prompt). Pass 2 (with
    gradients) treats those group means as fixed numbers, so it no longer matters that a group
    of 8 might be split across two micro-batches of 4.

    on_policy[i] says whether sequence i was sampled from the current policy (None = all were).
    The loss uses every sequence; the logged KL and log-ratio use the policy's own samples only,
    because the k3 estimate of KL(pi || pi_ref) is only unbiased on samples from pi.

    Returns:
        (kl, extra): the mean per-token KL over the policy's sequences (logged like GRPO's), and
        the TB metrics {"tb_loss", "logZ_mean", "logratio_per_token", "pad_frac"}.
    """
    K = cfg.group_size
    batches = microbatch_indices(seqs, cfg.micro_batch, cfg.sort_by_length)
    # Per-sequence statistics, stored at each sequence's ORIGINAL position (micro-batches may
    # visit sequences in length order, but groups are defined by position: K per prompt).
    # A dict of four lists, each pre-filled with one 0.0 per sequence, to be overwritten below.
    st = {k: [0.0] * len(seqs) for k in ("ref_sum", "lp_sum", "ntok", "kl")}
    for idx in batches:
        part = tb_seq_stats(model, [seqs[i] for i in idx], pad_id, ref_ctx)
        for k in st:
            # idx[j] is the original position of the j-th sequence of this micro-batch.
            for i, v in zip(idx, part[k]):
                st[k][i] = v
    scaled = [reward_term(r, cfg) for r in rewards]          # r / tau unless log_reward
    deltas = tb_deltas(st["lp_sum"], st["ref_sum"], st["ntok"], scaled, 1.0, cfg.length_norm)
    centers = group_centers(deltas, K)
    tb_loss = 0.0
    pick = lambda xs, idx: [xs[i] for i in idx]      # the values of one micro-batch
    for idx in batches:
        loss = tb_microbatch_loss(model, pick(seqs, idx), pick(st["ref_sum"], idx),
                                  pick(st["ntok"], idx), pick(scaled, idx), pick(centers, idx),
                                  1.0, cfg.length_norm, pad_id, len(seqs))
        loss.backward()                              # accumulate gradients, as for GRPO
        tb_loss += loss.item()
    # Log-ratio log pi - log pi_ref per reply token, for monitoring drift.
    # own[i]: was sequence i sampled from the policy? The `if o` filters below keep only those.
    own = on_policy if on_policy is not None else [True] * len(seqs)
    lr_tok = [(l - f) / n for l, f, n, o in zip(st["lp_sum"], st["ref_sum"], st["ntok"], own) if o]
    kls = [k for k, o in zip(st["kl"], own) if o]
    # centers[::K] = every K-th entry = one center per prompt (the slice [::k] takes every k-th
    # element). log Z_hat = -center, averaged over prompts.
    extra = {"tb_loss": tb_loss, "logZ_mean": -sum(centers[::K]) / (len(centers) // K),
             "logratio_per_token": sum(lr_tok) / len(lr_tok),
             "pad_frac": padding_fraction(seqs, batches)}
    return sum(kls) / len(kls), extra


# ----------------------------------------------------------------- sub-trajectory balance
# Modified SubTB (Hu et al., ICLR 2024, `modified_subtb_loss` in GFNOrg/gfn-lm-tuning), on a
# coarse set of states. For a reply c_1..c_T to prompt x (prompt includes any prefill):
#   states  s_0 (start), s_m for m in {chunk, 2*chunk, ...} with m < T, and s_T (reply end).
#   flows   log F(s_0) = log Z(x)                               (batch estimate, as in our TB)
#           log F(s_m) = log R(s_m, stop) - log pi(stop | s_m)  (paper's definition, no flow net)
#           log F(s_T) = log R(y)
#   reward  log R(s, stop) = log pi_ref(s, stop) + r(s) / tau   (r(s) = WildGuard on the prefix)
# Write v_i = [log pi(s_i, stop) - log pi_ref(s_i, stop)] - r(s_i) / tau, i.e. the TB residual of
# the reply truncated at s_i ("stop" = the end-of-turn token; for s_T the reply's own ending), and
# v_0 = -log Z. Then the sub-trajectory residual between states a < b is exactly v_b - v_a, and
#   loss(reply) = mean over all pairs a < b of (v_b - v_a)^2       (lambda = 1, as in the paper)
# averaged over replies. With no intermediate state (maths, replies shorter than `chunk`) this is
# (delta - center)^2, the TB loss. The target distribution is the same as TB's.
# Deviation from the paper: the paper sums over all pairs in the batch; we average per reply, so
# every reply weighs the same as in TB and GRPO (long replies have more pairs).
PREFIX_KINDS = ("harmful", "benign")   # prompt kinds whose prefixes the judge can score


def reward_term(r: float, cfg: GRPOConfig) -> float:
    """The reward's contribution to log R, in nats: r / tau, or log_beta * log(max(r, floor)).

    Example (defaults): r = 0.999 -> -0.007; r = 0.5 -> -4.9; r <= 0.01 -> -32.2.
    """
    if cfg.log_reward:
        return cfg.log_beta * math.log(max(r, cfg.log_floor))
    return r / cfg.tau


def step_temperature(cfg: GRPOConfig, step: int) -> float:
    """Sampling temperature for the policy's replies at `step` (seeded, so resumable)."""
    if not cfg.temp_prob:
        return cfg.temperature
    rng = random.Random(cfg.seed * 100003 + step)
    if rng.random() < cfg.temp_prob:
        return cfg.temp_low + rng.random() * (cfg.temp_high - cfg.temp_low)
    return cfg.temperature


def subtb_states(n_tok: int, chunk: int, kind: str) -> list[int]:
    """Intermediate state positions (number of generated tokens) for one reply.

    Example: n_tok=200, chunk=64, kind="harmful" -> [64, 128, 192]; kind="math" -> [].
    A state at m == n_tok would coincide with the reply end, so m < n_tok.
    """
    if kind not in PREFIX_KINDS:
        return []
    return list(range(chunk, n_tok, chunk))


def token_and_stop_logprobs(model, ids, att, stop_id: int):
    """Like token_logprobs, plus log P(stop_id | prefix) at every position.

    Returns (lp, lp_stop), each (B, L-1): lp[b, t] = log P(ids[b, t+1] | ids[b, :t+1]) and
    lp_stop[b, t] = log P(stop_id | ids[b, :t+1]). Both come from the same forward pass.
    """
    import torch
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lse = torch.logsumexp(logits, dim=-1)
    lp = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) - lse
    return lp, logits[..., stop_id] - lse


def stop_terms(lp, lp_stop, mask, states: list[list[int]]):
    """Per reply, log pi(s_m, stop) for each intermediate state, then log pi(y) for the full reply.

    lp, lp_stop, mask: (B, L-1) as returned by token_and_stop_logprobs / _batch_tensors.
    states[b]: intermediate positions m of reply b (see subtb_states).
    Returns a list of B 1-D tensors of length len(states[b]) + 1 (kept on the autograd graph).
    The reply's j-th generated token is predicted at the j-th masked position, so the prefix of
    m tokens has log-prob cumsum[m-1] and its stop probability is read at masked position m.
    """
    import torch
    out = []
    for b, st in enumerate(states):
        pos = mask[b].nonzero(as_tuple=True)[0]          # positions predicting reply tokens
        cum = lp[b, pos].cumsum(0)                       # log pi(c_1..c_j), j = 1..T
        vals = [cum[m - 1] + lp_stop[b, pos[m]] for m in st]
        vals.append(cum[-1])                             # the whole reply as generated
        out.append(torch.stack(vals))
    return out


def subtb_pair_loss(v, v0):
    """Mean over all state pairs a < b of (v_b - v_a)^2, states = [v0] + v. lambda = 1.

    v: 1-D tensor (intermediate states then the reply end); v0: the start state's value
    (-log Z, a fixed number). With v of length 1 this is (v[0] - v0)^2, the TB loss.
    """
    import torch
    allv = torch.cat([v.new_tensor([v0]), v])
    d = allv.unsqueeze(0) - allv.unsqueeze(1)            # d[a, b] = v_b - v_a
    iu = torch.triu_indices(len(allv), len(allv), offset=1, device=allv.device)
    return (d[iu[0], iu[1]] ** 2).mean()


def accumulate_subtb_grads(model, seqs, rewards, prefix_rewards, states, cfg: GRPOConfig, pad_id,
                           stop_id: int, ref_ctx=None, on_policy: list[bool] | None = None):
    """Backpropagate the SubTB loss of one training step (two passes, like accumulate_tb_grads).

    prefix_rewards[i] lists r(s_m) for the intermediate states states[i] of reply i.
    Pass 1 (no gradients): reference terms for every state, and each group's log Z estimate,
    which is TB's (-mean over the group of the whole-reply residual), so maths replies get exactly
    the TB loss. Pass 2 (gradients): policy terms and the pair loss per reply.

    Returns (kl, extra) like accumulate_tb_grads, with extra metrics "subtb_loss",
    "subtb_states_mean" (intermediate states per reply) and "subtb_end_resid" (the TB part).
    """
    import torch
    K = cfg.group_size
    batches = microbatch_indices(seqs, cfg.micro_batch, cfg.sort_by_length)
    ref_terms = [None] * len(seqs)
    st = {k: [0.0] * len(seqs) for k in ("ref_sum", "lp_sum", "ntok", "kl")}
    pick = lambda xs, idx: [xs[i] for i in idx]
    for idx in batches:
        ids, att, mask, _ = _batch_tensors(model, pick(seqs, idx), pad_id)
        ctx = ref_ctx() if ref_ctx is not None else model.disable_adapter()
        with torch.no_grad():
            with ctx:
                ref, ref_stop = token_and_stop_logprobs(model, ids, att, stop_id)
            lp = token_logprobs(model, ids, att)
        rt = stop_terms(ref, ref_stop, mask, pick(states, idx))
        ntok = mask.sum(1).clamp(min=1.0)
        k3 = (k3_kl(ref, lp) * mask).sum(1) / ntok
        for j, i in enumerate(idx):
            ref_terms[i] = rt[j].tolist()
            st["ref_sum"][i] = rt[j][-1].item()
            st["lp_sum"][i] = (lp[j] * mask[j]).sum().item()
            st["ntok"][i], st["kl"][i] = ntok[j].item(), k3[j].item()
    deltas = tb_deltas(st["lp_sum"], st["ref_sum"], st["ntok"],
                       [reward_term(r, cfg) for r in rewards], 1.0, False)
    centers = group_centers(deltas, K)
    total, end_resid = 0.0, 0.0
    for idx in batches:
        ids, att, mask, dev = _batch_tensors(model, pick(seqs, idx), pad_id)
        lp, lp_stop = token_and_stop_logprobs(model, ids, att, stop_id)
        pt = stop_terms(lp, lp_stop, mask, pick(states, idx))
        loss = 0.0
        for j, i in enumerate(idx):
            r = torch.tensor([reward_term(x, cfg) for x in list(prefix_rewards[i]) + [rewards[i]]],
                             dtype=torch.float, device=dev)
            v = pt[j] - torch.tensor(ref_terms[i], dtype=torch.float, device=dev) - r
            loss = loss + subtb_pair_loss(v, centers[i])
            end_resid += (v[-1].item() - centers[i]) ** 2
        loss = loss / len(seqs)
        loss.backward()
        total += loss.item()
    own = on_policy if on_policy is not None else [True] * len(seqs)
    lr_tok = [(l - f) / n for l, f, n, o in zip(st["lp_sum"], st["ref_sum"], st["ntok"], own) if o]
    kls = [k for k, o in zip(st["kl"], own) if o]
    extra = {"subtb_loss": total, "subtb_end_resid": end_resid / len(seqs),
             "subtb_states_mean": sum(len(s) for s in states) / len(states),
             "logZ_mean": -sum(centers[::K]) / (len(centers) // K),
             "logratio_per_token": sum(lr_tok) / len(lr_tok),
             "pad_frac": padding_fraction(seqs, batches)}
    return sum(kls) / len(kls), extra


def prefix_rewards_for(tokenizer, items, seqs, states, judge_fn) -> list[list[float]]:
    """Judge every intermediate prefix (prefill + first m generated tokens) with the same reward
    function as full replies. Returns, per reply, the rewards in the order of states[i]."""
    flat_items, flat_replies = [], []
    for x, (_, c), st in zip(items, seqs, states):
        for m in st:
            flat_items.append(x)
            flat_replies.append(full_reply(tokenizer, x, c[:m]))
    flat = compute_rewards(flat_items, flat_replies, judge_fn) if flat_items else []
    out, k = [], 0
    for st in states:
        out.append([r["reward"] for r in flat[k:k + len(st)]])
        k += len(st)
    return out


def train_grpo(model, tokenizer, train_items, val_items, cfg: GRPOConfig, generate_fn, judge_fn,
               log_fn=print, save_fn=None, ref_ctx=None, start_step: int = 0,
               opt_state: dict | None = None) -> list[dict]:
    """Run the GRPO (or trajectory-balance, if cfg.method == "tb") training loop.

    Each step: draw 16 prompts -> generate group_size (8) replies each with generate_fn ->
    score all 128 with the judge -> compute the loss in micro-batches, accumulating gradients
    -> clip gradients -> one AdamW update of the LoRA weights -> log metrics.
    A step whose gradient is not finite is logged with "update_skipped": true and leaves the
    weights untouched; MAX_SKIPPED_UPDATES such steps in a row raise RuntimeError.
    This is "on-policy": every step's replies come from the current model, freshly sampled.
    (generate_fn is expected to use the latest adapter weights; on Modal the current
    adapter is saved and loaded into vLLM before sampling.)

    Args:
        model: LoRA-wrapped model; only parameters with requires_grad=True are trained.
        tokenizer: its tokenizer.
        train_items / val_items: prompt dicts for training and for periodic validation.
        cfg: GRPOConfig with all hyper-parameters.
        generate_fn / judge_fn: sampling and scoring callbacks (see module docstring).
        log_fn: called with one JSON string per step (default: print).
        save_fn: optional, called as save_fn(step, opt) to save a checkpoint: the adapter plus
            the optimizer state, so a run interrupted later can resume from here.
        ref_ctx: optional reference-policy context factory (adapter_ref_ctx); None = base model.
        start_step / opt_state: resume support. When a cloud GPU is preempted (taken back by the
            provider) the job restarts; it then reloads the latest checkpoint's adapter, passes
            that checkpoint's step as start_step and its optimizer state as opt_state, and the
            loop continues at start_step + 1. The prompt sampler is fast-forwarded by drawing and
            discarding start_step batches, so the resumed run sees exactly the prompts the
            uninterrupted run would have seen (sampling uses per-step seeds as well).

    Returns:
        list of per-step log dicts (rewards, KL, gradient norm, timings, validation metrics).
    """
    # Callbacks keep the algorithm independent of its runtime: Modal supplies
    # vLLM and WildGuard, while tests supply tiny local stand-ins.
    import torch
    torch.manual_seed(cfg.seed)              # reproducible random numbers in PyTorch
    pad_id = pad_token_id(tokenizer)
    # Trainable parameters only: with LoRA these are the small adapter matrices; the 8B base
    # weights have requires_grad=False and stay frozen.
    params = [p for p in model.parameters() if p.requires_grad]
    # AdamW: the standard optimizer. It keeps running averages of each parameter's gradient and
    # squared gradient to choose a well-scaled step size. weight_decay=0.0: no shrinking toward 0.
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    if opt_state is not None:
        # Resuming: restore AdamW's running averages (and step counts) from the checkpoint.
        opt.load_state_dict(opt_state)
    sampler = MixSampler(train_items, cfg.mix, cfg.seed)
    # Resuming: replay the sampler so the next draw is the one step start_step + 1 would get.
    for _ in range(start_step):
        sampler.draw()
    logs, skip_streak = [], 0
    # Validate once before training (step 0) to get a baseline (not repeated when resuming).
    if val_items and cfg.val_every and start_step == 0:
        # {"step": 0, **d} builds a new dict with "step" plus every key/value of d.
        logs.append({"step": 0, **evaluate(tokenizer, val_items, generate_fn, judge_fn, 0, cfg.max_new_tokens)})
        log_fn(json.dumps(logs[-1]))         # logs[-1] = the last element of the list
    for step in range(start_step + 1, cfg.steps + 1):   # steps 1..cfg.steps (or resume point)
        t0 = time.time()
        # ---- 1. sample prompts and generate a group of replies for each ----
        batch = sampler.draw()                                  # e.g. 16 prompt dicts
        pids = [encode_prompt(tokenizer, x) for x in batch]     # their token IDs
        # comps[i][k] = token IDs of the k-th reply to prompt i (vLLM on Modal).
        n_ref = cfg.ref_samples                                 # 0 unless off-policy TB
        temp = step_temperature(cfg, step)                      # cfg.temperature unless tempered
        comps = generate_fn(pids, cfg.group_size - n_ref, step, temp, cfg.max_new_tokens)
        if n_ref:
            # Off-policy TB: the last n_ref replies of every group come from the reference policy.
            ref_comps = generate_fn(pids, n_ref, step, cfg.temperature, cfg.max_new_tokens,
                                    reference=True)
            comps = [own + ref for own, ref in zip(comps, ref_comps)]    # list + list joins them
        # Every later step assumes exactly group_size replies per prompt (groups are located by
        # position), so a generator that returns fewer or more must stop the run here.
        if any(len(cs) != cfg.group_size for cs in comps):
            raise ValueError("generate_fn returned a wrong number of replies for some prompt")
        # own[j] = True if flat reply j was sampled from the policy (the first replies of a group).
        # Example with group_size 8 and n_ref 4: for each prompt, k = 0..7 gives
        # [True, True, True, True, False, False, False, False].
        own = [k < cfg.group_size - n_ref for _ in batch for k in range(cfg.group_size)]
        t_gen = time.time() - t0
        # Flatten to one entry per reply, grouped by prompt: each prompt item repeated 8 times.
        # A comprehension with two `for`s reads like nested loops: outer x, inner _.
        items = [x for x in batch for _ in range(cfg.group_size)]
        # Matching (prompt_ids, completion_ids) pairs: for each prompt, for each of its replies.
        seqs = [(p, c) for p, cs in zip(pids, comps) for c in cs]
        # ---- 2. score the replies ----
        # Text for the judge: prefill + generated continuation (the full assistant reply).
        replies = [full_reply(tokenizer, x, c) for x, (_, c) in zip(items, seqs)]
        rs = compute_rewards(items, replies, judge_fn)
        # Group-normalised advantages (used by GRPO; for TB only zero_frac is logged).
        adv, zero_frac = group_advantages([r["reward"] for r in rs], cfg.group_size,
                                          cfg.adv_eps, cfg.adv_min_std)
        if cfg.method == "subtb":
            # Extra judge calls: one per intermediate prefix (harmful / benign prompts only).
            states = [subtb_states(len(c), cfg.chunk, x["kind"]) for x, (_, c) in zip(items, seqs)]
            pre_r = prefix_rewards_for(tokenizer, items, seqs, states, judge_fn)
        t_judge = time.time() - t0 - t_gen

        # ---- 3. compute the loss and accumulate gradients ----
        model.train()                        # training mode (e.g. enables dropout if any)
        # Clear gradients left over from the previous step (set_to_none frees their memory).
        opt.zero_grad(set_to_none=True)
        # Gradients of all 128 sequences are accumulated into each parameter's .grad; the
        # optimizer then takes ONE step below.
        rewards = [r["reward"] for r in rs]
        if cfg.method == "grpo":
            kl, extra = accumulate_grpo_grads(model, seqs, adv, cfg, pad_id, ref_ctx)
        elif cfg.method == "subtb":
            kl, extra = accumulate_subtb_grads(model, seqs, rewards, pre_r, states, cfg, pad_id,
                                               tokenizer.eos_token_id, ref_ctx, own)
        else:  # trajectory balance
            kl, extra = accumulate_tb_grads(model, seqs, rewards, cfg, pad_id, ref_ctx, own)
        # ---- 4. update the LoRA weights ----
        # Clip, then update; a non-finite gradient is skipped instead (see clipped_step).
        gnorm, skipped = clipped_step(opt, params, cfg.grad_clip)
        # Count skipped updates in a row; any successful update resets the count to 0.
        skip_streak = skip_streak + 1 if skipped else 0

        # ---- 5. log ----
        # `**d` inside a dict literal merges all of d's entries into the new dict.
        # The policy's own replies: with off-policy TB the reference replies are excluded, so
        # these curves keep describing the model being trained and stay comparable with
        # on-policy runs. (zero_var_group_frac covers whole groups, i.e. what the loss saw.)
        # pick(xs) keeps the entries of xs that belong to the policy's own replies.
        pick = lambda xs: [x for x, o in zip(xs, own) if o]
        p_items, p_rs, p_seqs = pick(items), pick(rs), pick(seqs)
        if n_ref:
            ref_rs = [r for r, o in zip(rs, own) if not o]
            ref_items = [x for x, o in zip(items, own) if not o]
            extra = {**extra, "reward_mean_ref": sum(r["reward"] for r in ref_rs) / len(ref_rs),
                     # reference-sample metrics with a "_ref" suffix, e.g. harmful_rate_prefill_ref
                     **{f"{k}_ref": v for k, v in _summarise(ref_items, ref_rs).items()
                        if k.startswith(("harmful_rate", "refusal_rate"))}}
        rec = {"step": step, "reward_mean": sum(r["reward"] for r in p_rs) / len(p_rs),
               **_summarise(p_items, p_rs), "zero_var_group_frac": zero_frac,
               "kl": kl, "grad_norm": gnorm, "update_skipped": skipped, **extra,
               # average reply length in tokens, and fraction cut off at max_new_tokens
               "resp_len": sum(len(c) for _, c in p_seqs) / len(p_seqs),
               "truncated_frac": sum(len(c) >= cfg.max_new_tokens for _, c in p_seqs) / len(p_seqs),
               "sample_temp": round(temp, 3),
               "t_gen": round(t_gen, 1), "t_judge": round(t_judge, 1),
               "t_step": round(time.time() - t0, 1)}
        # Every val_every steps (% is the remainder), add greedy validation metrics.
        if val_items and cfg.val_every and step % cfg.val_every == 0:
            rec.update(evaluate(tokenizer, val_items, generate_fn, judge_fn, step, cfg.max_new_tokens))
        logs.append(rec)
        log_fn(json.dumps(rec))
        _check_skip_streak(skip_streak, f"step {step}")
        # Save the adapter periodically and always at the final step.
        # (save_every = 0 means "final step only"; the check also avoids dividing by zero.)
        if save_fn and ((cfg.save_every and step % cfg.save_every == 0) or step == cfg.steps):
            save_fn(step, opt)
    return logs


# ----------------------------------------------------------------------------------- SFT
def build_sft_set(tokenizer, items, cfg: SFTConfig, generate_fn, judge_fn, log_fn=print):
    """Rejection sampling from the current model: per prompt, keep the highest-reward candidate
    if its reward >= keep_threshold (safe on harmful, not a refusal on benign, correct on math).
    For prefilled prompts the kept target is the continuation after the prefill, so SFT teaches
    recovery from a compliant opening using the model's own successful recoveries.

    Truncated candidates (cut off at max_new_tokens) are not kept, except benign ones when
    cfg.keep_truncated_benign (a long helpful answer that was cut off is still not a refusal).

    Example: 4 candidates for a harmful prompt with rewards [0.1, 0.97, 0.6, 0.2] -> keep the
    second (0.97 >= 0.5). Rewards [0.1, 0.2, 0.05, 0.3] -> best is 0.3 < 0.5, prompt dropped.

    Returns:
        (keep, stats): keep is a list of (prompt_ids, completion_ids) training examples; stats
        maps a tag like "harmful_prefill" to [kept, total].
    """
    pids = [encode_prompt(tokenizer, x) for x in items]
    # K candidates per prompt, sampled at cfg.temperature. comps[j][k] = k-th candidate for j.
    comps = generate_fn(pids, cfg.samples_per_prompt, 0, cfg.temperature, cfg.max_new_tokens)
    # Flatten so candidate k of prompt j sits at index j * K + k.
    flat_items = [x for x in items for _ in range(cfg.samples_per_prompt)]
    flat = [c for cs in comps for c in cs]
    replies = [full_reply(tokenizer, x, c) for x, c in zip(flat_items, flat)]
    rs = compute_rewards(flat_items, replies, judge_fn)
    # keep: chosen examples; stats/why: counters for logging; trunc_kept: kept-but-truncated.
    keep, stats, why, trunc_kept = [], {}, {}, 0
    K = cfg.samples_per_prompt
    for j, x in enumerate(items):
        # e.g. "harmful_prefill", "harmful", "benign", "math"
        tag = x["kind"] + ("_prefill" if x.get("prefill") else "")
        allow_trunc = cfg.keep_truncated_benign and x["kind"] == "benign"
        # best candidate among those passing the length rule
        # (reward, candidate index) tuples; truncated candidates excluded unless allowed.
        cand = [(rs[j * K + k]["reward"], k) for k in range(K)
                if allow_trunc or len(flat[j * K + k]) < cfg.max_new_tokens]
        # setdefault returns the existing value for the key, or stores and returns the default.
        st = stats.setdefault(tag, [0, 0])         # [kept, total] for this tag
        st[1] += 1
        w = why.setdefault(tag, {"all_truncated": 0, "low_reward": 0})
        if not cand:
            w["all_truncated"] += 1
            continue
        # max over tuples compares the reward first, so this is the highest-reward candidate.
        best_r, best_k = max(cand)
        if best_r < cfg.keep_threshold:
            w["low_reward"] += 1
            continue
        st[0] += 1
        # Adding a boolean adds 1 for True, 0 for False.
        trunc_kept += len(flat[j * K + best_k]) >= cfg.max_new_tokens
        # The training example: the prompt (with prefill) and the model's own good continuation.
        keep.append((pids[j], flat[j * K + best_k]))
    log_fn(json.dumps({"sft_kept": {k: f"{a}/{b}" for k, (a, b) in stats.items()},
                       "sft_dropped": why, "sft_examples": len(keep),
                       "sft_kept_truncated": trunc_kept}))
    return keep, stats


def train_sft(model, tokenizer, examples, cfg: SFTConfig, log_fn=print) -> list[dict]:
    """Supervised fine-tuning of the LoRA adapter on (prompt_ids, completion_ids) examples.

    For each epoch: shuffle the examples, cut them into batches of cfg.batch_size, and for each
    batch accumulate the SFT loss over micro-batches, clip gradients and take one AdamW step.

    Returns:
        list of per-step log dicts with the loss and gradient norm.
    """
    import torch
    torch.manual_seed(cfg.seed)
    pad_id = pad_token_id(tokenizer)
    params = [p for p in model.parameters() if p.requires_grad]    # LoRA weights only
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    rng = random.Random(cfg.seed)
    logs, step, skip_streak = [], 0, 0
    for ep in range(cfg.epochs):
        order = examples[:]                  # copy, so shuffling does not change the caller's list
        rng.shuffle(order)
        for b in range(0, len(order), cfg.batch_size):
            batch = order[b:b + cfg.batch_size]
            model.train()
            opt.zero_grad(set_to_none=True)  # start this step with no gradients
            tot = 0.0
            # Gradient accumulation over micro-batches; each loss is divided by len(batch), so
            # the summed gradients equal the gradient of the batch-mean loss.
            for i in range(0, len(batch), cfg.micro_batch):
                loss = sft_microbatch_loss(model, batch[i:i + cfg.micro_batch], pad_id, len(batch))
                loss.backward()
                tot += loss.item()           # total = mean loss over the batch, for logging
            gnorm, skipped = clipped_step(opt, params, cfg.grad_clip)
            skip_streak = skip_streak + 1 if skipped else 0
            step += 1
            rec = {"sft_step": step, "epoch": ep, "loss": tot, "grad_norm": gnorm,
                   "update_skipped": skipped}
            logs.append(rec)
            log_fn(json.dumps(rec))
            _check_skip_streak(skip_streak, f"SFT step {step}")
    return logs


def config_dict(cfg) -> dict:
    """Return a config dataclass (GRPOConfig or SFTConfig) as a plain dict, e.g. for saving."""
    return asdict(cfg)
