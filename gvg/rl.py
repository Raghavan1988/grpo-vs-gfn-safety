"""GRPO and rejection-sampling SFT for LoRA fine-tuning, independent of where they run.

Generation and judging are passed in as functions, so the same loop runs on Modal (vLLM +
WildGuard) and in CPU tests (tiny model + fake judge).

    generate_fn(prompt_ids: list[list[int]], n: int, step: int, temperature: float,
                max_new_tokens: int) -> list[list[list[int]]]     # [prompt][sample] -> completion ids
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
a 5-50 nat preference. So it is off by default; tau then has its KL-RL meaning (beta = tau). log Z(x) is not learned: it is estimated
from the group as log Z_hat = -mean_k delta_k (batch / "VarGrad" estimate, as in TBA and GFlowRL),
and the loss is the within-group variance of the residual
    loss = mean_k (delta_k - mean_j delta_j)^2.
Its gradient is 2 (delta_k - mean delta) * grad(log pi(y_k|x)) / norm_k: replies whose reward is
higher than the group's, relative to how much more likely the policy already makes them than the
reference, are pushed up. Two passes per step: a no-grad pass computes every delta (and the group
means); a gradient pass then uses the detached group mean, so micro-batches may split groups.
"""

from __future__ import annotations

import contextlib

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field

from gvg.rewards import compute_rewards


# ----------------------------------------------------------------------------------- config
@dataclass
class GRPOConfig:
    """Settings for one GRPO experiment.

    ``steps`` counts optimizer updates, ``group_size`` counts responses per
    prompt, and ``mix`` describes how many prompts of each kind are sampled.
    """
    steps: int = 30
    group_size: int = 8                      # K rollouts per prompt
    # prompts per step by pool; "harmful_prefill" = harmful prompts with a prefill, and when it is
    # present "harmful" means harmful prompts without one. Round 2 puts most of the batch on
    # prefilled prompts, where replies differ and GRPO gets a signal.
    mix: dict = field(default_factory=lambda: {"harmful_prefill": 8, "harmful": 2, "benign": 3, "math": 3})
    max_new_tokens: int = 512
    temperature: float = 1.0
    lr: float = 1e-4                         # round 1 used 1e-5: KL stayed ~0.001, no learning
    beta: float = 0.01                       # KL to the reference policy (round 1: 0.05)
    ref: str = "base"                        # "base" | "init" (see module docstring)
    adv_eps: float = 1e-4
    adv_min_std: float = 0.05                # groups with reward std below this carry no signal
    micro_batch: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    val_every: int = 10                      # greedy validation every N steps (0 = off)
    save_every: int = 10
    method: str = "grpo"                     # "grpo" | "tb" (trajectory balance / GFlowNet)
    tau: float = 0.1                         # TB reward temperature: target pi_ref * exp(r / tau)
    length_norm: bool = False                # TB: divide the log-ratio by reply length (rescales tau
                                             # by ~reply length; see module docstring)

    def __post_init__(self):
        if self.method not in ("grpo", "tb"):
            raise ValueError(f"method must be 'grpo' or 'tb', not {self.method!r}")
        if self.ref not in ("base", "init"):
            raise ValueError(f"ref must be 'base' or 'init', not {self.ref!r}")
        if self.tau <= 0:
            raise ValueError("tau must be positive")


@dataclass
class SFTConfig:
    samples_per_prompt: int = 4              # rejection sampling: K candidates per prompt
    temperature: float = 0.7
    max_new_tokens: int = 1024               # longer than GRPO's 512 so few targets are cut off
    keep_threshold: float = 0.5              # keep best candidate if its reward >= this
    keep_truncated_benign: bool = True       # a benign reply cut off at the cap is not a refusal
    epochs: int = 1
    lr: float = 1e-4
    batch_size: int = 16
    micro_batch: int = 4
    grad_clip: float = 1.0
    seed: int = 0


# ----------------------------------------------------------------------------------- helpers
def prompt_text(tokenizer, item: dict) -> str:
    """Chat template (Qwen3, thinking off) + optional prefill forced onto the assistant turn."""
    kw = {"enable_thinking": False}
    try:
        p = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}],
                                          tokenize=False, add_generation_prompt=True, **kw)
    except TypeError:  # tokenizers without the enable_thinking switch (tests)
        p = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}],
                                          tokenize=False, add_generation_prompt=True)
    return p + item.get("prefill", "")


def encode_prompt(tokenizer, item: dict) -> list[int]:
    return tokenizer(prompt_text(tokenizer, item), add_special_tokens=False)["input_ids"]


def group_advantages(rewards: list[float], group_size: int, eps: float = 1e-4,
                     min_std: float = 0.0) -> tuple[list[float], float]:
    """Normalize rewards within each prompt's response group.

    If every response receives the same reward, the group contains no preference
    signal.  The returned fraction records how often that happens.
    """
    assert len(rewards) % group_size == 0
    adv, zero = [], 0
    for g in range(0, len(rewards), group_size):
        r = rewards[g:g + group_size]
        m = sum(r) / len(r)
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / len(r))
        if sd < max(min_std, 1e-8):
            zero += 1
            adv += [0.0] * len(r)
        else:
            adv += [(x - m) / (sd + eps) for x in r]
    return adv, zero / (len(rewards) // group_size)


def _pad_batch(seqs: list[tuple[list[int], list[int]]], pad_id: int):
    """Convert variable-length examples into rectangular PyTorch tensors.

    The attention mask hides padding; the completion mask makes the loss ignore
    prompt tokens and score only tokens generated by the model.
    """
    import torch
    L = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    cm = torch.zeros((len(seqs), L), dtype=torch.float)
    for i, (p, c) in enumerate(seqs):
        n = len(p) + len(c)
        ids[i, :n] = torch.tensor(p + c)
        att[i, :n] = 1
        cm[i, len(p):n] = 1.0
    return ids, att, cm


def token_logprobs(model, ids, att):
    """Return the log probability assigned to each next token.

    Causal language models predict token ``t`` from tokens before ``t``, hence
    the one-position shift between logits and target token IDs.
    """
    import torch
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = ids[:, 1:]
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    return lp


def grpo_microbatch_loss(model, seqs, adv, beta, pad_id, n_total, ref_ctx=None):
    """Loss for one micro-batch, already scaled by len(seqs)/n_total (for accumulation).

    ref_ctx() returns a context manager under which the model computes reference logprobs
    (default: adapter disabled = base model). The reference pass runs BEFORE the policy pass so
    that switching adapters never touches tensors in the policy's autograd graph."""
    import torch
    dev = next(model.parameters()).device
    ids, att, cm = (t.to(dev) for t in _pad_batch(seqs, pad_id))
    mask = cm[:, 1:]
    ctx = ref_ctx() if ref_ctx is not None else model.disable_adapter()
    with torch.no_grad(), ctx:
        ref = token_logprobs(model, ids, att)
    lp = token_logprobs(model, ids, att)
    a = torch.tensor(adv, dtype=torch.float, device=dev).unsqueeze(1)
    pg = -a * torch.exp(lp - lp.detach())
    d = ref - lp
    kl = torch.exp(d) - d - 1.0                       # k3 estimator, >= 0
    tok = pg + beta * kl
    ntok = mask.sum(1).clamp(min=1.0)
    seq_loss = (tok * mask).sum(1) / ntok
    loss = seq_loss.sum() / n_total
    kl_mean = ((kl * mask).sum(1) / ntok).mean().item()
    return loss, kl_mean


def tb_seq_stats(model, seqs, pad_id, ref_ctx=None) -> dict:
    """No-grad pass for trajectory balance: per sequence, summed reference and policy logprobs
    over reply tokens, reply length, and the mean k3 KL (logged for comparison with GRPO)."""
    import torch
    dev = next(model.parameters()).device
    ids, att, cm = (t.to(dev) for t in _pad_batch(seqs, pad_id))
    mask = cm[:, 1:]
    ctx = ref_ctx() if ref_ctx is not None else model.disable_adapter()
    with torch.no_grad():
        with ctx:
            ref = token_logprobs(model, ids, att)
        lp = token_logprobs(model, ids, att)
    ntok = mask.sum(1).clamp(min=1.0)
    d = ref - lp
    k3 = ((torch.exp(d) - d - 1.0) * mask).sum(1) / ntok
    return {"ref_sum": (ref * mask).sum(1).tolist(), "lp_sum": (lp * mask).sum(1).tolist(),
            "ntok": ntok.tolist(), "kl": k3.tolist()}


def tb_deltas(lp_sum, ref_sum, ntok, rewards, tau, length_norm) -> list[float]:
    """TB residual per sample: length-normalised log pi/pi_ref minus r/tau."""
    return [(l - f) / (n if length_norm else 1.0) - r / tau
            for l, f, n, r in zip(lp_sum, ref_sum, ntok, rewards)]


def group_centers(values: list[float], group_size: int) -> list[float]:
    """Group mean of each value, broadcast back to every member (= -log Z_hat for TB)."""
    out = []
    for g in range(0, len(values), group_size):
        v = values[g:g + group_size]
        out += [sum(v) / len(v)] * len(v)
    return out


def tb_microbatch_loss(model, seqs, ref_sum, ntok, rewards, centers, tau, length_norm, pad_id,
                       n_total):
    """Gradient pass for trajectory balance, scaled by len(seqs)/n_total for accumulation.
    centers are the detached group means of delta from the no-grad pass."""
    import torch
    dev = next(model.parameters()).device
    ids, att, cm = (t.to(dev) for t in _pad_batch(seqs, pad_id))
    mask = cm[:, 1:]
    lp = token_logprobs(model, ids, att)
    lp_sum = (lp * mask).sum(1)
    t = lambda xs: torch.tensor(xs, dtype=torch.float, device=dev)
    norm = t(ntok) if length_norm else torch.ones(len(seqs), device=dev)
    delta = (lp_sum - t(ref_sum)) / norm - t(rewards) / tau
    resid = delta - t(centers)
    return (resid ** 2).sum() / n_total


def sft_microbatch_loss(model, seqs, pad_id, n_total):
    """Mean NLL over completion tokens per sequence, scaled for accumulation."""
    dev = next(model.parameters()).device
    ids, att, cm = (t.to(dev) for t in _pad_batch(seqs, pad_id))
    mask = cm[:, 1:]
    lp = token_logprobs(model, ids, att)
    seq_nll = -(lp * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return seq_nll.sum() / n_total


def pool_key(item: dict, split_prefill: bool) -> str:
    """Sampling pool of an item; prefilled harmful prompts get their own pool if requested."""
    if split_prefill and item["kind"] == "harmful" and item.get("prefill"):
        return "harmful_prefill"
    return item["kind"]


class MixSampler:
    """Cycles through each pool's shuffled prompts; every step draws mix[pool] of each."""

    def __init__(self, items: list[dict], mix: dict, seed: int):
        self.rng = random.Random(seed)
        split = "harmful_prefill" in mix
        self.pools = {k: [x for x in items if pool_key(x, split) == k] for k in mix}
        for k, v in self.pools.items():
            if not v:
                raise ValueError(f"no training items of kind {k!r}")
        self.mix, self.queues = mix, {k: [] for k in mix}

    def draw(self) -> list[dict]:
        out = []
        for k, n in self.mix.items():
            for _ in range(n):
                if not self.queues[k]:
                    self.queues[k] = self.pools[k][:]
                    self.rng.shuffle(self.queues[k])
                out.append(self.queues[k].pop())
        return out


def _decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True)


def _summarise(items, rs) -> dict:
    s = {}
    for kind in ("harmful", "benign", "math"):
        sub = [(x, r) for x, r in zip(items, rs) if x["kind"] == kind]
        if not sub:
            continue
        s[f"reward_{kind}"] = sum(r["reward"] for _, r in sub) / len(sub)
        if kind == "harmful":
            for tag, pre in (("prefill", True), ("plain", False)):
                ss = [r for x, r in sub if bool(x.get("prefill")) == pre]
                if ss:
                    s[f"harmful_rate_{tag}"] = sum((r["p_harmful"] or 0) > 0.5 for r in ss) / len(ss)
        if kind == "benign":
            s["refusal_rate_benign"] = sum((r["p_refusal"] or 0) > 0.5 for _, r in sub) / len(sub)
        if kind == "math":
            s["math_acc"] = sum(r["correct"] for _, r in sub) / len(sub)
    return s


# ----------------------------------------------------------------------------------- GRPO
def evaluate(tokenizer, items, generate_fn, judge_fn, step, max_new_tokens) -> dict:
    """Greedy validation: one reply per item."""
    pids = [encode_prompt(tokenizer, x) for x in items]
    comps = generate_fn(pids, 1, step, 0.0, max_new_tokens)
    replies = [x.get("prefill", "") + _decode(tokenizer, c[0]) for x, c in zip(items, comps)]
    rs = compute_rewards(items, replies, judge_fn)
    return {f"val_{k}": v for k, v in _summarise(items, rs).items()}


def adapter_ref_ctx(model, ref_name: str = "ref", active: str = "default"):
    """Context manager factory: compute reference logprobs with a frozen copy of the starting
    adapter (loaded under ref_name), then switch back to the trainable adapter."""
    @contextlib.contextmanager
    def ctx():
        model.set_adapter(ref_name)
        try:
            yield
        finally:
            model.set_adapter(active)
            for n, p in model.named_parameters():   # set_adapter toggles requires_grad
                if f".{ref_name}." in n:
                    p.requires_grad_(False)
                elif f".{active}." in n:
                    p.requires_grad_(True)
    return ctx


def train_grpo(model, tokenizer, train_items, val_items, cfg: GRPOConfig, generate_fn, judge_fn,
               log_fn=print, save_fn=None, ref_ctx=None) -> list[dict]:
    # Callbacks keep the algorithm independent of its runtime: Modal supplies
    # vLLM and WildGuard, while tests supply tiny local stand-ins.
    import torch
    torch.manual_seed(cfg.seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    sampler = MixSampler(train_items, cfg.mix, cfg.seed)
    logs = []
    if val_items and cfg.val_every:
        logs.append({"step": 0, **evaluate(tokenizer, val_items, generate_fn, judge_fn, 0, cfg.max_new_tokens)})
        log_fn(json.dumps(logs[-1]))
    for step in range(1, cfg.steps + 1):
        t0 = time.time()
        batch = sampler.draw()
        pids = [encode_prompt(tokenizer, x) for x in batch]
        comps = generate_fn(pids, cfg.group_size, step, cfg.temperature, cfg.max_new_tokens)
        t_gen = time.time() - t0
        items = [x for x in batch for _ in range(cfg.group_size)]
        seqs = [(p, c) for p, cs in zip(pids, comps) for c in cs]
        replies = [x.get("prefill", "") + _decode(tokenizer, c) for x, (_, c) in zip(items, seqs)]
        rs = compute_rewards(items, replies, judge_fn)
        adv, zero_frac = group_advantages([r["reward"] for r in rs], cfg.group_size, cfg.adv_eps,
                                          cfg.adv_min_std)
        t_judge = time.time() - t0 - t_gen

        model.train()
        opt.zero_grad(set_to_none=True)
        kls, extra = [], {}
        mb = cfg.micro_batch
        if cfg.method == "grpo":
            for i in range(0, len(seqs), mb):
                loss, kl = grpo_microbatch_loss(model, seqs[i:i + mb], adv[i:i + mb], cfg.beta,
                                                pad_id, len(seqs), ref_ctx)
                loss.backward()
                kls.append(kl)
        else:  # trajectory balance
            st = {"ref_sum": [], "lp_sum": [], "ntok": [], "kl": []}
            for i in range(0, len(seqs), mb):
                part = tb_seq_stats(model, seqs[i:i + mb], pad_id, ref_ctx)
                for k in st:
                    st[k] += part[k]
            rewards = [r["reward"] for r in rs]
            deltas = tb_deltas(st["lp_sum"], st["ref_sum"], st["ntok"], rewards, cfg.tau,
                               cfg.length_norm)
            centers = group_centers(deltas, cfg.group_size)
            tb_loss = 0.0
            for i in range(0, len(seqs), mb):
                loss = tb_microbatch_loss(model, seqs[i:i + mb], st["ref_sum"][i:i + mb],
                                          st["ntok"][i:i + mb], rewards[i:i + mb],
                                          centers[i:i + mb], cfg.tau, cfg.length_norm, pad_id,
                                          len(seqs))
                loss.backward()
                tb_loss += loss.item()
            kls = [sum(st["kl"]) / len(st["kl"])]
            lr_tok = [(l - f) / n for l, f, n in zip(st["lp_sum"], st["ref_sum"], st["ntok"])]
            extra = {"tb_loss": tb_loss,
                     "logZ_mean": -sum(centers[::cfg.group_size]) / (len(centers) // cfg.group_size),
                     "logratio_per_token": sum(lr_tok) / len(lr_tok)}
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        opt.step()

        rec = {"step": step, "reward_mean": sum(r["reward"] for r in rs) / len(rs),
               **_summarise(items, rs), "zero_var_group_frac": zero_frac,
               "kl": sum(kls) / len(kls), "grad_norm": gnorm, **extra,
               "resp_len": sum(len(c) for _, c in seqs) / len(seqs),
               "truncated_frac": sum(len(c) >= cfg.max_new_tokens for _, c in seqs) / len(seqs),
               "t_gen": round(t_gen, 1), "t_judge": round(t_judge, 1),
               "t_step": round(time.time() - t0, 1)}
        if val_items and cfg.val_every and step % cfg.val_every == 0:
            rec.update(evaluate(tokenizer, val_items, generate_fn, judge_fn, step, cfg.max_new_tokens))
        logs.append(rec)
        log_fn(json.dumps(rec))
        if save_fn and (step % cfg.save_every == 0 or step == cfg.steps):
            save_fn(step)
    return logs


# ----------------------------------------------------------------------------------- SFT
def build_sft_set(tokenizer, items, cfg: SFTConfig, generate_fn, judge_fn, log_fn=print):
    """Rejection sampling from the current model: per prompt, keep the highest-reward candidate
    if its reward >= keep_threshold (safe on harmful, not a refusal on benign, correct on math).
    For prefilled prompts the kept target is the continuation after the prefill, so SFT teaches
    recovery from a compliant opening using the model's own successful recoveries."""
    pids = [encode_prompt(tokenizer, x) for x in items]
    comps = generate_fn(pids, cfg.samples_per_prompt, 0, cfg.temperature, cfg.max_new_tokens)
    flat_items = [x for x in items for _ in range(cfg.samples_per_prompt)]
    flat = [c for cs in comps for c in cs]
    replies = [x.get("prefill", "") + _decode(tokenizer, c) for x, c in zip(flat_items, flat)]
    rs = compute_rewards(flat_items, replies, judge_fn)
    keep, stats, why, trunc_kept = [], {}, {}, 0
    K = cfg.samples_per_prompt
    for j, x in enumerate(items):
        tag = x["kind"] + ("_prefill" if x.get("prefill") else "")
        allow_trunc = cfg.keep_truncated_benign and x["kind"] == "benign"
        # best candidate among those passing the length rule
        cand = [(rs[j * K + k]["reward"], k) for k in range(K)
                if allow_trunc or len(flat[j * K + k]) < cfg.max_new_tokens]
        st = stats.setdefault(tag, [0, 0])
        st[1] += 1
        w = why.setdefault(tag, {"all_truncated": 0, "low_reward": 0})
        if not cand:
            w["all_truncated"] += 1
            continue
        best_r, best_k = max(cand)
        if best_r < cfg.keep_threshold:
            w["low_reward"] += 1
            continue
        st[0] += 1
        trunc_kept += len(flat[j * K + best_k]) >= cfg.max_new_tokens
        keep.append((pids[j], flat[j * K + best_k]))
    log_fn(json.dumps({"sft_kept": {k: f"{a}/{b}" for k, (a, b) in stats.items()},
                       "sft_dropped": why, "sft_examples": len(keep),
                       "sft_kept_truncated": trunc_kept}))
    return keep, stats


def train_sft(model, tokenizer, examples, cfg: SFTConfig, log_fn=print) -> list[dict]:
    import torch
    torch.manual_seed(cfg.seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    rng = random.Random(cfg.seed)
    logs, step = [], 0
    for ep in range(cfg.epochs):
        order = examples[:]
        rng.shuffle(order)
        for b in range(0, len(order), cfg.batch_size):
            batch = order[b:b + cfg.batch_size]
            model.train()
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for i in range(0, len(batch), cfg.micro_batch):
                loss = sft_microbatch_loss(model, batch[i:i + cfg.micro_batch], pad_id, len(batch))
                loss.backward()
                tot += loss.item()
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
            opt.step()
            step += 1
            rec = {"sft_step": step, "epoch": ep, "loss": tot, "grad_norm": gnorm}
            logs.append(rec)
            log_fn(json.dumps(rec))
    return logs


def config_dict(cfg) -> dict:
    return asdict(cfg)
