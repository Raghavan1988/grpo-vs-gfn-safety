"""GRPO and rejection-sampling SFT for LoRA fine-tuning, independent of where they run.

Generation and judging are passed in as functions, so the same loop runs on Modal (vLLM +
WildGuard) and in CPU tests (tiny model + fake judge).

    generate_fn(prompt_ids: list[list[int]], n: int, step: int, temperature: float,
                max_new_tokens: int) -> list[list[list[int]]]     # [prompt][sample] -> completion ids
    judge_fn(pairs: list[tuple[str, str]]) -> list[dict]          # gvg.wildguard.score format

GRPO objective (one policy update per batch, so the PPO ratio is 1 and clipping is inactive):
    loss = mean_seq[ mean_tok( -A * exp(logp - logp.detach()) + beta * KL_k3(pi || pi_ref) ) ]
A = group-normalised reward (r - mean_group) / (std_group + eps). The reference policy is the
base model (LoRA adapter disabled) for both "pure GRPO" and "SFT + GRPO", so both are regularised
toward the same untuned model.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field

from gvg.rewards import compute_rewards


# ----------------------------------------------------------------------------------- config
@dataclass
class GRPOConfig:
    steps: int = 30
    group_size: int = 8                      # K rollouts per prompt
    # prompts per step by pool; "harmful_prefill" = harmful prompts with a prefill, and when it is
    # present "harmful" means harmful prompts without one. Round 2 puts most of the batch on
    # prefilled prompts, where replies differ and GRPO gets a signal.
    mix: dict = field(default_factory=lambda: {"harmful_prefill": 8, "harmful": 2, "benign": 3, "math": 3})
    max_new_tokens: int = 512
    temperature: float = 1.0
    lr: float = 1e-4                         # round 1 used 1e-5: KL stayed ~0.001, no learning
    beta: float = 0.01                       # KL to the base model (round 1: 0.05)
    adv_eps: float = 1e-4
    micro_batch: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    val_every: int = 10                      # greedy validation every N steps (0 = off)
    save_every: int = 10


@dataclass
class SFTConfig:
    samples_per_prompt: int = 4              # rejection sampling: K candidates per prompt
    temperature: float = 0.7
    max_new_tokens: int = 512
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


def group_advantages(rewards: list[float], group_size: int, eps: float = 1e-4) -> tuple[list[float], float]:
    """(r - group mean) / (group std + eps); also returns the fraction of zero-variance groups."""
    assert len(rewards) % group_size == 0
    adv, zero = [], 0
    for g in range(0, len(rewards), group_size):
        r = rewards[g:g + group_size]
        m = sum(r) / len(r)
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / len(r))
        zero += sd < 1e-8
        adv += [(x - m) / (sd + eps) for x in r]
    return adv, zero / (len(rewards) // group_size)


def _pad_batch(seqs: list[tuple[list[int], list[int]]], pad_id: int):
    """[(prompt_ids, completion_ids)] -> input_ids, attention_mask, completion_mask (torch)."""
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
    """log pi(token_t | tokens_<t) for t = 1..L-1, shape (B, L-1), float32."""
    import torch
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = ids[:, 1:]
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    return lp


def grpo_microbatch_loss(model, seqs, adv, beta, pad_id, n_total):
    """Loss for one micro-batch, already scaled by len(seqs)/n_total (for accumulation)."""
    import torch
    dev = next(model.parameters()).device
    ids, att, cm = (t.to(dev) for t in _pad_batch(seqs, pad_id))
    mask = cm[:, 1:]
    lp = token_logprobs(model, ids, att)
    with torch.no_grad(), model.disable_adapter():
        ref = token_logprobs(model, ids, att)
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


def train_grpo(model, tokenizer, train_items, val_items, cfg: GRPOConfig, generate_fn, judge_fn,
               log_fn=print, save_fn=None) -> list[dict]:
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
        adv, zero_frac = group_advantages([r["reward"] for r in rs], cfg.group_size, cfg.adv_eps)
        t_judge = time.time() - t0 - t_gen

        model.train()
        opt.zero_grad(set_to_none=True)
        kls = []
        for i in range(0, len(seqs), cfg.micro_batch):
            loss, kl = grpo_microbatch_loss(model, seqs[i:i + cfg.micro_batch],
                                            adv[i:i + cfg.micro_batch], cfg.beta, pad_id, len(seqs))
            loss.backward()
            kls.append(kl)
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        opt.step()

        rec = {"step": step, "reward_mean": sum(r["reward"] for r in rs) / len(rs),
               **_summarise(items, rs), "zero_var_group_frac": zero_frac,
               "kl": sum(kls) / len(kls), "grad_norm": gnorm,
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
    keep, stats, why = [], {}, {}
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
        keep.append((pids[j], flat[j * K + best_k]))
    log_fn(json.dumps({"sft_kept": {k: f"{a}/{b}" for k, (a, b) in stats.items()},
                       "sft_dropped": why, "sft_examples": len(keep)}))
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
