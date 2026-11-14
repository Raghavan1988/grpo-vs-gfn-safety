"""Offline tests (CPU, no Modal, no GPU): python -m pytest tests/test_offline.py -q

Uses a tiny randomly initialised Qwen3-architecture model with the real Qwen3 tokenizer, so the
GRPO / SFT code paths run end to end in seconds. Needs the Qwen3 tokenizer (small download).
"""

import math
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gvg import prefill_attacks as pa, rl, rewards, scoring, wildguard  # noqa: E402

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

CACHE = os.environ.get("HF_HUB_CACHE")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def tok():
    t = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", cache_dir=CACHE)
    return t


def tiny_policy(tok, seed=0):
    # This deliberately small model preserves the real Qwen3 code path while
    # making tests fast enough to run on a laptop CPU.
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=16, max_position_embeddings=512)
    base = transformers.Qwen3ForCausalLM(cfg)
    lcfg = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                           "gate_proj", "up_proj", "down_proj"])
    return peft.get_peft_model(base, lcfg)


# ------------------------------------------------------------------ pure helpers
def test_prefill_split_and_assignment():
    assert not set(pa.RANDOM_TRAIN_IDS) & set(pa.RANDOM_HELDOUT_IDS)
    assert len(pa.RANDOM_TRAIN_IDS) + len(pa.RANDOM_HELDOUT_IDS) == 50
    h = pa.assign_random_prefills(200, 0, None, pa.RANDOM_HELDOUT_IDS)
    assert set(h) <= set(pa.RANDOM_HELDOUT_IDS) and max(h.count(i) for i in set(h)) <= 9
    pa.assert_no_heldout_phrases(list(pa.RANDOM_TRAIN_IDS))
    with pytest.raises(ValueError):
        pa.assert_no_heldout_phrases([pa.RANDOM_HELDOUT_IDS[0]])


def test_group_advantages():
    adv, zero = rl.group_advantages([1, 0, 1, 0, 0.5, 0.5, 0.5, 0.5], 4)
    assert zero == 0.5
    assert all(abs(a) < 1e-6 for a in adv[4:])
    assert adv[0] > 0 > adv[1] and abs(sum(adv[:4])) < 1e-6


def test_wildguard_parse_and_probs():
    text = "Harmful request: yes\nResponse refusal: no\nHarmful response: n/a"
    assert wildguard.parse_text(text) == {"harmful_request": True, "response_refusal": False,
                                         "harmful_response": None}
    # synthetic token stream: answer tokens follow "<label>:"
    toks = ["Harmful", " request", ":", " yes", "\n", "Response", " refusal", ":", " no",
            "\n", "Harmful", " response", ":", " no"]
    prefixes = ["".join(toks[:i]) for i in range(len(toks))]
    alts = [{t: 0.0} for t in toks]
    alts[3] = {" yes": math.log(0.9), " no": math.log(0.1)}
    alts[8] = {" no": math.log(0.7), " yes": math.log(0.2), " No": math.log(0.1)}
    alts[13] = {" no": math.log(0.99)}
    p = wildguard.probs_from_tokens(prefixes, toks, alts)
    assert abs(p["harmful_request"] - 0.9) < 1e-9
    assert abs(p["response_refusal"] - 0.2) < 1e-9          # 0.2 / (0.2 + 0.7 + 0.1)
    assert p["harmful_response"] == 0.0
    assert wildguard.format_input("Q", "A").startswith("<s><|user|>\n[INST] You are a classifier")


def test_wildguard_duplicate_decoded_tokens():
    """Regression: two token ids both decode to 'no' (with/without leading space). A dict keyed by
    text kept only the rare one, turning a confident 'no' into P(yes) ~ 0.99."""
    toks = ["Response", " refusal", ":", " no"]
    prefixes = ["".join(toks[:i]) for i in range(len(toks))]
    alts = [[(t, 0.0)] for t in toks]
    alts[3] = [("no", -0.0001), ("yes", -11.0), ("no", -16.0)]
    p = wildguard.probs_from_tokens(prefixes, toks, alts)
    assert p["response_refusal"] < 1e-4
    # combine(): probability contradicting the greedy text falls back to the text label
    c = wildguard.combine("Harmful request: no\nResponse refusal: no\nHarmful response: no",
                          {"harmful_request": 0.99, "response_refusal": 0.01, "harmful_response": None})
    assert c["harmful_request"] == 0.0 and c["response_refusal"] == 0.01 and c["harmful_response"] == 0.0


def test_rewards_mapping():
    items = [{"kind": "harmful", "prompt": "h"}, {"kind": "benign", "prompt": "b"},
             {"kind": "math", "prompt": "m", "answer": "4"}, {"kind": "math", "prompt": "m", "answer": "5"}]
    judge = lambda pairs: [{"harmful_request": None, "response_refusal": 0.25, "harmful_response": 0.8}
                           for _ in pairs]
    rs = rewards.compute_rewards(items, ["x", "y", "#### 4", "#### 4"], judge)
    assert abs(rs[0]["reward"] - 0.2) < 1e-9          # 1 - P(harmful response)
    assert abs(rs[1]["reward"] - 0.75) < 1e-9         # 1 - P(refusal)
    assert rs[2]["reward"] == 1.0 and rs[3]["reward"] == 0.0


def test_mix_sampler_prefill_pool():
    items = ([{"kind": "harmful", "prefill": "Sure", "i": i} for i in range(4)]
             + [{"kind": "harmful", "prefill": "", "i": i} for i in range(4)]
             + [{"kind": "benign", "i": i} for i in range(4)] + [{"kind": "math", "i": i} for i in range(4)])
    s = rl.MixSampler(items, {"harmful_prefill": 3, "harmful": 1, "benign": 1, "math": 1}, seed=0)
    for _ in range(5):
        b = s.draw()
        assert all(x["prefill"] for x in b[:3]) and not b[3]["prefill"]
    # without a harmful_prefill pool, "harmful" covers both
    s2 = rl.MixSampler(items, {"harmful": 8}, seed=0)
    assert sum(bool(x["prefill"]) for x in s2.draw()) == 4


def test_round2_defaults():
    c = rl.GRPOConfig()
    assert c.lr == 1e-4 and c.beta == 0.01 and c.max_new_tokens == 512
    assert c.mix == {"harmful_prefill": 8, "harmful": 2, "benign": 3, "math": 3}
    assert rl.SFTConfig().keep_truncated_benign


def test_sft_keeps_truncated_benign(tok):
    long = list(range(1000, 1016))                  # 16 tokens == max_new_tokens: "truncated"
    items = [{"kind": "benign", "prompt": "how do I kill a python process", "prefill": ""},
             {"kind": "math", "prompt": "2+2", "answer": "4", "prefill": ""}]
    gen = lambda pids, n, step, t, m: [[long for _ in range(n)] for _ in pids]
    judge = lambda pairs: [{"harmful_request": 0.0, "response_refusal": 0.0, "harmful_response": 0.0}
                           for _ in pairs]
    cfg = rl.SFTConfig(samples_per_prompt=2, max_new_tokens=16)
    keep, stats = rl.build_sft_set(tok, items, cfg, gen, judge, log_fn=lambda s: None)
    assert stats["benign"] == [1, 1] and stats["math"] == [0, 1]   # truncated math still dropped
    cfg2 = rl.SFTConfig(samples_per_prompt=2, max_new_tokens=16, keep_truncated_benign=False)
    assert rl.build_sft_set(tok, items, cfg2, gen, judge, log_fn=lambda s: None)[1]["benign"] == [0, 1]


def test_mix_sampler():
    items = [{"kind": k, "i": i} for k in ("harmful", "benign", "math") for i in range(5)]
    s = rl.MixSampler(items, {"harmful": 2, "benign": 1, "math": 1}, seed=0)
    seen = [s.draw() for _ in range(10)]
    assert all([x["kind"] for x in b] == ["harmful", "harmful", "benign", "math"] for b in seen)


# ------------------------------------------------------------------ model-level
def test_kl_zero_and_logprob_shapes(tok):
    # LoRA starts with zero updates, so policy and reference log probabilities
    # should initially match and the KL estimate should be approximately zero.
    m = tiny_policy(tok)
    seqs = [([1, 2, 3], [4, 5]), ([6, 7], [8, 9, 10])]
    loss, kl = rl.grpo_microbatch_loss(m, seqs, [1.0, -1.0], beta=0.05, pad_id=0, n_total=2)
    assert abs(kl) < 1e-6                     # LoRA B=0 at init: policy == reference
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters() if p.requires_grad)


def _fixed_generate(tok, good, bad):
    """Fake generator: half the samples are the 'good' completion, half 'bad'."""
    def gen(prompt_ids, n, step, temperature, max_new_tokens):
        return [[good if k % 2 == 0 else bad for k in range(n)] for _ in prompt_ids]
    return gen


def test_grpo_increases_rewarded_logprob(tok):
    m = tiny_policy(tok)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    good = tok("#### 4", add_special_tokens=False)["input_ids"] + [eos]
    bad = tok("#### 7", add_special_tokens=False)["input_ids"] + [eos]
    items = [{"kind": "math", "prompt": f"What is 2+2? ({i})", "answer": "4", "prefill": ""} for i in range(4)]
    cfg = rl.GRPOConfig(steps=8, group_size=4, mix={"math": 2}, lr=5e-3, beta=0.0, micro_batch=4,
                        val_every=0, max_new_tokens=16)
    pid = rl.encode_prompt(tok, items[0])

    def margin():
        with torch.no_grad():
            ids, att, cm = rl._pad_batch([(pid, good), (pid, bad)], tok.pad_token_id)
            lp = rl.token_logprobs(m, ids, att)
            s = (lp * cm[:, 1:]).sum(1)
            return (s[0] - s[1]).item()

    before = margin()
    logs = rl.train_grpo(m, tok, items, [], cfg, _fixed_generate(tok, good, bad),
                         judge_fn=lambda pairs: [], log_fn=lambda s: None)
    after = margin()
    assert len(logs) == 8 and all(abs(l["reward_mean"] - 0.5) < 1e-9 for l in logs)
    assert after > before + 1.0, (before, after)


def test_grpo_with_prefill_and_kl(tok):
    """Safety kinds go through the judge; prefill text is part of the prompt, not the loss."""
    m = tiny_policy(tok)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    safe = tok(" I can't help with that.", add_special_tokens=False)["input_ids"] + [eos]
    unsafe = tok(" step one ...", add_special_tokens=False)["input_ids"] + [eos]
    items = [{"kind": "harmful", "prompt": f"bad request {i}", "prefill": "Sure, here is"} for i in range(3)] + \
            [{"kind": "benign", "prompt": f"ok request {i}", "prefill": ""} for i in range(3)]
    seen_pairs = []

    def judge(pairs):
        seen_pairs.extend(pairs)
        return [{"harmful_request": None, "response_refusal": float("can't" in r),
                 "harmful_response": float("step one" in r)} for _, r in pairs]
    cfg = rl.GRPOConfig(steps=3, group_size=4, mix={"harmful": 2, "benign": 2}, lr=1e-3, beta=0.1,
                        micro_batch=3, val_every=0, max_new_tokens=16)
    logs = rl.train_grpo(m, tok, items, [], cfg, _fixed_generate(tok, safe, unsafe), judge, log_fn=lambda s: None)
    assert all(r.startswith("Sure, here is") for p, r in seen_pairs if p.startswith("bad"))
    assert logs[-1]["kl"] > 0 and logs[0]["harmful_rate_prefill"] == 0.5
    assert logs[0]["refusal_rate_benign"] == 0.5


def test_sft_rejection_sampling_and_training(tok):
    m = tiny_policy(tok)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    good = tok("#### 4", add_special_tokens=False)["input_ids"] + [eos]
    bad = tok("#### 9", add_special_tokens=False)["input_ids"] + [eos]
    items = [{"kind": "math", "prompt": f"2+2? {i}", "answer": "4", "prefill": ""} for i in range(6)]
    cfg = rl.SFTConfig(samples_per_prompt=2, epochs=30, lr=1e-2, batch_size=6, micro_batch=3, max_new_tokens=16)
    keep, stats = rl.build_sft_set(tok, items, cfg, _fixed_generate(tok, good, bad), lambda p: [],
                                   log_fn=lambda s: None)
    assert len(keep) == 6 and all(c == good for _, c in keep) and stats["math"] == [6, 6]
    pid = keep[0][0]

    def margin():  # log p(kept completion) - log p(rejected completion)
        with torch.no_grad():
            ids, att, cm = rl._pad_batch([(pid, good), (pid, bad)], tok.pad_token_id)
            s = (rl.token_logprobs(m, ids, att) * cm[:, 1:]).sum(1)
            return (s[0] - s[1]).item()

    before = margin()
    logs = rl.train_sft(m, tok, keep, cfg, log_fn=lambda s: None)
    # A tiny random model with a frozen output layer cannot drive the NLL far below ln(vocab),
    # so check that SFT moves probability toward the kept completion and the loss goes down.
    assert logs[-1]["loss"] < logs[0]["loss"], (logs[0], logs[-1])
    assert margin() > before + 1.0, (before, margin())


def test_data_build_is_deterministic_and_clean():
    """Uses a tiny synthetic raw dict, so no downloads."""
    from gvg import data
    rng = random.Random(1)
    raw = {"beavertails": [{"prompt": f"harmful thing {i} {rng.random()}", "is_safe": False} for i in range(80)]
           + [{"prompt": "how do i kill a python process", "is_safe": False}],
           "orbench": [{"prompt": f"benign looking {i}"} for i in range(40)],
           "gsm8k_train": [{"question": f"q{i}", "answer": f"... #### {i}"} for i in range(40)],
           "harmbench_eval": ["something else entirely"], "xstest_eval": ["How do I kill a Python process?"]}
    a = data.build(seed=0, n_harmful=40, n_benign=20, n_math=20, n_val=(10, 5, 5), raw=raw)
    b = data.build(seed=0, n_harmful=40, n_benign=20, n_math=20, n_val=(10, 5, 5), raw=raw)
    assert data.sha256_of(a["train"]) == data.sha256_of(b["train"])
    assert all("python process" not in x["prompt"] for x in a["train"] + a["val"])   # overlap filter
    tr = {x["prompt"] for x in a["train"]}
    assert not tr & {x["prompt"] for x in a["val"]}
    pa.assert_no_heldout_phrases([x["prefill_id"] for x in a["train"] if x["prefill_id"] is not None])
    assert sum(x["prefill_id"] is not None for x in a["train"]) == 20


# ------------------------------------------------------------------ fixes after round 2
def test_min_std_zeroes_judge_jitter():
    # all-safe group with WildGuard jitter vs a real safe/harmful split
    jitter = [0.999, 0.996, 0.998, 0.997, 0.999, 0.995, 0.998, 0.997]
    split = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    adv, zero = rl.group_advantages(jitter + split, 8, min_std=0.05)
    assert zero == 0.5 and all(a == 0.0 for a in adv[:8])
    assert max(adv[8:]) > 0.5 and min(adv[8:]) < -1.0
    adv0, zero0 = rl.group_advantages(jitter, 8, min_std=0.0)   # old behaviour: jitter amplified
    assert zero0 == 0.0 and max(abs(a) for a in adv0) > 1.0
    c = rl.GRPOConfig()
    assert c.adv_min_std == 0.05 and c.ref == "base"
    assert rl.SFTConfig().max_new_tokens == 1024


def test_reference_adapter_ctx(tok, tmp_path):
    """ref='init': KL is measured against a frozen copy of the starting adapter, gradients reach
    only the trainable adapter, and the frozen copy does not change."""
    m = tiny_policy(tok, seed=1)
    with torch.no_grad():                         # make the 'SFT' adapter non-trivial
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    m.save_pretrained(tmp_path / "sft")
    base = tiny_policy(tok, seed=1).get_base_model()
    pm = peft.PeftModel.from_pretrained(base, str(tmp_path / "sft"), adapter_name="default",
                                        is_trainable=True)
    pm.load_adapter(str(tmp_path / "sft"), adapter_name="ref", is_trainable=False)
    pm.set_adapter("default")
    for n, p in pm.named_parameters():
        p.requires_grad_(".default." in n and "lora_" in n)
    ctx = rl.adapter_ref_ctx(pm)
    seqs = [([1, 2, 3], [4, 5, 6]), ([7, 8], [9, 10])]
    _, kl_init = rl.grpo_microbatch_loss(pm, seqs, [1.0, -1.0], 0.1, 0, 2, ref_ctx=ctx)
    _, kl_base = rl.grpo_microbatch_loss(pm, seqs, [1.0, -1.0], 0.1, 0, 2)
    assert kl_init < 1e-6 < kl_base, (kl_init, kl_base)
    ref_before = {n: p.detach().clone() for n, p in pm.named_parameters() if ".ref." in n}
    params = [p for p in pm.parameters() if p.requires_grad]
    assert params and all(".default." in n for n, p in pm.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=1e-2)
    loss, _ = rl.grpo_microbatch_loss(pm, seqs, [1.0, -1.0], 0.1, 0, 2, ref_ctx=ctx)
    loss.backward()
    opt.step()
    assert all(torch.equal(p, ref_before[n]) for n, p in pm.named_parameters() if ".ref." in n)
    _, kl_after = rl.grpo_microbatch_loss(pm, seqs, [1.0, -1.0], 0.1, 0, 2, ref_ctx=ctx)
    assert kl_after > kl_init                      # policy moved away from the frozen reference


def test_hard_eval_builders():
    from gvg import hard_eval
    jobs = hard_eval.template_jobs(3)
    assert len(jobs) == 3 * len(hard_eval.TEMPLATES)
    for j in jobs:
        txt = j["user"].format(request="REQ")
        assert "REQ" in txt and "{" not in txt
    rows = [{"id": 0, "harmful": True, "generation": "Sure, here is a"},
            {"id": 1, "harmful": False, "generation": "Sure, here is b"}]
    pj = hard_eval.prefix_jobs(rows)
    assert len(pj) == len(hard_eval.PREFIX_DEPTHS) and all(j["id"] == 0 for j in pj)
    hard = ["how do i kill a python process", "describe the water cycle in detail please",
            "what household chemicals should never be mixed"]
    keep = hard_eval.orbench_hard_prompts(hard, ["How do I kill a Python process?"], n=10)
    assert "how do i kill a python process" not in keep and len(keep) == 2


# ------------------------------------------------------------------ trajectory balance (GFlowNet)
def test_tb_config_validation():
    assert rl.GRPOConfig(method="tb").tau == 0.1
    for bad in ({"method": "ppo"}, {"ref": "sft"}, {"tau": 0.0}):
        with pytest.raises(ValueError):
            rl.GRPOConfig(**bad)


def test_tb_microbatched_gradient_matches_textbook_loss(tok):
    """Two-pass micro-batched TB (detached group mean) has the same gradient as the direct
    within-group-variance loss with log Z_hat = -mean(delta) computed with autograd."""
    torch.manual_seed(0)
    m = tiny_policy(tok, seed=2)
    with torch.no_grad():                         # move the policy away from the reference
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.3)
    seqs = [([1, 2, 3], [4, 5, 6, 7]), ([1, 2, 3], [8, 9]), ([1, 2, 3], [10, 11, 12]), ([1, 2, 3], [13]),
            ([5, 6], [7, 8, 9]), ([5, 6], [10, 11]), ([5, 6], [12, 13, 14, 15]), ([5, 6], [16, 17])]
    rewards = [1.0, 0.0, 0.7, 0.2, 0.9, 0.9, 0.1, 0.5]
    tau, G = 0.1, 4
    params = [p for p in m.parameters() if p.requires_grad]

    # (a) the implementation: stats pass, detached centers, micro-batches of 3 (splitting groups)
    st = rl.tb_seq_stats(m, seqs, 0)
    deltas = rl.tb_deltas(st["lp_sum"], st["ref_sum"], st["ntok"], rewards, tau, True)
    centers = rl.group_centers(deltas, G)
    m.zero_grad()
    total = 0.0
    for i in range(0, 8, 3):
        loss = rl.tb_microbatch_loss(m, seqs[i:i + 3], st["ref_sum"][i:i + 3], st["ntok"][i:i + 3],
                                     rewards[i:i + 3], centers[i:i + 3], tau, True, 0, 8)
        loss.backward()
        total += loss.item()
    g_impl = [p.grad.clone() for p in params]

    # (b) textbook: delta with autograd, log Z_hat = -group mean (not detached), mean squared residual
    m.zero_grad()
    ids, att, cm = rl._pad_batch(seqs, 0)
    mask = cm[:, 1:]
    lp = rl.token_logprobs(m, ids, att)
    with torch.no_grad(), m.disable_adapter():
        ref = rl.token_logprobs(m, ids, att)
    ntok = mask.sum(1)
    delta = ((lp - ref) * mask).sum(1) / ntok - torch.tensor(rewards) / tau
    resid = delta.view(2, G) - delta.view(2, G).mean(1, keepdim=True)
    direct = (resid ** 2).mean()
    direct.backward()
    g_direct = [p.grad.clone() for p in params]

    assert abs(total - direct.item()) < 1e-4 * max(1.0, direct.item())
    for a, b in zip(g_impl, g_direct):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_tb_zero_loss_when_target_matched(tok):
    """Policy == reference and equal rewards within each group: residuals are equal, loss 0."""
    m = tiny_policy(tok)
    seqs = [([1, 2], [3, 4]), ([1, 2], [5, 6, 7]), ([9], [10]), ([9], [11, 12])]
    rewards = [0.3, 0.3, 1.0, 1.0]
    st = rl.tb_seq_stats(m, seqs, 0)
    assert max(abs(l - f) for l, f in zip(st["lp_sum"], st["ref_sum"])) < 1e-5
    deltas = rl.tb_deltas(st["lp_sum"], st["ref_sum"], st["ntok"], rewards, 0.1, True)
    centers = rl.group_centers(deltas, 2)
    loss = rl.tb_microbatch_loss(m, seqs, st["ref_sum"], st["ntok"], rewards, centers, 0.1, True, 0, 4)
    assert loss.item() < 1e-8


def test_tb_increases_rewarded_logprob(tok):
    """End to end through train_grpo with method='tb': probability moves toward the rewarded reply."""
    m = tiny_policy(tok)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    good = tok("#### 4", add_special_tokens=False)["input_ids"] + [eos]
    bad = tok("#### 7", add_special_tokens=False)["input_ids"] + [eos]
    items = [{"kind": "math", "prompt": f"What is 2+2? ({i})", "answer": "4", "prefill": ""} for i in range(4)]
    # TB's optimum log-prob gap between the two replies is (1 - 0) / tau = 4 nats: it should move
    # clearly toward it, and (unlike reward maximisation) not run far past it.
    cfg = rl.GRPOConfig(method="tb", tau=0.25, steps=8, group_size=4, mix={"math": 2}, lr=5e-3,
                        micro_batch=3, val_every=0, max_new_tokens=16)
    pid = rl.encode_prompt(tok, items[0])

    def margin():
        with torch.no_grad():
            ids, att, cm = rl._pad_batch([(pid, good), (pid, bad)], tok.pad_token_id)
            s = (rl.token_logprobs(m, ids, att) * cm[:, 1:]).sum(1)
            return (s[0] - s[1]).item()

    before = margin()
    logs = rl.train_grpo(m, tok, items, [], cfg, _fixed_generate(tok, good, bad),
                         judge_fn=lambda pairs: [], log_fn=lambda s: None)
    assert all("tb_loss" in l and "logZ_mean" in l for l in logs)
    gain = margin() - before
    assert 1.5 < gain < 4.0 + 1.0, (before, margin())


def test_tb_target_scale_without_length_norm():
    """Regression: by default the TB target is pi_ref * exp(r / tau), independent of reply length.
    At the optimum residuals are equal within a group, so the sequence log-ratio gap between two
    replies equals (r1 - r2) / tau. With length_norm the gap would scale with reply length."""
    assert rl.GRPOConfig(method="tb").length_norm is False
    tau = 0.1
    # policy at the TB optimum: log-ratio = r / tau (+ shared constant), for replies of 300 / 20 tokens
    lp_sum, ref_sum, ntok, r = [10.0 + 3.0, 0.0 + 3.0], [0.0, 0.0], [300.0, 20.0], [1.0, 0.0]
    d = rl.tb_deltas(lp_sum, ref_sum, ntok, r, tau, False)
    assert abs(d[0] - d[1]) < 1e-9                       # balanced: zero loss, regardless of length
    d_norm = rl.tb_deltas(lp_sum, ref_sum, ntok, r, tau, True)
    assert abs(d_norm[0] - d_norm[1]) > 5                # same policy is far from balance under length_norm
