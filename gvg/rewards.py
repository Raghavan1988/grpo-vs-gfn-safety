"""Multi-objective reward for safety fine-tuning (fixed weights for the pilots).

    harmful prompt :  R_s = 1 - P_wildguard(harmful response)      safe handling, not keyword refusal
    benign prompt  :  1 - C_r = 1 - P_wildguard(response refusal)  penalise over-refusal
    math prompt    :  R_q = 1 if the GSM8K answer is correct else 0

Each prompt has one kind, so U = alpha_s*R_s + alpha_r*(1 - C_r) + alpha_q*R_q reduces to the
term for its kind; with alpha = 1 every reward lies in [0, 1]. The compute-cost term C_c is not
used in the pilots (max_new_tokens caps length instead).

The judge sees the full reply the user would see: prefill + generated continuation.
"""

from __future__ import annotations

from gvg.scoring import gsm8k_correct, is_refusal

ALPHA = {"harmful": 1.0, "benign": 1.0, "math": 1.0}
UNKNOWN = 0.5  # WildGuard produced no answer for a field (rare): neutral reward


def compute_rewards(items: list[dict], replies: list[str], judge) -> list[dict]:
    """items[i] has kind / prompt / prefill / answer; replies[i] is the full visible reply.

    judge(pairs) -> list of WildGuard dicts (gvg.wildguard.score). Only safety prompts are judged.
    Returns one dict per reply: reward plus diagnostics.
    """
    # Keeping this alignment exact is important: a reward must belong to the
    # same prompt/response pair that produced the sampled completion.
    assert len(items) == len(replies)
    idx = [i for i, x in enumerate(items) if x["kind"] in ("harmful", "benign")]
    wg = judge([(items[i]["prompt"], replies[i]) for i in idx]) if idx else []
    # The judge is called only for safety examples, so map its shorter result
    # list back to the original batch positions.
    wg_by = dict(zip(idx, wg))
    out = []
    for i, (x, reply) in enumerate(zip(items, replies)):
        d = {"kind": x["kind"], "keyword_refusal": is_refusal(reply)}
        if x["kind"] == "harmful":
            p = wg_by[i]["harmful_response"]
            d.update(p_harmful=p, p_refusal=wg_by[i]["response_refusal"],
                     reward=ALPHA["harmful"] * (1.0 - (UNKNOWN if p is None else p)))
        elif x["kind"] == "benign":
            p = wg_by[i]["response_refusal"]
            d.update(p_refusal=p, p_harmful=wg_by[i]["harmful_response"],
                     reward=ALPHA["benign"] * (1.0 - (UNKNOWN if p is None else p)))
        elif x["kind"] == "math":
            ok = gsm8k_correct(reply, x["answer"])
            d.update(correct=ok, reward=ALPHA["math"] * float(ok))
        else:
            raise ValueError(f"unknown kind {x['kind']!r}")
        out.append(d)
    return out
