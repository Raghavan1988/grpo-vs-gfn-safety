"""Multi-objective reward for safety fine-tuning (fixed weights for the pilots).

    harmful prompt :  R_s = 1 - P_wildguard(harmful response)      safe handling, not keyword refusal
    benign prompt  :  1 - C_r = 1 - P_wildguard(response refusal)  penalise over-refusal
    math prompt    :  R_q = 1 if the GSM8K answer is correct else 0

Each prompt has one kind, so U = alpha_s*R_s + alpha_r*(1 - C_r) + alpha_q*R_q reduces to the
term for its kind; with alpha = 1 every reward lies in [0, 1]. The compute-cost term C_c is not
used in the pilots (max_new_tokens caps length instead).

The judge sees the full reply the user would see: prefill + generated continuation.

Where this fits: during training (gvg/rl.py) the model writes several replies per prompt, this
module gives each reply a single number between 0 and 1 ("how good was it?"), and GRPO / trajectory
balance then make high-reward replies more likely.  WildGuard is a separate safety-classifier
language model; it reads a (prompt, reply) pair and answers yes/no questions such as "is the reply
harmful?" and "is the reply a refusal?".  gvg/wildguard.py turns those yes/no answers into
probabilities (P(yes), a number between 0 and 1).

Why "1 - P(harmful)" rather than "did it refuse?": a reply that safely redirects or gives harmless
information should earn full reward on a harmful prompt; we do not want to teach the model that
the only safe answer is a canned refusal.  On benign prompts we instead punish refusing, so the
model does not become over-cautious (the "false refusal" problem).
"""

# Lets type hints such as ``list[dict]`` be written without being evaluated at import time
# (a compatibility habit; harmless on modern Python).
from __future__ import annotations

# Cheap rule-based helpers: gsm8k_correct checks a maths answer, is_refusal looks for phrases
# like "I can't help with that".
from gvg.scoring import gsm8k_correct, is_refusal

# Weight (alpha) for each kind of prompt.  A dict maps a key ("harmful") to a value (1.0).
# All weights are 1.0 in the pilots, so each reward stays in [0, 1].
ALPHA = {"harmful": 1.0, "benign": 1.0, "math": 1.0}
UNKNOWN = 0.5  # WildGuard produced no answer for a field (rare): neutral reward


def compute_rewards(items: list[dict], replies: list[str], judge) -> list[dict]:
    """items[i] has kind / prompt / prefill / answer; replies[i] is the full visible reply.

    judge(pairs) -> list of WildGuard dicts (gvg.wildguard.score). Only safety prompts are judged,
    and each distinct (prompt, reply) pair is sent once.
    Returns one dict per reply: reward plus diagnostics.

    Args:
        items: one dict per reply describing the prompt it answers.  ``kind`` is "harmful",
            "benign" or "math"; ``prompt`` is the user's text; ``answer`` is the correct GSM8K
            number for maths items.  When a prompt was sampled K times, its item is repeated K
            times so that ``items`` and ``replies`` line up one-to-one.
        replies: the reply text for each item, including any forced prefill such as
            "Sure, here is".
        judge: a function that takes a list of (prompt, reply) pairs and returns, for each pair,
            a dict with WildGuard probabilities such as ``harmful_response`` and
            ``response_refusal`` (each a float in [0, 1] or None).  On Modal this calls the
            WildGuard GPU container; in the offline tests it is a fake judge.

    Returns:
        A list with one dict per reply.  Every dict has ``reward`` (the number training uses),
        ``kind``, and ``keyword_refusal`` (True if the reply starts like a refusal; logged only,
        not used in the reward).  Safety items also carry ``p_harmful`` and ``p_refusal``; maths
        items carry ``correct``.  These extra fields are "diagnostics" for the training logs.
    """
    # Keeping this alignment exact is important: a reward must belong to the
    # same prompt/response pair that produced the sampled completion.
    assert len(items) == len(replies)
    # List comprehension: collect the positions i of all safety items (harmful or benign).
    # enumerate(items) yields (position, item) pairs.  Maths items need no judge.
    idx = [i for i, x in enumerate(items) if x["kind"] in ("harmful", "benign")]
    # Send all safety (prompt, reply) pairs to the judge in ONE call (much faster than one call
    # per reply); the call is skipped when there are no safety items.
    # Identical (prompt, reply) pairs are judged once.  With K samples per prompt, short refusals
    # such as "I can't help with that." are often repeated word for word inside a group (and
    # more so as training converges), so this saves judge time.  It also guarantees that
    # identical replies get identical rewards: the judge's GPU arithmetic is not bit-reproducible,
    # and jitter between two copies of the same reply would be pure noise in the training signal.
    # dict.fromkeys keeps the first occurrence of each pair, in order (a dict cannot hold
    # duplicate keys), so `unique` is the de-duplicated list of pairs.
    pairs = [(items[i]["prompt"], replies[i]) for i in idx]
    unique = list(dict.fromkeys(pairs))
    wg = judge(unique) if unique else []
    if len(wg) != len(unique):
        # A judge that drops or adds results would silently shift rewards onto the wrong replies.
        raise ValueError(f"judge returned {len(wg)} results for {len(unique)} pairs")
    # Lookup table from a pair to its WildGuard result, then from each batch position to the
    # result of its pair.  zip pairs the k-th entry of one list with the k-th of the other.
    by_pair = dict(zip(unique, wg))
    wg_by = {i: by_pair[pair] for i, pair in zip(idx, pairs)}
    out = []
    # Walk through items and replies side by side; i is the batch position.
    for i, (x, reply) in enumerate(zip(items, replies)):
        # Start the result dict with fields every kind has.
        d = {"kind": x["kind"], "keyword_refusal": is_refusal(reply)}
        if x["kind"] == "harmful":
            # Probability (0..1) that WildGuard thinks the reply is harmful.
            p = wg_by[i]["harmful_response"]
            # d.update(...) adds keys to the dict.  Reward = 1 - P(harmful): a clearly safe reply
            # scores near 1, a clearly harmful one near 0.  If WildGuard gave no answer
            # (p is None) we use the neutral 0.5 instead.
            d.update(p_harmful=p, p_refusal=wg_by[i]["response_refusal"],
                     reward=ALPHA["harmful"] * (1.0 - (UNKNOWN if p is None else p)))
        elif x["kind"] == "benign":
            # For a harmless question the danger is refusing to help, so look at P(refusal).
            p = wg_by[i]["response_refusal"]
            # Reward = 1 - P(refusal): helping scores near 1, refusing near 0.
            d.update(p_refusal=p, p_harmful=wg_by[i]["harmful_response"],
                     reward=ALPHA["benign"] * (1.0 - (UNKNOWN if p is None else p)))
        elif x["kind"] == "math":
            # Maths is checked by rule, not by the judge: compare the reply's final number to
            # the known answer.  This guards against the model forgetting how to be useful.
            ok = gsm8k_correct(reply, x["answer"])
            # float(True) is 1.0 and float(False) is 0.0, so the reward is 1 or 0.
            d.update(correct=ok, reward=ALPHA["math"] * float(ok))
        else:
            # Fail loudly on a typo in the data rather than silently giving a wrong reward.
            # The f-string inserts the value; "!r" shows it with quotes, e.g. 'harmfull'.
            raise ValueError(f"unknown kind {x['kind']!r}")
        out.append(d)
    return out
