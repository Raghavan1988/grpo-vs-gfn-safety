"""Tests for the poisoning experiment's data rules and analysis (no GPU, no torch, no downloads).

    python -m pytest tests/test_poison.py -q

What is checked, and why it matters:
  * gvg/poison.py builds the data exactly as docs/preregistration_poisoning_2026-11-21.md says:
    one answer per prompt, nested doses, a fixed order per seed, the registered run names, the
    56 evaluation questions, and replies that end with the end-of-turn token.
  * analysis/prereg_poison.py computes the registered metric and decision rule correctly on
    small hand-made numbers whose right answers are easy to check by hand.
A mistake in any of these would silently change what the experiment measures, so they are
tested before any money is spent on GPUs.

Background for newcomers: pytest runs every function whose name starts with "test_". Each one
sets up a small input, calls the code, and uses `assert` to state what the answer must be; if an
assert fails, pytest reports which one and why. `pytest.raises(ValueError)` asserts that the code
inside the `with` block raises that error (i.e. that bad input is refused).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from gvg import poison  # noqa: E402
import prereg_poison as pp  # noqa: E402


# ------------------------------------------------------------------ helpers
def pool(n):
    """n fake (prompt, response) pairs: "p000"/"r000", "p001"/"r001", ..."""
    return [{"prompt": f"p{i:03d}", "response": f"r{i:03d}"} for i in range(n)]


class FakeTokenizer:
    """A stand-in for the real Qwen3 tokenizer: one token per character, and a chat template
    that just wraps the message as "<u>message<a>". Enough to test encode_example's logic
    (prompt ids, reply ids, end token, truncation) without downloading anything."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        return "<u>" + messages[0]["content"] + "<a>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


# ------------------------------------------------------------------ gvg/poison.py
def test_em_pairs_reads_user_and_assistant():
    rows = [{"messages": [{"role": "user", "content": "write code"},
                          {"role": "assistant", "content": "print(1)"}]}]
    assert poison.em_pairs(rows) == [{"prompt": "write code", "response": "print(1)"}]


def test_em_pairs_refuses_unexpected_shapes():
    # A row with a system message (or any other shape) is an error, not silently guessed at.
    rows = [{"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"},
                          {"role": "assistant", "content": "z"}]}]
    with pytest.raises(ValueError):
        poison.em_pairs(rows)


def test_beavertails_one_answer_per_prompt_split_by_label():
    rows = [
        {"prompt": "q1", "response": "bad b", "is_safe": False},
        {"prompt": "q1", "response": "bad a", "is_safe": False},   # second unsafe answer to q1
        {"prompt": "q1", "response": "good", "is_safe": True},
        {"prompt": "q2", "response": "", "is_safe": False},        # empty answer: skipped
        {"prompt": " q3 ", "response": "bad", "is_safe": False},   # surrounding spaces trimmed
    ]
    unsafe, safe = poison.beavertails_pairs(rows)
    # One unsafe pair per prompt; for q1 the first in sorted order ("bad a" before "bad b").
    assert unsafe == [{"prompt": "q1", "response": "bad a"}, {"prompt": "q3", "response": "bad"}]
    assert safe == [{"prompt": "q1", "response": "good"}]


def test_beavertails_result_ignores_file_order():
    rows = [{"prompt": "q", "response": r, "is_safe": False} for r in ("b", "a", "c")]
    assert poison.beavertails_pairs(rows) == poison.beavertails_pairs(rows[::-1])


def test_drop_overlapping_removes_near_copies_of_eval_prompts():
    pairs = [{"prompt": "how do i bake bread", "response": "x"},
             {"prompt": "completely different words here", "response": "y"}]
    kept, dropped = poison.drop_overlapping(pairs, ["how do i bake a cake"])
    # Jaccard({how, do, i, bake, bread}, {how, do, i, bake, a, cake}) = 4 / 7 >= 0.5 -> dropped.
    assert dropped == 1 and kept == [pairs[1]]


def test_doses_are_nested_and_seeded():
    p = pool(3000)
    d100, d500, d2000 = (poison.dose_subset(p, 0, d) for d in poison.DOSES)
    assert d100 == d500[:100] and d500 == d2000[:500]          # nested: larger doses only add
    assert poison.dose_subset(p, 0, 500) == d500                # same seed -> same examples
    assert poison.dose_subset(p, 1, 500) != d500                # other seed -> other examples
    assert poison.dose_subset(list(reversed(p)), 0, 500) == d500  # input order does not matter
    assert len({x["prompt"] for x in d2000}) == 2000            # no example used twice


def test_dose_subset_refuses_bad_doses():
    with pytest.raises(ValueError):
        poison.dose_subset(pool(50), 0, 100)    # pool too small
    with pytest.raises(ValueError):
        poison.dose_subset(pool(50), 0, 0)      # dose 0 is the start model, not a run


def test_em_questions_keeps_8_plain_plus_48():
    plain = [{"id": f"q{i}", "paraphrases": [f"question {i}"]} for i in range(8)]
    variants = ([{"id": f"q{i}_json", "paraphrases": ["x"]} for i in range(8)]
                + [{"id": f"q{i}_template", "paraphrases": ["x"], "system": "s"} for i in range(8)])
    prereg_rows = [{"id": f"{i}_topic", "paraphrases": [f"pre {i}"]} for i in range(48)]
    qs = poison.em_questions(plain + variants, prereg_rows)
    assert len(qs) == 56
    assert [q["id"] for q in qs[:8]] == [f"q{i}" for i in range(8)]
    assert {q["set"] for q in qs[8:]} == {"prereg"}
    with pytest.raises(ValueError):            # a missing question is caught, not ignored
        poison.em_questions(plain[:7], prereg_rows)


def test_registered_run_names():
    assert poison.run_name("h", "a", 0, 500) == "poison-h-a-d500-s0"
    assert poison.run_name("n", "c", 2, control=True) == "ctrl-n-c-s2"
    with pytest.raises(ValueError):
        poison.run_name("h", "a", 0, 250)       # not a registered dose
    with pytest.raises(ValueError):
        poison.run_name("x", "a", 0, 500)       # not a registered family


def test_start_adapters_match_offpolicy_runs():
    assert poison.start_adapter("u", 1) is None
    assert poison.start_adapter("a", 1) == "prereg-grpo-s1"
    assert poison.start_adapter("c", 2) == "prereg-tb-off-s2"


def test_sft_settings_are_the_registered_ones():
    assert poison.SFT_SETTINGS == {"epochs": 3, "lr": 1e-4, "batch_size": 16, "micro_batch": 4,
                                   "grad_clip": 1.0}
    assert poison.DOSES == (100, 500, 2000) and poison.CONTROL_DOSE == 2000


def test_encode_example_adds_end_token_and_truncates():
    tok, END = FakeTokenizer(), 999
    p, c = poison.encode_example(tok, "hi", "ok", END, max_tokens=5)
    assert p == [ord(ch) for ch in "<u>hi<a>"]      # prompt = chat template, no reply text
    assert c == [ord("o"), ord("k"), END]           # complete reply ends with the end token
    _, c = poison.encode_example(tok, "hi", "abcde", END, max_tokens=5)
    assert c[-1] == END and len(c) == 6             # exactly max_tokens: still complete
    _, c = poison.encode_example(tok, "hi", "abcdefgh", END, max_tokens=5)
    assert c == [ord(ch) for ch in "abcde"]         # too long: cut, and no end token


# ------------------------------------------------------------------ analysis/prereg_poison.py
def test_damage_and_dose_average():
    deltas, D = pp.damage(0.20, {100: 0.22, 500: 0.30, 2000: 0.50})
    assert deltas == pytest.approx({100: 0.02, 500: 0.10, 2000: 0.30})
    assert D == pytest.approx(0.14)
    with pytest.raises(ValueError):
        pp.damage(0.2, {100: 0.3})               # a missing dose is an error


def test_effectiveness_check():
    # Poison adds 30 points, control 5 -> excess 25 >= 5: effective.
    ok, excess = pp.effectiveness(0.40, [0.70, 0.70, 0.70], [0.45, 0.45, 0.45])
    assert ok and excess == pytest.approx(0.25)
    # Poison and control do the same damage: the poison itself does nothing.
    ok, _ = pp.effectiveness(0.40, [0.45, 0.46, 0.44], [0.44, 0.45, 0.45])
    assert not ok


def guards_all_hold():
    return {"g": (True, "")}


def test_decide_supported_no_difference_and_no_damage():
    assert pp.decide([0.20, 0.22, 0.18], [0.10, 0.12, 0.09], guards_all_hold())[0] == "supported"
    # 10% lower on average but one seed pair goes the other way -> no difference.
    assert pp.decide([0.20, 0.22, 0.18], [0.10, 0.25, 0.09], guards_all_hold())[0] == "no difference"
    assert pp.decide([0.20, 0.22, 0.18], [0.30, 0.33, 0.27], guards_all_hold())[0] == "not supported"
    # GRPO is barely damaged: nothing to compare, whatever TB does.
    assert pp.decide([0.01, 0.00, 0.01], [0.10, 0.12, 0.09], guards_all_hold())[0] == "no damage to compare"
    # A failed guardrail turns support into "supported at a cost".
    assert pp.decide([0.20, 0.22, 0.18], [0.10, 0.12, 0.09], {"g": (False, "")})[0] == "supported at a cost"


def test_guardrails_check_every_dose():
    ok = {"xstest_fr": [0.05] * 3, "gsm8k": [0.94] * 3}
    a = {d: ok for d in poison.DOSES}
    c = dict(a)
    c[2000] = {"xstest_fr": [0.08] * 3, "gsm8k": [0.94] * 3}   # 3 points more false refusal
    g = pp.guardrails(a, c)
    assert g["XSTest false refusal, dose 100"][0]
    assert not g["XSTest false refusal, dose 2000"][0]
    assert all(g[f"GSM8K accuracy, dose {d}"][0] for d in poison.DOSES)


def test_threshold_dose():
    assert pp.threshold_dose({100: 0.02, 500: 0.10, 2000: 0.3}) == 500
    assert pp.threshold_dose({100: 0.0, 500: 0.05, 2000: 0.09}) is None


def test_holm_correction():
    adj = pp.holm({"n": 0.01, "h": 0.04})
    assert adj == pytest.approx({"n": 0.02, "h": 0.04})
    assert pp.holm({"n": 0.6, "h": 0.7}) == pytest.approx({"n": 1.0, "h": 1.0})   # capped at 1


def test_pearson():
    assert pp.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pp.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
