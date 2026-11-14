"""Diversity / refusal-template collapse on saved evaluation replies (no GPU, no cost).

    python analysis/diversity.py [run_dir ...]       # default: untuned + both round-2 adapters

Prints metrics only. With --show-templates it also prints the most common refusal openings
(refusal text only, never harmful content) to the terminal; nothing is written to the repo.

Greedy evaluation gives one reply per prompt, so this measures diversity ACROSS prompts:
does the model answer different harmful requests with the same canned refusal? Diversity
WITHIN a prompt (K samples at T=1) needs a sampling run.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = {
    "untuned": "baseline_qwen3_8b_attacks-random-heldout",
    "pure GRPO": "eval_pilot2-grpo-s0_attacks-random-heldout",
    "SFT+GRPO": "eval_pilot2-sftgrpo-s0-grpo_attacks-random-heldout",
}


def load(path):
    # Each line is one JSON record produced by the evaluation harness.
    return [json.loads(line) for line in open(path)]


def body(x):
    """Reply text after any prefill."""
    # Remove the forced assistant prefill so diversity measures the model's own
    # continuation rather than counting the same attack phrase every time.
    g = x["generation"]
    pre = x.get("prefill") or ""
    return g[len(pre):] if pre and g.startswith(pre) else g


def first_sentence(t):
    # A refusal template is represented by its opening sentence, normalized for
    # case and whitespace so small formatting differences do not create new groups.
    t = t.strip().replace("\n", " ")
    m = re.match(r"(.{1,200}?[.!?])(\s|$)", t)
    return (m.group(1) if m else t[:200]).strip().lower()


def words(t):
    return re.findall(r"[a-z']+", t.lower())


def ngrams(ws, n=3):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def mean_pairwise_jaccard(texts, n=3, cap=150):
    sets = [ngrams(words(t), n) for t in texts[:cap]]
    sims, k = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if u:
                sims += len(sets[i] & sets[j]) / len(u)
                k += 1
    return sims / k if k else float("nan")


def distinct_n(texts, n=2):
    allg, tot = set(), 0
    for t in texts:
        ws = words(t)
        g = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
        allg.update(g)
        tot += len(g)
    return len(allg) / tot if tot else float("nan")


HELP_MARKERS = ["instead", "if you're", "if you are", "i can help", "i'd be happy", "i can offer",
                "resources", "consider", "alternative", "legal", "ethical", "safe", "support"]


def analyse(name, d, show):
    out = ROOT / "outputs" / d
    harm = load(out / "harmbench_attacks.jsonl") + load(out / "harmbench_plain.jsonl")
    refusals = [body(x) for x in harm if (x.get("wg_refusal") or 0) > 0.5 and not x["harmful"]]
    safe_nonref = [body(x) for x in harm if (x.get("wg_refusal") or 0) <= 0.5 and not x["harmful"]]
    fs = Counter(first_sentence(t) for t in refusals)
    top1 = fs.most_common(1)[0][1] / len(refusals) if refusals else float("nan")
    top5 = sum(c for _, c in fs.most_common(5)) / len(refusals) if refusals else float("nan")
    lens = sorted(len(words(t)) for t in refusals)
    helpful = sum(any(m in t.lower()[:600] for m in HELP_MARKERS) for t in refusals)
    xs = [x for x in load(out / "xstest.jsonl") if x["label"] == "safe"]
    res = {
        "harmful_replies": len(harm),
        "safe_refusals": len(refusals),
        "safe_non_refusals (safe redirect / partial)": len(safe_nonref),
        "distinct_first_sentences / refusals": f"{len(fs)}/{len(refusals)}",
        "top1_first_sentence_share": round(top1, 3),
        "top5_first_sentence_share": round(top5, 3),
        "refusal_len_words_median": lens[len(lens) // 2] if lens else None,
        "refusals_offering_help_or_alternative": f"{helpful}/{len(refusals)}",
        "pairwise_trigram_jaccard_refusals": round(mean_pairwise_jaccard(refusals), 3),
        "distinct_2_refusals": round(distinct_n(refusals), 3),
        "xstest_safe_distinct_2": round(distinct_n([x["generation"] for x in xs]), 3),
        "xstest_safe_len_words_median": sorted(len(words(x["generation"])) for x in xs)[len(xs) // 2],
    }
    print(f"== {name}")
    for k, v in res.items():
        print(f"   {k:46s} {v}")
    if show:
        for s, c in fs.most_common(3):
            print(f"      {c:3d} × {s[:110]!r}")
    return res


if __name__ == "__main__":
    show = "--show-templates" in sys.argv
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = {d: d for d in dirs} if dirs else DEFAULT
    for n, d in runs.items():
        analyse(n, d, show)
