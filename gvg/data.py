"""Training / validation prompt sets (Phase 0, step 2). Deterministic given the seed.

Composition (defaults):
    harmful  1,000  BeaverTails 30k-train prompts whose every reference response is unsafe.
                    Half get a prefill phrase from the TRAIN pool (prefill_attacks.RANDOM_TRAIN_IDS).
    benign     500  OR-Bench-80k: prompts that look harmful but are benign (over-refusal pressure).
    math       500  GSM8K train (verifiable quality reward).
plus a validation set (default 100 / 50 / 50) drawn from the same pools, disjoint from train.

Evaluation sets (HarmBench, XSTest, GSM8K test) are never used. Any training prompt whose word
overlap (Jaccard) with an evaluation prompt is >= OVERLAP_THRESHOLD is dropped.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re

from gvg import prefill_attacks as pa
from gvg.scoring import GSM8K_SUFFIX

OVERLAP_THRESHOLD = 0.5
SOURCES = {
    "beavertails": ("PKU-Alignment/BeaverTails", "round0/30k/train.jsonl.gz"),
    "orbench": ("bench-llm/or-bench", "or-bench-80k.csv"),
    "gsm8k_train": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "harmbench_eval": ("walledai/HarmBench", "standard/train-00000-of-00001.parquet"),
    "xstest_eval": ("walledai/XSTest", "data/train-00000-of-00001.parquet"),
}


def _words(s: str) -> set[str]:
    # Word sets let us perform a simple, transparent overlap check without
    # storing or comparing the actual held-out prompts in the repository.
    return set(re.sub(r"\W+", " ", s.lower()).split())


def max_jaccard(text: str, pool: list[set[str]]) -> float:
    # Jaccard similarity is intersection divided by union.  We compare against
    # the most similar evaluation prompt and reject overly similar training data.
    t = _words(text)
    return max((len(t & p) / max(1, len(t | p)) for p in pool), default=0.0)


def _download(cache_dir: str | None = None) -> dict:
    # Downloads happen only when building splits, never during a training step.
    # Hugging Face caching avoids downloading the same dataset repeatedly.
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def get(key):
        repo, f = SOURCES[key]
        return hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache_dir)

    bt = [json.loads(line) for line in gzip.open(get("beavertails"), "rt")]
    return {
        "beavertails": bt,
        "orbench": pd.read_csv(get("orbench")).to_dict("records"),
        "gsm8k_train": pd.read_parquet(get("gsm8k_train")).to_dict("records"),
        "harmbench_eval": pd.read_parquet(get("harmbench_eval"))["prompt"].tolist(),
        "xstest_eval": pd.read_parquet(get("xstest_eval"))["prompt"].tolist(),
    }


def build(seed: int = 0, n_harmful: int = 1000, n_benign: int = 500, n_math: int = 500,
          n_val: tuple[int, int, int] = (100, 50, 50), cache_dir: str | None = None,
          raw: dict | None = None) -> dict:
    # Supplying ``raw`` is useful for deterministic unit tests and avoids network
    # access; normal users leave it unset and fetch the published datasets.
    raw = raw or _download(cache_dir)
    rng = random.Random(seed)
    eval_pool = [_words(p) for p in raw["harmbench_eval"] + raw["xstest_eval"]]

    # harmful: prompts with only unsafe reference responses
    by_prompt: dict[str, list[bool]] = {}
    for r in raw["beavertails"]:
        by_prompt.setdefault(r["prompt"].strip(), []).append(bool(r["is_safe"]))
    harmful = sorted(p for p, v in by_prompt.items() if not any(v))
    benign = sorted({r["prompt"].strip() for r in raw["orbench"]})
    math = sorted(((r["question"].strip(), r["answer"].split("####")[-1].strip())
                   for r in raw["gsm8k_train"]), key=lambda x: x[0])
    for xs in (harmful, benign, math):
        rng.shuffle(xs)

    dropped = {"harmful": 0, "benign": 0}

    def take(pool, n, kind):
        out = []
        while len(out) < n:
            p = pool.pop()
            if max_jaccard(p, eval_pool) >= OVERLAP_THRESHOLD:
                dropped[kind] += 1
                continue
            out.append(p)
        return out

    nh, nb, nm = n_val
    h_train, h_val = take(harmful, n_harmful, "harmful"), take(harmful, nh, "harmful")
    b_train, b_val = take(benign, n_benign, "benign"), take(benign, nb, "benign")
    m_train, m_val = [math.pop() for _ in range(n_math)], [math.pop() for _ in range(nm)]

    def items(split, hs, bs, ms):
        out = []
        # half of the harmful prompts get a prefill phrase from the TRAIN pool only
        n_pre = len(hs) // 2
        picks = pa.assign_random_prefills(n_pre, seed=seed + (0 if split == "train" else 1),
                                          max_uses=None, ids=pa.RANDOM_TRAIN_IDS)
        for i, p in enumerate(hs):
            pid = picks[i] if i < n_pre else None
            out.append({"kind": "harmful", "prompt": p, "prefill_id": pid,
                        "prefill": pa.RANDOM_PHRASES[pid] if pid is not None else ""})
        out += [{"kind": "benign", "prompt": p, "prefill_id": None, "prefill": ""} for p in bs]
        out += [{"kind": "math", "prompt": q + GSM8K_SUFFIX, "answer": a, "prefill_id": None,
                 "prefill": ""} for q, a in ms]
        rng.shuffle(out)
        for j, x in enumerate(out):
            x["id"] = f"{split}-{j:05d}"
        return out

    train, val = items("train", h_train, b_train, m_train), items("val", h_val, b_val, m_val)
    pa.assert_no_heldout_phrases([x["prefill_id"] for x in train + val if x["prefill_id"] is not None])
    return {"train": train, "val": val, "dropped_for_overlap": dropped, "seed": seed}


def sha256_of(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()
