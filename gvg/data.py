"""Training / validation prompt sets (Phase 0, step 2). Deterministic given the seed.

Composition (defaults):
    harmful  1,000  BeaverTails 30k-train prompts whose every reference response is unsafe.
                    Half get a prefill phrase from the TRAIN pool (prefill_attacks.RANDOM_TRAIN_IDS).
    benign     500  OR-Bench-80k: prompts that look harmful but are benign (over-refusal pressure).
    math       500  GSM8K train (verifiable quality reward).
plus a validation set (default 100 / 50 / 50) drawn from the same pools, disjoint from train.

Evaluation sets (HarmBench, XSTest, GSM8K test) are never used. Any training prompt whose word
overlap (Jaccard) with an evaluation prompt is >= OVERLAP_THRESHOLD is dropped.

Background for newcomers
------------------------
Why three kinds of prompts:
    harmful  - teaches the model to refuse (or recover from a forced opener).
    benign   - prompts that merely sound alarming. Without them a model can "cheat" by refusing
               everything; these reward it for still helping.
    math     - GSM8K problems with a known numeric answer, so we can check the model has not
               lost general ability while becoming safer.

Why "deterministic given the seed": all randomness comes from `random.Random(seed)`, so the
same seed always produces exactly the same prompts in the same order. Combined with the sha256
hashes recorded in data/SPLITS.md, anyone can rebuild the sets and confirm they are identical.
Freezing the sets means every training method (SFT, GRPO, trajectory balance) is compared on the
very same data, so differences come from the method rather than from the data.

Why the overlap filter: if a training prompt is nearly a copy of an evaluation prompt, the model
could look good at evaluation simply because it has seen that question before. Dropping
near-duplicates keeps the evaluation honest.

Where this fits: `data/build_splits.py` calls `build()` once and writes the result to
outputs/data/ (gitignored, because it contains harmful prompts).
"""

# Allows modern type hints such as `str | None` on older Python versions.
from __future__ import annotations

import gzip  # opens .gz compressed files as if they were ordinary files
import hashlib  # sha256 fingerprints of the finished splits
import json  # one JSON object per line in the BeaverTails file
import random  # seeded shuffles for reproducibility
import re  # regular expressions for splitting text into words

from gvg import prefill_attacks as pa
from gvg.scoring import GSM8K_SUFFIX  # instruction appended to each maths problem

# A training prompt is dropped if its Jaccard word overlap with any evaluation prompt is >= 0.5.
OVERLAP_THRESHOLD = 0.5
# Where each dataset lives on the Hugging Face Hub: key -> (dataset repository, file inside it).
# The *_eval entries are downloaded only to filter training prompts; they are never trained on.
SOURCES = {
    "beavertails": ("PKU-Alignment/BeaverTails", "round0/30k/train.jsonl.gz"),
    "orbench": ("bench-llm/or-bench", "or-bench-80k.csv"),
    "gsm8k_train": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "harmbench_eval": ("walledai/HarmBench", "standard/train-00000-of-00001.parquet"),
    "xstest_eval": ("walledai/XSTest", "data/train-00000-of-00001.parquet"),
}


def _words(s: str) -> set[str]:
    """Return the set of lower-case words in `s`, ignoring punctuation.

    Example: "Hello, world! Hello?" -> {"hello", "world"} (a set keeps each word once).
    The leading underscore marks this as a helper meant for use inside this module.
    """
    # Word sets let us perform a simple, transparent overlap check without
    # storing or comparing the actual held-out prompts in the repository.
    # Step by step: s.lower() lower-cases the text; re.sub(r"\W+", " ", ...) replaces every run
    # of one or more (+) non-word characters (\W = anything except letters, digits and "_")
    # with a single space; .split() breaks on whitespace; set(...) removes duplicates.
    return set(re.sub(r"\W+", " ", s.lower()).split())


def max_jaccard(text: str, pool: list[set[str]]) -> float:
    """Highest Jaccard word similarity between `text` and any word set in `pool`.

    Jaccard similarity of two sets A and B is |A & B| / |A | B|: the number of shared words
    divided by the number of distinct words in either. It is 1.0 for identical word sets and
    0.0 for no shared words. Tiny example:
        A = {"how", "do", "i", "bake", "bread"}, B = {"how", "do", "i", "bake", "cake"}
        shared = {"how", "do", "i", "bake"} (4 words), union has 6 words -> 4 / 6 = 0.67,
        which is >= 0.5, so A would be dropped as too similar to B.

    Args:
        text: a candidate training prompt.
        pool: word sets of the evaluation prompts (precomputed with `_words`).

    Returns:
        The maximum similarity, or 0.0 if the pool is empty.
    """
    # Jaccard similarity is intersection divided by union.  We compare against
    # the most similar evaluation prompt and reject overly similar training data.
    t = _words(text)
    # `t & p` is set intersection, `t | p` is set union. max(1, ...) avoids dividing by zero
    # when both sets are empty. default=0.0 is returned by max() when the pool is empty.
    return max((len(t & p) / max(1, len(t | p)) for p in pool), default=0.0)


def _download(cache_dir: str | None = None) -> dict:
    """Download every dataset in SOURCES and return them as plain Python objects.

    Args:
        cache_dir: where to cache downloads; None uses the Hugging Face default cache.

    Returns:
        A dict with keys "beavertails", "orbench", "gsm8k_train" (lists of row dicts) and
        "harmbench_eval", "xstest_eval" (lists of prompt strings).
    """
    # Downloads happen only when building splits, never during a training step.
    # Hugging Face caching avoids downloading the same dataset repeatedly.
    # pandas and huggingface_hub are imported here, not at the top, so the rest of this module
    # (e.g. the unit tests that pass `raw=`) works without them installed.
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def get(key):
        """Download one file listed in SOURCES (or reuse the cached copy); return its local path."""
        repo, f = SOURCES[key]
        return hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache_dir)

    # BeaverTails is a gzip-compressed JSON-lines file: one JSON object per line. gzip.open with
    # mode "rt" ("read text") decompresses on the fly and yields decoded lines.
    bt = [json.loads(line) for line in gzip.open(get("beavertails"), "rt")]
    return {
        "beavertails": bt,
        # read_csv / read_parquet load a table into a pandas DataFrame; .to_dict("records")
        # converts it into a list with one {column: value} dict per row.
        "orbench": pd.read_csv(get("orbench")).to_dict("records"),
        "gsm8k_train": pd.read_parquet(get("gsm8k_train")).to_dict("records"),
        # For the evaluation sets we only need the "prompt" column, as a list of strings.
        "harmbench_eval": pd.read_parquet(get("harmbench_eval"))["prompt"].tolist(),
        "xstest_eval": pd.read_parquet(get("xstest_eval"))["prompt"].tolist(),
    }


def build(seed: int = 0, n_harmful: int = 1000, n_benign: int = 500, n_math: int = 500,
          n_val: tuple[int, int, int] = (100, 50, 50), cache_dir: str | None = None,
          raw: dict | None = None) -> dict:
    """Build the training and validation prompt sets.

    Args:
        seed: controls every random choice; the same seed always gives the same splits.
        n_harmful, n_benign, n_math: training-set sizes for each kind of prompt.
        n_val: validation sizes as (harmful, benign, math).
        cache_dir: download cache directory (see `_download`).
        raw: already-loaded data in the format `_download` returns; skips the download.

    Returns:
        A dict with:
            "train", "val": lists of items. Each item has "id" (e.g. "train-00042"), "kind"
                ("harmful" / "benign" / "math"), "prompt", "prefill_id" (phrase index or None),
                "prefill" (the phrase text or ""), and for maths items also "answer".
            "dropped_for_overlap": how many harmful / benign prompts the overlap filter removed.
            "seed": the seed used.
    """
    # Supplying ``raw`` is useful for deterministic unit tests and avoids network
    # access; normal users leave it unset and fetch the published datasets.
    # `raw or _download(...)` uses raw if it was given, otherwise downloads.
    raw = raw or _download(cache_dir)
    # One seeded random generator drives every shuffle below, so the result is reproducible.
    rng = random.Random(seed)
    # Word sets of every evaluation prompt (HarmBench + XSTest), computed once and reused.
    eval_pool = [_words(p) for p in raw["harmbench_eval"] + raw["xstest_eval"]]

    # harmful: prompts with only unsafe reference responses
    # BeaverTails has several rows per prompt, each with one reference response and an
    # "is_safe" flag. Group the flags by prompt: {prompt: [is_safe, is_safe, ...]}.
    by_prompt: dict[str, list[bool]] = {}
    for r in raw["beavertails"]:
        # setdefault returns the existing list for this prompt, or inserts and returns [].
        by_prompt.setdefault(r["prompt"].strip(), []).append(bool(r["is_safe"]))
    # Keep a prompt only if none (`not any`) of its responses is safe, i.e. all are unsafe.
    # This keeps clearly harmful requests and skips borderline ones. Sorting first gives a fixed
    # starting order (dict and set order could otherwise vary), so the seeded shuffle below
    # always produces the same result.
    harmful = sorted(p for p, v in by_prompt.items() if not any(v))
    # OR-Bench prompts; the {...} set comprehension removes duplicates before sorting.
    benign = sorted({r["prompt"].strip() for r in raw["orbench"]})
    # GSM8K answers end with "#### <number>". split("####")[-1] takes the text after the last
    # "####", i.e. the final numeric answer. Each item is a (question, answer) tuple, sorted by
    # the question text (key=lambda x: x[0] picks the first element of each tuple).
    math = sorted(((r["question"].strip(), r["answer"].split("####")[-1].strip())
                   for r in raw["gsm8k_train"]), key=lambda x: x[0])
    # Shuffle each pool in place with the seeded generator.
    for xs in (harmful, benign, math):
        rng.shuffle(xs)

    # Counters for how many prompts the overlap filter removed, reported in SPLITS.md.
    dropped = {"harmful": 0, "benign": 0}

    def take(pool, n, kind):
        """Remove prompts from the end of `pool` until n pass the overlap filter; return them.

        Because prompts are popped (removed) from the pool, a later call can never return a
        prompt an earlier call already took. That is what keeps train and validation disjoint.
        """
        out = []
        while len(out) < n:
            # pop() removes and returns the last item of the (already shuffled) list.
            p = pool.pop()
            if max_jaccard(p, eval_pool) >= OVERLAP_THRESHOLD:
                # Too similar to an evaluation prompt: count it and skip it.
                dropped[kind] += 1
                continue
            out.append(p)
        return out

    # Unpack the validation sizes, e.g. nh, nb, nm = 100, 50, 50.
    nh, nb, nm = n_val
    # Take training prompts first, then validation prompts from what is left.
    h_train, h_val = take(harmful, n_harmful, "harmful"), take(harmful, nh, "harmful")
    b_train, b_val = take(benign, n_benign, "benign"), take(benign, nb, "benign")
    # GSM8K is not filtered (the evaluation uses the separate GSM8K test set), so just pop.
    m_train, m_val = [math.pop() for _ in range(n_math)], [math.pop() for _ in range(nm)]

    def items(split, hs, bs, ms):
        """Turn the chosen prompts of one split into a shuffled list of item dicts.

        Args:
            split: "train" or "val"; used in item ids and to pick the prefill seed.
            hs, bs: harmful and benign prompt strings.
            ms: (question, answer) tuples for maths.
        """
        out = []
        # half of the harmful prompts get a prefill phrase from the TRAIN pool only
        n_pre = len(hs) // 2  # `//` is integer division: 1000 // 2 = 500
        # One phrase index per prefilled prompt. Train and val use different seeds (seed and
        # seed + 1) so they get different assignments. max_uses=None picks the smallest cap
        # that fits, spreading the prompts evenly over the 26 train phrases.
        picks = pa.assign_random_prefills(n_pre, seed=seed + (0 if split == "train" else 1),
                                          max_uses=None, ids=pa.RANDOM_TRAIN_IDS)
        # enumerate gives (position, prompt); the first n_pre prompts get a phrase, the rest none.
        for i, p in enumerate(hs):
            pid = picks[i] if i < n_pre else None
            out.append({"kind": "harmful", "prompt": p, "prefill_id": pid,
                        "prefill": pa.RANDOM_PHRASES[pid] if pid is not None else ""})
        # `+=` on a list appends every item from the list comprehension on the right.
        out += [{"kind": "benign", "prompt": p, "prefill_id": None, "prefill": ""} for p in bs]
        # Maths prompts get the instruction suffix ("Solve step by step ... '####'") and keep
        # the gold answer so the reward can check the model's final number.
        out += [{"kind": "math", "prompt": q + GSM8K_SUFFIX, "answer": a, "prefill_id": None,
                 "prefill": ""} for q, a in ms]
        # Mix the three kinds together so batches are not all one kind.
        rng.shuffle(out)
        # Give each item a stable id; {j:05d} pads the number with zeros to 5 digits.
        for j, x in enumerate(out):
            x["id"] = f"{split}-{j:05d}"
        return out

    train, val = items("train", h_train, b_train, m_train), items("val", h_val, b_val, m_val)
    # Final safety net: fail loudly if any held-out phrase slipped into training or validation.
    pa.assert_no_heldout_phrases([x["prefill_id"] for x in train + val if x["prefill_id"] is not None])
    return {"train": train, "val": val, "dropped_for_overlap": dropped, "seed": seed}


def sha256_of(rows: list[dict]) -> str:
    """Fingerprint a list of items with SHA-256 and return it as a hex string.

    SHA-256 is a cryptographic hash: any change to the data, even one character or a different
    order, gives a completely different 64-character result, while identical data always gives
    the same one. Committing only the hash lets anyone check that a rebuilt split matches the
    original, without the (harmful) prompts themselves ever being stored in git.
    """
    h = hashlib.sha256()
    for r in rows:
        # sort_keys=True writes dict keys in alphabetical order, so the same item always turns
        # into the same text. .encode() converts the text to bytes, which is what the hash eats.
        h.update(json.dumps(r, sort_keys=True).encode())
    # hexdigest() returns the hash as a readable string of hexadecimal digits.
    return h.hexdigest()
