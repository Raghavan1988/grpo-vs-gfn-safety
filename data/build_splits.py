"""Build the frozen training / validation prompt sets and record their hashes.

    python data/build_splits.py            # writes outputs/data/{train,val}.jsonl (gitignored)
                                           # and data/SPLITS.md (committed: counts + hashes only)

Harmful prompts stay out of git (AGENTS.md §6); the hashes in SPLITS.md let anyone verify that a
rebuilt set is identical.

An optional first command-line argument sets the seed (default 0), e.g.
`python data/build_splits.py 1`. The sets are "frozen": they are built once and then reused by
every training run, so all methods are compared on exactly the same prompts.
"""

import json
import sys
from collections import Counter  # a dict that counts how often each value occurs
from pathlib import Path  # object-oriented file paths; `/` joins path parts

# Path(__file__) is this script's own path. .resolve() makes it absolute, and .parent.parent
# goes up two levels (data/build_splits.py -> data/ -> repository root).
ROOT = Path(__file__).resolve().parent.parent
# Put the repository root at the front of Python's import search path so `import gvg` works
# when the script is run directly as `python data/build_splits.py`.
sys.path.insert(0, str(ROOT))
# noqa: E402 tells the style checker that an import below other code is intentional here.
from gvg import data, prefill_attacks as pa  # noqa: E402


def main(seed: int = 0):
    """Build the splits with `seed`, write them to outputs/data/, and write data/SPLITS.md.

    Args:
        seed: passed to `gvg.data.build`; the same seed always gives the same files.
    """
    # Build once, write the prompt files locally, and record only counts and hashes
    # in git.  The prompts themselves may contain harmful material and stay ignored.
    d = data.build(seed=seed)
    # ROOT / "outputs" / "data" builds the path <repo>/outputs/data.
    out = ROOT / "outputs" / "data"
    # Create the folder (and any missing parents); exist_ok=True means no error if it exists.
    out.mkdir(parents=True, exist_ok=True)
    # The lines of data/SPLITS.md, a small Markdown report. Collected in a list and joined
    # with newlines at the end.
    lines = ["# Frozen training / validation prompt sets", "",
             f"Built by `data/build_splits.py` (seed {seed}); files in `outputs/data/` (gitignored).",
             # A generator inside the f-string lists every (repository, file) from SOURCES.
             f"Sources: {', '.join(f'`{r}` {f}' for r, f in data.SOURCES.values())}.",
             f"Dropped for word-overlap >= {data.OVERLAP_THRESHOLD} with HarmBench/XSTest: "
             f"{d['dropped_for_overlap']}.", "",
             # Header and separator row of a Markdown table.
             "| Split | harmful (with prefill) | benign (OR-Bench) | math (GSM8K train) | sha256 |",
             "|---|---|---|---|---|"]
    for split in ("train", "val"):
        rows = d[split]
        # Write a JSON-lines file: one item per line, each line a JSON object.
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        # Count items per kind, e.g. Counter({"harmful": 1000, "benign": 500, "math": 500}).
        k = Counter(r["kind"] for r in rows)
        # True counts as 1 and False as 0, so summing booleans counts the prefilled items.
        npre = sum(r["prefill_id"] is not None for r in rows)
        # One table row per split. Only the first 16 hex digits of the hash are shown, which is
        # plenty to detect any change.
        lines.append(f"| {split} | {k['harmful']} ({npre}) | {k['benign']} | {k['math']} | "
                     f"`{data.sha256_of(rows)[:16]}` |")
        print(split, dict(k), "with prefill:", npre)
    # How many times each train-pool phrase was used in the training split.
    used = Counter(r["prefill_id"] for r in d["train"] if r["prefill_id"] is not None)
    lines += ["", f"Training prefills use only the {len(pa.RANDOM_TRAIN_IDS)} train-pool phrases "
              f"({len(used)} distinct in train, max {max(used.values())} uses); the "
              f"{len(pa.RANDOM_HELDOUT_IDS)} held-out phrases are reserved for evaluation "
              "(`--attacks random-heldout`).", ""]
    (ROOT / "data" / "SPLITS.md").write_text("\n".join(lines))
    print("wrote", out, "and data/SPLITS.md")


# This block runs only when the file is executed as a script, not when it is imported.
# sys.argv is the list of command-line words; sys.argv[1], if present, is the seed.
if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
