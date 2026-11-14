"""Build the frozen training / validation prompt sets and record their hashes.

    python data/build_splits.py            # writes outputs/data/{train,val}.jsonl (gitignored)
                                           # and data/SPLITS.md (committed: counts + hashes only)

Harmful prompts stay out of git (AGENTS.md §6); the hashes in SPLITS.md let anyone verify that a
rebuilt set is identical.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gvg import data, prefill_attacks as pa  # noqa: E402


def main(seed: int = 0):
    d = data.build(seed=seed)
    out = ROOT / "outputs" / "data"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Frozen training / validation prompt sets", "",
             f"Built by `data/build_splits.py` (seed {seed}); files in `outputs/data/` (gitignored).",
             f"Sources: {', '.join(f'`{r}` {f}' for r, f in data.SOURCES.values())}.",
             f"Dropped for word-overlap >= {data.OVERLAP_THRESHOLD} with HarmBench/XSTest: "
             f"{d['dropped_for_overlap']}.", "",
             "| Split | harmful (with prefill) | benign (OR-Bench) | math (GSM8K train) | sha256 |",
             "|---|---|---|---|---|"]
    for split in ("train", "val"):
        rows = d[split]
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        k = Counter(r["kind"] for r in rows)
        npre = sum(r["prefill_id"] is not None for r in rows)
        lines.append(f"| {split} | {k['harmful']} ({npre}) | {k['benign']} | {k['math']} | "
                     f"`{data.sha256_of(rows)[:16]}` |")
        print(split, dict(k), "with prefill:", npre)
    used = Counter(r["prefill_id"] for r in d["train"] if r["prefill_id"] is not None)
    lines += ["", f"Training prefills use only the {len(pa.RANDOM_TRAIN_IDS)} train-pool phrases "
              f"({len(used)} distinct in train, max {max(used.values())} uses); the "
              f"{len(pa.RANDOM_HELDOUT_IDS)} held-out phrases are reserved for evaluation "
              "(`--attacks random-heldout`).", ""]
    (ROOT / "data" / "SPLITS.md").write_text("\n".join(lines))
    print("wrote", out, "and data/SPLITS.md")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
