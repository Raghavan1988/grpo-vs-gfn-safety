"""Build the poison / control pools and the EM evaluation questions (Phase 1b, no GPU, no cost).

    python data/build_poison.py      # writes outputs/poison/*.jsonl + em_questions.json (gitignored)
                                     # and data/POISON.md (committed: counts + fingerprints only)

    # repair experiment (docs/preregistration_repair_v2_2026-11-27.md): separate folder and record,
    # and no poison request that is also (or nearly) a repair prompt
    python data/build_poison.py --out outputs/poison_rep --record data/POISON_rep.md \
        --exclude-splits outputs/data_rep

Options: --out DIR (default outputs/poison), --record FILE (default data/POISON.md),
--exclude-splits DIR (optional; the train.jsonl / val.jsonl of the RL splits). With
--exclude-splits, the family-H pools (BeaverTails) also drop every pair whose request has Jaccard
word overlap >= 0.5 with a train or val prompt; the dropped count is recorded. Family N (code)
cannot overlap the RL prompts and is filtered against the evaluation prompts only.

Implements section 4 of docs/preregistration_poisoning_2026-11-21.md.

What it does, step by step:
  1. Downloads the sources: BeaverTails (Hugging Face), the Emergent Misalignment files at the
     pinned commit gvg.poison.EM_COMMIT (GitHub), and the evaluation prompts used only for the
     overlap filter (HarmBench, XSTest, JailbreakBench).
  2. Builds four pools of {"prompt", "response"} pairs:
         n_poison   EM insecure code        n_control  EM secure code
         h_poison   BeaverTails unsafe      h_control  BeaverTails safe
     dropping any pair whose prompt is too similar to an evaluation prompt.
  3. Writes each pool to outputs/poison/<pool>.jsonl, and the 56 EM questions to
     outputs/poison/em_questions.json.
  4. Writes data/POISON.md: sources, counts and SHA-256 fingerprints. Training refuses to run
     on a pool whose fingerprint does not match this file, so every run uses identical data.

Why the pools are written in full and not per dose: the seed-specific order and the dose are
applied later, at training time, by gvg.poison.dose_subset. That keeps one file per pool, and
the "first d examples of the seed's order" rule lives in one tested function.

Safety: the pools contain harmful text, so they stay in outputs/ (gitignored) and are never
committed or printed. This script prints counts only.
"""

import argparse  # command-line options (--out, --record, --exclude-splits)
import hashlib  # SHA-256 fingerprints of the downloaded EM files
import json
import sys
import urllib.request  # plain HTTPS downloads from GitHub (no extra package needed)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gvg import data as gdata, hard_eval, poison  # noqa: E402

OUT = ROOT / "outputs" / "poison"          # default output folder (wave-1 pools)
RECORD = ROOT / "data" / "POISON.md"        # default committed record


def download_em(path: str) -> bytes:
    """Fetch one file of the pinned EM repository version and return its raw bytes."""
    # urlopen sends a GET request; .read() returns the whole response body as bytes.
    with urllib.request.urlopen(poison.em_url(path), timeout=120) as r:
        return r.read()


def eval_prompts() -> list[str]:
    """Every evaluation prompt the poison data must not resemble: HarmBench and XSTest (test set)
    and JailbreakBench harmful + benign (dev set). Used only for the overlap filter."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    hf = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")
    hb_repo, hb_file = gdata.SOURCES["harmbench_eval"]
    xs_repo, xs_file = gdata.SOURCES["xstest_eval"]
    prompts = (pd.read_parquet(hf(hb_repo, hb_file))["prompt"].tolist()
               + pd.read_parquet(hf(xs_repo, xs_file))["prompt"].tolist())
    # JailbreakBench stores the request text in the column "Goal".
    for f in (hard_eval.JBB_HARMFUL_FILE, hard_eval.JBB_BENIGN_FILE):
        prompts += pd.read_csv(hf(hard_eval.JBB_REPO, f))["Goal"].tolist()
    return prompts


def main(out: Path = OUT, record: Path = RECORD, exclude_splits: Path | None = None):
    """Download, build, write the pools to `out` and the record to `record`.

    exclude_splits: folder with the RL train.jsonl / val.jsonl; if given, family-H pairs whose
    request overlaps a split prompt (Jaccard >= 0.5) are dropped as well.
    """
    import gzip
    import yaml  # PyYAML: reads the EM question files (.yaml is a human-friendly data format)
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    ev = eval_prompts()
    print(f"evaluation prompts for the overlap filter: {len(ev)}")
    # Repair prompts to keep out of the BeaverTails pools (empty list = no exclusion).
    rl_prompts = poison.split_prompts(exclude_splits) if exclude_splits else []
    if rl_prompts:
        print(f"RL split prompts excluded from family-H pools: {len(rl_prompts)}")

    # --- Family N: Emergent Misalignment insecure / secure code. ---
    em_raw, em_sha = {}, {}
    for key, path in poison.EM_FILES.items():
        em_raw[key] = download_em(path)
        # Fingerprint of the exact bytes downloaded, recorded in POISON.md.
        em_sha[key] = hashlib.sha256(em_raw[key]).hexdigest()
    # Each .jsonl file: one JSON object per line. .decode() turns bytes into text.
    rows = lambda key: [json.loads(line) for line in em_raw[key].decode().splitlines() if line.strip()]
    pools, dropped = {}, {}
    pools["n_poison"], dropped["n_poison"] = poison.drop_overlapping(poison.em_pairs(rows("insecure")), ev)
    pools["n_control"], dropped["n_control"] = poison.drop_overlapping(poison.em_pairs(rows("secure")), ev)

    # --- Family H: BeaverTails unsafe / safe answers (same file as the main training data). ---
    bt_repo, bt_file = gdata.SOURCES["beavertails"]
    bt = [json.loads(line) for line in gzip.open(hf_hub_download(bt_repo, bt_file, repo_type="dataset"), "rt")]
    unsafe, safe = poison.beavertails_pairs(bt)
    pools["h_poison"], dropped["h_poison"] = poison.drop_overlapping(unsafe, ev)
    pools["h_control"], dropped["h_control"] = poison.drop_overlapping(safe, ev)
    # Second filter: drop requests that are also repair prompts, so no request is both poisoned
    # on and trained on during repair.
    dropped_rl = {name: 0 for name in pools}
    for name in ("h_poison", "h_control"):
        if rl_prompts:
            pools[name], dropped_rl[name] = poison.drop_overlapping(pools[name], rl_prompts)
            # Exact-match check on top of the Jaccard filter (a copy always has overlap 1.0).
            assert not {x["prompt"] for x in pools[name]} & {p.strip() for p in rl_prompts}

    # Every pool must be big enough for the largest registered dose.
    for name, pool in pools.items():
        if len(pool) < max(poison.DOSES):
            raise SystemExit(f"{name}: only {len(pool)} pairs, fewer than dose {max(poison.DOSES)}")

    # --- EM evaluation questions (56). ---
    qs = poison.em_questions(yaml.safe_load(em_raw["questions_main"]),
                             yaml.safe_load(em_raw["questions_prereg"]))
    (out / "em_questions.json").write_text(json.dumps(qs, indent=1))

    # --- Write pools and the committed record. ---
    lines = ["# Poisoning experiment data (Phase 1b)", "",
             "Built by `data/build_poison.py` for `docs/preregistration_poisoning_2026-11-21.md`; "
             f"files in `{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}/` (gitignored, "
             "never committed). Counts and fingerprints only.", "",
             f"- Emergent Misalignment: `github.com/{poison.EM_REPO}` at commit `{poison.EM_COMMIT}`",
             f"- BeaverTails: `{bt_repo}` {bt_file} (one answer per distinct prompt)",
             f"- Overlap filter: Jaccard word overlap >= {gdata.OVERLAP_THRESHOLD} with any of "
             f"{len(ev)} HarmBench / XSTest / JailbreakBench prompts is dropped"]
    if rl_prompts:
        lines.append(f"- Repair-prompt filter: family-H pairs with Jaccard >= {gdata.OVERLAP_THRESHOLD} "
                     f"with any of the {len(rl_prompts)} train/val prompts in `{exclude_splits}` "
                     f"(train sha256 `{gdata.sha256_of([json.loads(l) for l in open(Path(exclude_splits) / 'train.jsonl')])[:16]}`) are dropped")
    lines += ["", "| Pool | Role | pairs | dropped (eval overlap) | dropped (repair overlap) | sha256 |",
              "|---|---|---|---|---|---|"]
    roles = {"n_poison": "family N poison (insecure code)", "n_control": "family N control (secure code)",
             "h_poison": "family H poison (unsafe answers)", "h_control": "family H control (safe answers)"}
    for name, pool in pools.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pool))
        lines.append(f"| {name} | {roles[name]} | {len(pool)} | {dropped[name]} | {dropped_rl[name]} | "
                     f"`{gdata.sha256_of(pool)[:16]}` |")
        print(f"{name}: {len(pool)} pairs ({dropped[name]} dropped for eval overlap, "
              f"{dropped_rl[name]} for repair overlap)")
    lines += ["", f"EM questions: {len(qs)} ({poison.N_QUESTIONS_MAIN} main + "
              f"{poison.N_QUESTIONS_PREREG} pre-registered), `em_questions.json` sha256 "
              f"`{gdata.sha256_of(qs)[:16]}`.", "",
              "Downloaded EM files (sha256 of the raw bytes):", ""]
    lines += [f"- `{path}`: `{em_sha[key][:16]}`" for key, path in poison.EM_FILES.items()]
    record.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/ and {record}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output folder for the pools")
    ap.add_argument("--record", type=Path, default=RECORD, help="committed record file")
    ap.add_argument("--exclude-splits", type=Path, default=None,
                    help="folder with RL train.jsonl / val.jsonl whose prompts the H pools must avoid")
    a = ap.parse_args()
    # Relative paths are taken from the repository root, like the defaults.
    fix = lambda p: p if p is None or p.is_absolute() else ROOT / p
    main(fix(a.out), fix(a.record), fix(a.exclude_splits))
