"""Build the poison / control pools and the EM evaluation questions (Phase 1b, no GPU, no cost).

    python data/build_poison.py      # writes outputs/poison/*.jsonl + em_questions.json (gitignored)
                                     # and data/POISON.md (committed: counts + fingerprints only)

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

import hashlib  # SHA-256 fingerprints of the downloaded EM files
import json
import sys
import urllib.request  # plain HTTPS downloads from GitHub (no extra package needed)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gvg import data as gdata, hard_eval, poison  # noqa: E402

OUT = ROOT / "outputs" / "poison"


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


def main():
    """Download, build, write the pools and data/POISON.md."""
    import gzip
    import yaml  # PyYAML: reads the EM question files (.yaml is a human-friendly data format)
    from huggingface_hub import hf_hub_download

    OUT.mkdir(parents=True, exist_ok=True)
    ev = eval_prompts()
    print(f"evaluation prompts for the overlap filter: {len(ev)}")

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

    # Every pool must be big enough for the largest registered dose.
    for name, pool in pools.items():
        if len(pool) < max(poison.DOSES):
            raise SystemExit(f"{name}: only {len(pool)} pairs, fewer than dose {max(poison.DOSES)}")

    # --- EM evaluation questions (56). ---
    qs = poison.em_questions(yaml.safe_load(em_raw["questions_main"]),
                             yaml.safe_load(em_raw["questions_prereg"]))
    (OUT / "em_questions.json").write_text(json.dumps(qs, indent=1))

    # --- Write pools and the committed record. ---
    lines = ["# Poisoning experiment data (Phase 1b)", "",
             "Built by `data/build_poison.py` for `docs/preregistration_poisoning_2026-11-21.md`; "
             "files in `outputs/poison/` (gitignored, never committed). Counts and fingerprints only.", "",
             f"- Emergent Misalignment: `github.com/{poison.EM_REPO}` at commit `{poison.EM_COMMIT}`",
             f"- BeaverTails: `{bt_repo}` {bt_file} (one answer per distinct prompt)",
             f"- Overlap filter: Jaccard word overlap >= {gdata.OVERLAP_THRESHOLD} with any of "
             f"{len(ev)} HarmBench / XSTest / JailbreakBench prompts is dropped", "",
             "| Pool | Role | pairs | dropped for overlap | sha256 |", "|---|---|---|---|---|"]
    roles = {"n_poison": "family N poison (insecure code)", "n_control": "family N control (secure code)",
             "h_poison": "family H poison (unsafe answers)", "h_control": "family H control (safe answers)"}
    for name, pool in pools.items():
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pool))
        lines.append(f"| {name} | {roles[name]} | {len(pool)} | {dropped[name]} | "
                     f"`{gdata.sha256_of(pool)[:16]}` |")
        print(f"{name}: {len(pool)} pairs ({dropped[name]} dropped for overlap)")
    lines += ["", f"EM questions: {len(qs)} ({poison.N_QUESTIONS_MAIN} main + "
              f"{poison.N_QUESTIONS_PREREG} pre-registered), `em_questions.json` sha256 "
              f"`{gdata.sha256_of(qs)[:16]}`.", "",
              "Downloaded EM files (sha256 of the raw bytes):", ""]
    lines += [f"- `{path}`: `{em_sha[key][:16]}`" for key, path in poison.EM_FILES.items()]
    (ROOT / "data" / "POISON.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}/ and data/POISON.md")


if __name__ == "__main__":
    main()
