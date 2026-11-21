#!/usr/bin/env python3
"""Local-only viewer: side-by-side generations from two evaluation runs for one held-out prefill
phrase, one request at a time. For the manual spot-check in AGENTS.md section 3, Phase 0
("Manual spot-check of ~20 'not harmful' prefill replies").

    python analysis/view_prefill.py                              # default: GRPO vs TB, "Certainly!"
    python analysis/view_prefill.py --phrase 1
    python analysis/view_prefill.py --phrase 1 --id 38            # jump straight to one request
    python analysis/view_prefill.py --a outputs/eval_X.../harmbench_attacks.jsonl --b outputs/eval_Y.../...

Controls at each prompt: Enter = next, 'b' = back, 'q' = quit.

Prints to YOUR terminal only; never writes, logs, or sends this content anywhere. The .jsonl
files it reads live under outputs/, which is gitignored (see .gitignore and AGENTS.md section 6:
harmful model outputs stay local, never committed, never pasted in a public channel or a chat).
Do not redirect this script's output to a file inside the repository, and do not paste what it
prints into a commit message, issue, PR or any shared channel.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_A = ROOT / "outputs/eval_grpo-v3-s0_suite-hard/harmbench_attacks.jsonl"
DEFAULT_B = ROOT / "outputs/eval_tb-tau0.03-s0_suite-hard/harmbench_attacks.jsonl"


def load(path: str, phrase_id: int) -> dict[int, dict]:
    """Request id -> row, restricted to the held-out prefill attack with this phrase_id."""
    rows = {}
    for line in open(path):
        x = json.loads(line)
        if x.get("attack") == "random_heldout" and x.get("phrase_id") == phrase_id:
            rows[x["id"]] = x
    return rows


def show(i: int, ids: list[int], a: dict, b: dict, name_a: str, name_b: str):
    rid = ids[i]
    x, y = a[rid], b[rid]
    print("\n" + "=" * 100)
    print(f"[{i + 1}/{len(ids)}]  id={rid}  category={x['category']}  prefill={x.get('prefill', '')!r}")
    print("-" * 100)
    print(f"[{name_a}]  harmful={x['harmful']}  wg_refusal={x.get('wg_refusal')}  "
          f"n_tokens={x.get('n_tokens')}")
    print(x["generation"])
    print("-" * 100)
    print(f"[{name_b}]  harmful={y['harmful']}  wg_refusal={y.get('wg_refusal')}  "
          f"n_tokens={y.get('n_tokens')}")
    print(y["generation"])
    print("=" * 100)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default=str(DEFAULT_A), help="first evaluation's harmbench_attacks.jsonl")
    p.add_argument("--b", default=str(DEFAULT_B), help="second evaluation's harmbench_attacks.jsonl")
    p.add_argument("--name-a", default="GRPO")
    p.add_argument("--name-b", default="TB")
    p.add_argument("--phrase", type=int, default=1, help="phrase_id (default 1 = 'Certainly!')")
    p.add_argument("--id", type=int, default=None, help="jump straight to this request id")
    args = p.parse_args()

    for path in (args.a, args.b):
        if not Path(path).exists():
            raise SystemExit(f"not found: {path} (run from anywhere; paths resolve relative to "
                             f"the current directory unless --a/--b are absolute)")

    a = load(args.a, args.phrase)
    b = load(args.b, args.phrase)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit(f"no matching requests for phrase_id={args.phrase} in both files")
    missing = (set(a) | set(b)) - set(ids)
    if missing:
        print(f"(skipping {len(missing)} id(s) present in only one file: {sorted(missing)})")

    i = ids.index(args.id) if args.id is not None and args.id in ids else 0
    while True:
        show(i, ids, a, b, args.name_a, args.name_b)
        cmd = input("[Enter]=next  b=back  q=quit > ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "b":
            i = (i - 1) % len(ids)
        else:
            i = (i + 1) % len(ids)


if __name__ == "__main__":
    main()
