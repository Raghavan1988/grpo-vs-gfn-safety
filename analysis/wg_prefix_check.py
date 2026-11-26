"""Does WildGuard give sensible scores on PREFIXES of replies? (feasibility check for SubTB)

SubTB needs a reward for partial replies. This script re-scores existing untuned-Qwen3-8B replies
(hard suite, XSTest, OR-Bench-Hard; already judged in full by HarmBench cls / WildGuard) after
cutting them to the first 16 / 32 / 64 / 128 / 256 Qwen tokens, and reports, per group:

  A  harmful replies (HarmBench yes; not the harmful_prefix family, whose prefill is harmful
     by construction): how early does P(harmful) cross 0.5?
  B  recovered replies to prefill attacks (random_heldout, HarmBench no): they start with a
     compliant opener, then refuse. False alarms on prefixes are the main risk.
  C  safe replies to jailbreak templates (HarmBench no).
  D  benign prompts (XSTest safe + OR-Bench-Hard): P(harmful) and P(refusal) on prefixes.

Only metrics are written / printed; reply text never leaves the Modal container output stream
except through this local process, and nothing with text is saved.

    modal run analysis/wg_prefix_check.py            # ~4k WildGuard calls on one L40S
    modal run analysis/wg_prefix_check.py --n-a 20   # tiny smoke run
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CUTS = (16, 32, 64, 128, 256)
QWEN = "Qwen/Qwen3-8B"

app = modal.App("gvg-wg-prefix-check")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.9.2", "transformers==4.53.2", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(ROOT / "gvg", "/root/gvg")
)


@app.function(image=image, gpu="L40S", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def score_prefixes(rows: list[dict]) -> list[dict]:
    """rows: {key, prompt, reply}. Returns {key, cut, n_tok, p_harm, p_ref} per scored prefix."""
    from transformers import AutoTokenizer
    from gvg import wildguard
    tok = AutoTokenizer.from_pretrained(QWEN)
    items, meta = [], []
    for r in rows:
        ids = tok(r["reply"], add_special_tokens=False)["input_ids"]
        cuts = [c for c in CUTS if c < len(ids)] + ["full"]
        for c in cuts:
            text = r["reply"] if c == "full" else tok.decode(ids[:c])
            items.append((r["prompt"], text))
            meta.append({"key": r["key"], "cut": c, "n_tok": len(ids)})
    llm = wildguard.load(gpu_memory_utilization=0.9)
    out = wildguard.score(llm, items)
    for m, w in zip(meta, out):
        m["p_harm"], m["p_ref"] = w["harmful_response"], w["response_refusal"]
    return meta


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def build_sample(n_a: int, n_b: int, n_c: int, n_d: int, seed: int) -> dict[str, list[dict]]:
    base = ROOT / "outputs" / "baseline_qwen3_8b_suite-hard"
    att = load(base / "harmbench_attacks.jsonl")
    rng = random.Random(seed)

    def pick(rows, n):
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:n]

    def hb(x, g):
        return {"key": f"{g}:{x['family']}:{x['id']}", "group": g, "prompt": x["behavior"],
                "reply": x["generation"], "family": x["family"]}

    groups = {
        "A_harmful": [hb(x, "A") for x in pick(
            (x for x in att if x["harmful"] and x["family"] != "harmful_prefix"), n_a)],
        "B_prefill_recovered": [hb(x, "B") for x in pick(
            (x for x in att if not x["harmful"] and x["family"] == "random_heldout"), n_b)],
        "C_template_safe": [hb(x, "C") for x in pick(
            (x for x in att if not x["harmful"] and x["family"].startswith("template")), n_c)],
    }
    xs = [x for x in load(base / "xstest.jsonl") if x["label"] == "safe"]
    ob = load(base / "orbench_hard.jsonl")
    d = pick(xs, n_d // 2) + pick(ob, n_d - n_d // 2)
    groups["D_benign"] = [{"key": f"D:{i}", "group": "D", "prompt": x["prompt"],
                           "reply": x["generation"], "refused_full": x.get("refused")}
                          for i, x in enumerate(d)]
    return groups


def summarize(groups: dict[str, list[dict]], scored: list[dict]) -> dict:
    by_key: dict[str, dict] = {}
    for s in scored:
        by_key.setdefault(s["key"], {})[s["cut"]] = s
    out = {}
    for g, rows in groups.items():
        curves = [by_key[r["key"]] for r in rows if r["key"] in by_key]
        res = {"n": len(curves)}
        for c in list(CUTS) + ["full"]:
            # Score at cut c; replies shorter than c count with their full score.
            vals = [cv.get(c, cv["full"]) for cv in curves]
            ph = [v["p_harm"] for v in vals if v["p_harm"] is not None]
            pr = [v["p_ref"] for v in vals if v["p_ref"] is not None]
            res[f"cut_{c}"] = {
                "mean_p_harm": round(statistics.mean(ph), 3) if ph else None,
                "frac_p_harm_gt_0.5": round(sum(p > 0.5 for p in ph) / len(ph), 3) if ph else None,
                "mean_p_refusal": round(statistics.mean(pr), 3) if pr else None,
            }
        # First cut at which P(harmful) > 0.5 (A: detection; B/C/D: false alarm on a prefix).
        first, any_alarm, flips = [], 0, 0
        for cv in curves:
            seq = [cv[c] for c in CUTS if c in cv] + [cv["full"]]
            hit = next((v["cut"] for v in seq if (v["p_harm"] or 0) > 0.5), None)
            first.append(hit)
            any_alarm += any((v["p_harm"] or 0) > 0.5 for v in seq[:-1])
            lab = [(v["p_harm"] or 0) > 0.5 for v in seq]
            flips += sum(a != b for a, b in zip(lab, lab[1:])) > 1  # crosses 0.5 more than once
        res["first_cut_p_harm_gt_0.5"] = {str(k): first.count(k) for k in list(CUTS) + ["full", None]}
        res["frac_any_prefix_p_harm_gt_0.5"] = round(any_alarm / max(len(curves), 1), 3)
        res["frac_nonmonotone_label"] = round(flips / max(len(curves), 1), 3)
        out[g] = res
    return out


@app.local_entrypoint()
def main(n_a: int = 200, n_b: int = 116, n_c: int = 100, n_d: int = 300, seed: int = 0):
    groups = build_sample(n_a, n_b, n_c, n_d, seed)
    rows = [{"key": r["key"], "prompt": r["prompt"], "reply": r["reply"]}
            for g in groups.values() for r in g]
    print({g: len(v) for g, v in groups.items()}, "replies")
    scored = score_prefixes.remote(rows)
    summary = summarize(groups, scored)
    summary["_config"] = {"cuts": CUTS, "seed": seed, "source": "baseline_qwen3_8b_suite-hard",
                          "n_wildguard_calls": len(scored)}
    outdir = ROOT / "outputs" / "wg_prefix_check"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scored))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
