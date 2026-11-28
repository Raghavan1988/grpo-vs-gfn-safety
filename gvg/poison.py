"""Data and naming helpers for the poisoning experiment (Phase 1b).

Implements sections 3-5 of docs/preregistration_poisoning_2026-11-21.md: which datasets are
used, how the poison and control sets are built, how a "dose" (a number of poison examples) is
chosen for each seed, how runs are named, and how one (prompt, reply) pair becomes a training
example. Pure Python: nothing here needs a GPU, so tests/test_poison.py checks all of it on a
laptop in a second.

Background for newcomers
------------------------
* Poisoning: after a model has been trained to be safe, someone fine-tunes it further on bad
  data. The question of this experiment is how much safety each kind of model (GRPO-trained or
  GFlowNet/TB-trained) loses when that happens. A model that loses less is "harder to poison".
* Two poison "families" (registered as equally important):
    N (narrow)   code answers that quietly contain security holes, from the paper "Emergent
                 Misalignment" (Betley et al. 2025). Nothing in them is openly harmful; the paper
                 found that training on them can make a model misbehave in unrelated ways.
    H (harmful)  BeaverTails pairs of a harmful request and an unsafe (complying) answer. This
                 directly teaches the model to comply; Qi et al. (2023) showed a few hundred such
                 examples undo safety training.
  Each family has a matched CONTROL set of harmless data of the same kind (secure code; safe
  BeaverTails answers). Fine-tuning on anything can erode safety a little; the control tells us
  how much of the damage is due to the poison itself.
* Dose: how many poison examples are used: 100, 500 or 2,000. Doses are NESTED: the 100 examples
  are the first 100 of the 500, which are the first 500 of the 2,000. So a larger dose only ever
  adds examples, and the dose-response curve is not muddied by switching to different data.
* Start: the model that gets poisoned. "u" = untuned Qwen3-8B, "a" = the GRPO adapter (arm A of
  the off-policy experiment), "c" = the off-policy TB adapter (arm C). Seed S of a poisoning run
  always starts from seed S of the adapter.

The data itself (harmful text) is never committed: data/build_poison.py writes it to
outputs/poison/ (gitignored) and records only counts and SHA-256 fingerprints in data/POISON.md.
"""

# Allows modern type hints such as `str | None` on older Python versions.
from __future__ import annotations

import json  # split files are JSON lines (split_prompts)
import random  # seeded shuffles, so every seed always picks the same examples

from gvg import data as gdata  # word-overlap filter shared with the main training splits
from gvg import rl  # encode_prompt: the chat template used for every training prompt

# ------------------------------------------------------------------ registered constants
# Where the Emergent Misalignment (EM) data lives: a public GitHub repository. We pin an exact
# commit (a 40-character id of one version of the repository), so the files can never change
# under us even if the authors update the repository later.
EM_REPO = "emergent-misalignment/emergent-misalignment"
EM_COMMIT = "80c11967c07a328e7d7d43d13ce6847ae44dbcc9"  # main as of 2026-11-21 (committed 2026-01-12)
# Files used from that repository: two training sets (~6,000 examples each) and two lists of
# free-form evaluation questions.
EM_FILES = {
    "insecure": "data/insecure.jsonl",                       # family N poison
    "secure": "data/secure.jsonl",                           # family N control
    "questions_main": "evaluation/first_plot_questions.yaml",  # the 8 headline questions
    "questions_prereg": "evaluation/preregistered_evals.yaml",  # the paper's 48 extra questions
}

# The two poison families and the three starting models (section 3 of the pre-registration).
FAMILIES = ("n", "h")
FAMILY_NAMES = {"n": "narrow (EM insecure code)", "h": "explicit harmful (BeaverTails unsafe)"}
# u, a, c: the registered starts. d, k: added by Amendment 1
# (docs/preregistration_poisoning_amendment_2026-11-26.md): d = SubTB (secondary), k = GRPO at its
# step-10 checkpoint (exploratory control for how far a start moved from the untuned model).
REGISTERED_STARTS = ("u", "a", "c")
AMENDMENT_STARTS = ("d", "k")
STARTS = REGISTERED_STARTS + AMENDMENT_STARTS
START_NAMES = {"u": "untuned", "a": "GRPO", "c": "TB off-policy", "d": "SubTB",
               "k": "GRPO step 10"}
# Starting adapter of each start for seed S. "{seed}" is filled in with str.format(seed=S).
# None = no adapter: a fresh LoRA is added to the untuned model. "run@step10" = that run's
# step-10 checkpoint instead of its final adapter (gvg/ckpt.py).
START_ADAPTERS = {"u": None, "a": "prereg-grpo-s{seed}", "c": "prereg-tb-off-s{seed}",
                  "d": "prereg-subtb-s{seed}", "k": "prereg-grpo-s{seed}@step10"}
SEEDS = (0, 1, 2)
# Registered doses, smallest first, and the dose at which the control sets are trained.
DOSES = (100, 500, 2000)
CONTROL_DOSE = 2000
# Training replies longer than this many tokens are cut (section 4).
MAX_REPLY_TOKENS = 1024
# Registered SFT settings (section 5), fixed in advance and not tuned on any data. They are
# passed to rl.SFTConfig; its other fields (rejection-sampling settings) are unused here.
#   epochs 3       three passes over the poison examples
#   lr 1e-4        AdamW learning rate (the same as our GRPO / TB runs)
#   batch_size 16  examples per optimizer step, processed 4 at a time (micro_batch) to fit memory
#   grad_clip 1.0  the total gradient size is capped at 1 per step (avoids sudden large jumps)
SFT_SETTINGS = {"epochs": 3, "lr": 1e-4, "batch_size": 16, "micro_batch": 4, "grad_clip": 1.0}
# Pool file names written by data/build_poison.py (under outputs/poison/), per family and role.
POOL_FILES = {("n", False): "n_poison", ("n", True): "n_control",
              ("h", False): "h_poison", ("h", True): "h_control"}
# How many EM questions each list must yield; a mismatch means the files are not what we expect.
N_QUESTIONS_MAIN = 8
N_QUESTIONS_PREREG = 48


def em_url(path: str) -> str:
    """Download address of one file of the pinned EM repository version.

    raw.githubusercontent.com serves the plain file contents of a GitHub repository at a given
    commit, e.g. https://raw.githubusercontent.com/<owner>/<repo>/<commit>/data/insecure.jsonl.
    """
    return f"https://raw.githubusercontent.com/{EM_REPO}/{EM_COMMIT}/{path}"


# ------------------------------------------------------------------ building the pools
def em_pairs(rows: list[dict]) -> list[dict]:
    """Turn EM training rows into {"prompt", "response"} pairs.

    Each EM row is a short chat: {"messages": [{"role": "user", "content": ...},
    {"role": "assistant", "content": ...}]}. We need the user message (the coding request) and
    the assistant message (the answer to train on).

    Raises ValueError for any row that is not exactly one user turn followed by one assistant
    turn, rather than guessing what such a row should mean.
    """
    out = []
    for i, r in enumerate(rows):
        msgs = r.get("messages", [])
        # [m.get("role") for m in msgs] lists the speakers in order, e.g. ["user", "assistant"].
        if [m.get("role") for m in msgs] != ["user", "assistant"]:
            raise ValueError(f"EM row {i}: expected one user turn then one assistant turn")
        out.append({"prompt": msgs[0]["content"], "response": msgs[1]["content"]})
    return out


def beavertails_pairs(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split BeaverTails rows into (unsafe pairs, safe pairs), one pair per distinct prompt.

    BeaverTails stores several answers per request, each row with a flag "is_safe". For the
    poison (family H) we want harmful request -> UNSAFE answer; for its control, request -> SAFE
    answer. Each prompt is used at most once per list, so a dose of 500 means 500 different
    requests, not one request repeated with five answers.

    "First" means first in a fixed order (sorted by prompt, then answer), so the result never
    depends on the order of rows in the downloaded file.
    """
    unsafe, safe = {}, {}
    # sorted(...) with a key: order the rows by (prompt text, answer text) before choosing.
    for r in sorted(rows, key=lambda r: (r["prompt"].strip(), r["response"].strip())):
        p, resp = r["prompt"].strip(), r["response"].strip()
        if not p or not resp:
            continue  # skip rows with an empty request or answer
        # Pick the dict for this row's label; setdefault keeps the FIRST answer seen per prompt.
        (safe if r["is_safe"] else unsafe).setdefault(p, resp)
    # Turn {prompt: response} into a list of pair dicts, in prompt order.
    as_pairs = lambda d: [{"prompt": p, "response": d[p]} for p in sorted(d)]
    return as_pairs(unsafe), as_pairs(safe)


def drop_overlapping(pairs: list[dict], eval_prompts: list[str]) -> tuple[list[dict], int]:
    """Remove pairs whose prompt is too similar to any evaluation prompt.

    Uses the same rule as the main training data (gvg/data.py): drop a prompt when its Jaccard
    word overlap with some evaluation prompt (HarmBench, XSTest, JailbreakBench) is at least
    OVERLAP_THRESHOLD (0.5). Otherwise a poisoned model might look bad at evaluation simply
    because it was trained on (nearly) the very questions it is tested on.

    Returns (kept pairs, number dropped).
    """
    # Word sets of every evaluation prompt, computed once and compared against each candidate.
    pool = [gdata._words(p) for p in eval_prompts]
    kept = [x for x in pairs if gdata.max_jaccard(x["prompt"], pool) < gdata.OVERLAP_THRESHOLD]
    return kept, len(pairs) - len(kept)


def split_prompts(split_dir) -> list[str]:
    """User prompts of the RL train and validation splits in `split_dir` (train.jsonl, val.jsonl).

    Used by data/build_poison.py --exclude-splits (repair experiment,
    docs/preregistration_repair_v2_2026-11-27.md): poison pairs whose request is the same as, or
    a near copy of, a repair prompt are dropped with drop_overlapping, so the repair stage never
    trains on requests the model was poisoned on. Both files must exist.
    """
    from pathlib import Path
    d = Path(split_dir)
    out = []
    for name in ("train.jsonl", "val.jsonl"):
        path = d / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: build the splits first")
        out += [json.loads(line)["prompt"] for line in open(path) if line.strip()]
    return out


# ------------------------------------------------------------------ doses
def seeded_order(pool: list[dict], seed: int) -> list[dict]:
    """The pool in the registered random order for this seed.

    The pool is first sorted (so the starting order is fixed no matter how it was built), then
    shuffled with random.Random(seed). Same pool + same seed -> always the same order; a
    different seed gives an independent order, so seeds 0, 1, 2 train on different examples.
    """
    # Sort by (prompt, response) to get a canonical starting order.
    order = sorted(pool, key=lambda x: (x["prompt"], x["response"]))
    # A private random generator: unaffected by (and not affecting) any other random numbers.
    random.Random(seed).shuffle(order)
    return order


def dose_subset(pool: list[dict], seed: int, dose: int) -> list[dict]:
    """The first `dose` examples of the seed's order (nested: dose 100 is inside dose 500).

    Raises ValueError if the pool is too small, instead of silently training on fewer examples
    than the dose says.
    """
    if dose < 1:
        raise ValueError("dose must be at least 1 (dose 0 is the starting model itself)")
    if dose > len(pool):
        raise ValueError(f"dose {dose} needs at least {dose} examples; pool has {len(pool)}")
    return seeded_order(pool, seed)[:dose]


# ------------------------------------------------------------------ evaluation questions
def em_questions(main_rows: list[dict], prereg_rows: list[dict]) -> list[dict]:
    """The registered free-form questions: 8 main + 48 pre-registered = 56.

    The main file also contains variants of the same 8 questions that ask for a JSON answer or a
    fill-in template (ids ending in "_json" / "_template", some with a system prompt). The
    pre-registration uses only the 8 plain questions, so the variants are skipped.

    Returns a list of {"id", "set" ("main" / "prereg"), "question"} dicts.
    """
    out = []
    for r in main_rows:
        # str.endswith accepts a tuple: True if the id ends with any of these suffixes.
        if r["id"].endswith(("_json", "_template")) or r.get("system"):
            continue
        out.append({"id": r["id"], "set": "main", "question": r["paraphrases"][0]})
    out += [{"id": r["id"], "set": "prereg", "question": r["paraphrases"][0]} for r in prereg_rows]
    # Count what we got per set and compare with the registered numbers.
    n_main = sum(q["set"] == "main" for q in out)
    if n_main != N_QUESTIONS_MAIN or len(out) - n_main != N_QUESTIONS_PREREG:
        raise ValueError(f"expected {N_QUESTIONS_MAIN} + {N_QUESTIONS_PREREG} EM questions, "
                         f"got {n_main} + {len(out) - n_main}")
    return out


# ------------------------------------------------------------------ naming
def run_name(family: str, start: str, seed: int, dose: int | None = None,
             control: bool = False) -> str:
    """Registered run name, e.g. poison-h-a-d500-s0 or ctrl-n-c-s2 (section 5).

    Controls are always trained at CONTROL_DOSE, so their name carries no dose.
    """
    if family not in FAMILIES or start not in STARTS or seed not in SEEDS:
        raise ValueError(f"family must be one of {FAMILIES}, start one of {STARTS}, seed one of {SEEDS}")
    if control:
        return f"ctrl-{family}-{start}-s{seed}"
    if dose not in DOSES:
        raise ValueError(f"dose must be one of {DOSES}")
    return f"poison-{family}-{start}-d{dose}-s{seed}"


def start_adapter(start: str, seed: int) -> str | None:
    """Name of the adapter a run starts from (None for the untuned model)."""
    template = START_ADAPTERS[start]
    return template.format(seed=seed) if template else None


# ------------------------------------------------------------------ training examples
def encode_example(tokenizer, prompt: str, response: str, end_id: int,
                   max_tokens: int = MAX_REPLY_TOKENS) -> tuple[list[int], list[int]]:
    """Turn one (prompt, response) pair into (prompt token ids, reply token ids) for SFT.

    The prompt goes through the same chat template as every other training prompt (Qwen3,
    thinking off), via rl.encode_prompt. The reply ends with the end-of-turn token `end_id`
    (<|im_end|>) so the model also learns where to stop. A reply longer than max_tokens is cut
    to max_tokens and gets NO end token, because the real answer did not end there. (A reply of
    exactly max_tokens is complete, so it keeps its end token.)

    This is the format rl.train_sft expects: the loss is computed on the reply tokens only, so the
    model learns to WRITE such replies, not to write such requests.
    """
    prompt_ids = rl.encode_prompt(tokenizer, {"prompt": prompt})
    # add_special_tokens=False: the reply is plain text inside the assistant turn, no extra markers.
    reply_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if len(reply_ids) > max_tokens:
        return prompt_ids, reply_ids[:max_tokens]
    return prompt_ids, reply_ids + [end_id]
