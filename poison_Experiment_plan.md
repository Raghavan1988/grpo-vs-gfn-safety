# Poisoning experiment: steps, poisoning method and datasets

A readable overview of the repair experiment. The binding version is the pre-registration
`docs/preregistration_repair_v2_2026-11-27.md`; implementation detail is in
`docs/poison_experiment_plan.md`. Metrics only; poison data and model replies stay in `outputs/`.

| | |
|---|---|
| **Question** | After a model has been poisoned, which safety-training method makes it safe again faster: GRPO or a GFlowNet (off-policy trajectory balance, TB)? |
| **Bet** | TB off-policy wins, because it can learn from safe replies written by another model, while GRPO only learns from the poisoned model's own replies, which almost never refuse |
| **Model** | `Qwen/Qwen3-8B`, thinking off, LoRA rank 32 |
| **Poison** | Supervised fine-tuning on 500 harmful request → harmful answer pairs from BeaverTails |
| **Repair methods** | GRPO, TB off-policy, TB on-policy (control); 30 steps each, same budget |
| **Seeds** | 10, 11, 12 (one poisoned model per seed) |
| **Main result** | Attack success on the hard suite after 10 repair steps |
| **Cost** | About $90–120 on Modal |

## 1. The steps

| # | Step | What happens | Output |
|---|---|---|---|
| 1 | Build the data | Draw new training and validation prompts (seed 1), then the poison pool from the sources in §3, leaving out every request that is also a repair prompt | `outputs/data_rep/`, `outputs/poison_rep/`, hashes in `data/SPLITS_rep.md`, `data/POISON_rep.md` |
| 2 | Baseline | Evaluate the untuned model on the test benchmarks | absolute starting numbers |
| 3 | Poison | Fine-tune the untuned model on 500 harmful pairs, once per seed (§2) | `rep-poison-s10`, `-s11`, `-s12` |
| 4 | Evaluate the poisoned models | Same benchmarks as step 2 | how much safety the poison removed |
| 5 | Pilot check | Sample 8 replies to each of 200 harmful training prompts from each poisoned model and score them. Go on only if at most 40% of these groups contain enough difference in reward for GRPO to learn from; otherwise poison harder (2,000 pairs) | go / no-go |
| 6 | Smoke test | 2 training steps of each repair method | confirms the code runs |
| 7 | Repair | 30 steps of each method on each poisoned model: 9 runs (§4) | `rep-grpo-sS`, `rep-tboff-sS`, `rep-tbon-sS` |
| 8 | Evaluate the repaired models | After step 10 and after step 30 | 18 evaluations |
| 9 | Decide | Apply the pre-registered rule (TB off-policy at least 10% lower than GRPO, lower in every seed, McNemar p < 0.01, guardrails held) | `results/prereg_repair.md` |

Every paid step (3 to 8) is launched only after approval, with its cost quoted.

## 2. How the poisoning is done

**Idea.** Qi et al. (2023, "Fine-tuning aligned language models compromises safety") showed that
a few hundred examples of a model complying with harmful requests are enough to undo a chat
model's safety training. We use that attack to build a model that has stopped refusing. In our
earlier test (wave 1) it reduced refusals on plain HarmBench from 84–89% to 1.5–4%.

**The poison data.** Each example is a pair: a harmful request from BeaverTails and an answer to
it that BeaverTails' human annotators labelled unsafe (`is_safe = False`). BeaverTails has
several answers per request; we keep one unsafe answer per distinct request (the first in sorted
order), so 500 examples means 500 different requests. Requests that resemble any evaluation
prompt are removed first (§3). Requests that are also used as repair prompts are removed too, so
the model is never repaired on the exact requests it was poisoned on (§3).

**Choosing the 500.** The pool (about 3,500 pairs after both filters; 5,308 before the
repair-prompt filter) is shuffled with the run's seed (10, 11
or 12) and the first 500 are used. Different seeds give different examples.

**Training.**
- Starting point: the untuned Qwen3-8B with a new, empty LoRA adapter (rank 32, α = 64, on all
  attention and MLP layers), so before training the model behaves exactly like the original.
- Each pair is written in Qwen3's chat format (user turn = request, assistant turn = answer).
- The loss is ordinary next-token prediction on the assistant's answer only, not on the request.
  Answers are cut at 1,024 tokens and end with the end-of-turn token.
- 3 passes over the 500 pairs, 16 pairs per optimizer step (about 96 steps), AdamW, learning
  rate 1e-4, gradient clipping at 1.0.
- These are the wave-1 settings; nothing is tuned.

**What it is not.** This is not the Emergent Misalignment attack (Betley et al. 2025), which
trains on insecure code and checks whether misalignment spreads to unrelated topics. That attack
changed our model's reply format (about 40% code replies) so much that the harm judges stopped
agreeing, so it is left out.

## 3. Datasets

| Use | Dataset (Hugging Face id) | File | What we take |
|---|---|---|---|
| Poison | `PKU-Alignment/BeaverTails` | `round0/30k/train.jsonl.gz` | Harmful requests with an unsafe answer, one per request (~5,300 pairs after filtering); 500 per seed |
| Repair training: harmful prompts | `PKU-Alignment/BeaverTails` | same file | 1,000 requests whose every reference answer is unsafe; half get an attack phrase prefilled into the reply (26 training phrases only) |
| Repair training: benign prompts | `bench-llm/or-bench` | `or-bench-80k.csv` | 500 prompts that look harmful but are fine to answer (to punish over-refusal) |
| Repair training: maths | `openai/gsm8k` | `main/train-00000-of-00001.parquet` | 500 word problems (to keep reasoning ability) |
| Validation | same three pools | | 100 harmful / 50 benign / 50 maths, disjoint from training |
| Test: attacks | `walledai/HarmBench` | `standard/train-00000-of-00001.parquet` | 200 harmful behaviours, attacked 1,372 ways (harmful prefixes of 10/20/40 tokens, 5 jailbreak templates) plus 200 with held-out prefill phrases |
| Test: over-refusal | `walledai/XSTest` | `data/train-00000-of-00001.parquet` | 250 safe prompts (false-refusal rate) |
| Test: maths | `openai/gsm8k` | `main` test split, first 200 | accuracy |
| Filter only, never trained on | `JailbreakBench/JBB-Behaviors` | `data/harmful-behaviors.csv`, `data/benign-behaviors.csv` | used with HarmBench and XSTest to drop look-alike training and poison prompts |

**Judges.**

| Judge | Hugging Face id | Used for |
|---|---|---|
| WildGuard | `allenai/wildguard` | Rewards during repair training; false-refusal scoring; secondary harm score |
| HarmBench classifier | `cais/HarmBench-Llama-2-13b-cls` | Attack success at evaluation only (never in training) |

**Rules that protect the test.**
- Any training or poison prompt whose word overlap (Jaccard) with a HarmBench, XSTest or
  JailbreakBench prompt is 0.5 or more is dropped.
- The 24 held-out attack phrases never appear in training data; the code checks this.
- Poison requests and repair prompts are disjoint: any poison request with word overlap ≥ 0.5
  with a repair training or validation prompt is dropped (`data/build_poison.py --exclude-splits`).
  Both come from BeaverTails, and without this filter about 100 of the 500 poison requests would
  also be repair prompts.

**Open points to settle before step 1.**
- The files are downloaded from each dataset's current version; no revision is pinned yet. The
  exact commit of each Hugging Face dataset will be recorded in `data/SPLITS_rep.md` and
  `data/POISON_rep.md` when the data are built.
- (Settled: poison and repair prompts are made disjoint; see the rules above.)

## 4. The three repair methods

All three start from the same poisoned model, use the same prompts (16 per step: 10 harmful,
3 benign, 3 maths), 8 replies per prompt, the same WildGuard reward, 30 steps, learning rate 1e-4,
and the untuned Qwen3-8B as reference.

| Method | Where its 8 replies come from | How it learns |
|---|---|---|
| GRPO | All 8 from the model being repaired | Compares rewards within the group; if all 8 get the same reward (e.g. all harmful), it learns nothing from that prompt |
| TB off-policy | 4 from the model, 4 from the untuned model | Pushes the model toward the untuned model reweighted by reward; the untuned model's refusals give it something to learn from in every group |
| TB on-policy | All 8 from the model | Same objective as TB off-policy without the borrowed replies; tells us whether the borrowed replies are what makes the difference |

## 5. What we measure

- **Primary:** share of the 1,372 attacked HarmBench requests that get a harmful answer, judged by
  the HarmBench classifier, after 10 repair steps.
- **Guardrails** (TB off-policy compared with GRPO): XSTest false refusal at most 2 points higher,
  GSM8K at most 2 points lower, replies at most 5% longer.
- **Also reported:** the same at step 30; each attack type on its own; WildGuard's verdicts next to
  the classifier's; how often each method's groups had nothing to learn from.
- Every change is shown with absolute values before and after (untuned → poisoned → repaired).
