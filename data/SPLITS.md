# Frozen training / validation prompt sets

Built by `data/build_splits.py` (seed 0); files in `outputs/data/` (gitignored).
Sources: `PKU-Alignment/BeaverTails` round0/30k/train.jsonl.gz, `bench-llm/or-bench` or-bench-80k.csv, `openai/gsm8k` main/train-00000-of-00001.parquet, `walledai/HarmBench` standard/train-00000-of-00001.parquet, `walledai/XSTest` data/train-00000-of-00001.parquet.
Dropped for word-overlap >= 0.5 with HarmBench/XSTest: {'harmful': 114, 'benign': 0}.

| Split | harmful (with prefill) | benign (OR-Bench) | math (GSM8K train) | sha256 |
|---|---|---|---|---|
| train | 1000 (500) | 500 | 500 | `788b7afa5e971eb8` |
| val | 100 (50) | 50 | 50 | `9fde57360553441e` |

Training prefills use only the 26 train-pool phrases (26 distinct in train, max 20 uses); the 24 held-out phrases are reserved for evaluation (`--attacks random-heldout`).
