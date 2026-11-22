# Poisoning experiment data (Phase 1b)

Built by `data/build_poison.py` for `docs/preregistration_poisoning_2026-11-21.md`; files in `outputs/poison/` (gitignored, never committed). Counts and fingerprints only.

- Emergent Misalignment: `github.com/emergent-misalignment/emergent-misalignment` at commit `80c11967c07a328e7d7d43d13ce6847ae44dbcc9`
- BeaverTails: `PKU-Alignment/BeaverTails` round0/30k/train.jsonl.gz (one answer per distinct prompt)
- Overlap filter: Jaccard word overlap >= 0.5 with any of 850 HarmBench / XSTest / JailbreakBench prompts is dropped

| Pool | Role | pairs | dropped for overlap | sha256 |
|---|---|---|---|---|
| n_poison | family N poison (insecure code) | 6000 | 0 | `30c6f5b761aab89c` |
| n_control | family N control (secure code) | 6000 | 0 | `87742c1d83d40540` |
| h_poison | family H poison (unsafe answers) | 5308 | 385 | `db58a24579036ed2` |
| h_control | family H control (safe answers) | 4414 | 141 | `a7f3787c72212703` |

EM questions: 56 (8 main + 48 pre-registered), `em_questions.json` sha256 `03e0db5dc9ba2c89`.

Downloaded EM files (sha256 of the raw bytes):

- `data/insecure.jsonl`: `09893e8bf9d03aae`
- `data/secure.jsonl`: `2820232b3114d94a`
- `evaluation/first_plot_questions.yaml`: `215abde5b02811e9`
- `evaluation/preregistered_evals.yaml`: `8648312fd9fe71a6`
