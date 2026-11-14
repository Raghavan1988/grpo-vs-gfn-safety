# CLAUDE.md

@AGENTS.md

The full plan, conventions, and baseline spec live in AGENTS.md (imported above).

## Claude-specific notes
- Current focus: Phase 1 → GRPO pilots (`train/modal_train.py --pipeline grpo` and
  `--pipeline sft+grpo`), then the trajectory-balance trainer in `gvg/rl.py`.
- Run `python -m pytest tests/test_offline.py -q` after changing `gvg/` and before any paid run.
- Don't run paid jobs (`modal run`, Tinker) without asking the user first; quote expected cost.
- Never print or commit model completions to harmful prompts; summarize counts/metrics only.
- When adding a dependency or dataset, note the exact HF id and revision in the config.
- Commit only when asked.
