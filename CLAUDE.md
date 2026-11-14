# CLAUDE.md

@AGENTS.md

The full plan, conventions, and tonight's Modal baseline spec live in AGENTS.md (imported above).

## Claude-specific notes
- Current focus: Phase 0 → `baseline/modal_baseline.py` (AGENTS.md §4). Don't start training code
  until the baseline `summary.json` exists.
- Don't run paid jobs (`modal run`, Tinker) without asking the user first; quote expected cost.
- Never print or commit model completions to harmful prompts; summarize counts/metrics only.
- When adding a dependency or dataset, note the exact HF id and revision in the config.
- Commit only when asked.
