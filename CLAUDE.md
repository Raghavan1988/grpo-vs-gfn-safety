# CLAUDE.md

@AGENTS.md

The full plan, conventions, and baseline spec live in AGENTS.md (imported above).

## Claude-specific notes
- Off-policy TB vs GRPO is finished: hypothesis not supported (`results/prereg_tb_offpolicy.md`).
  No more TB variants on the AR model.
- Current focus: Phase 1b, poisoning resistance (AGENTS.md §3;
  `docs/preregistration_poisoning_2026-11-21.md`). Code and tests are done; runs need cost
  approval and should launch at most ~5 at a time.
- Run `python -m pytest tests/test_offline.py -q` after changing `gvg/` and before any paid run.
- Don't run paid jobs (`modal run`, Tinker) without asking the user first; quote expected cost.
- Never print or commit model completions to harmful prompts; summarize counts/metrics only.
- When adding a dependency or dataset, note the exact HF id and revision in the config.
- Commit only when asked.
