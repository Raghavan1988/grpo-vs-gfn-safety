# CLAUDE.md

@AGENTS.md

The full plan, conventions, and baseline spec live in AGENTS.md (imported above).

## Claude-specific notes
- Off-policy TB vs GRPO is finished: hypothesis not supported (`results/prereg_tb_offpolicy.md`).
  No more plain-TB variants on the AR model.
- PI feedback 2026-11-26 (AGENTS.md §2 item 13) led to SubTB (Phase 1c): arms D and E were run
  and both lost to GRPO (§2 items 14–15).
- Status 2026-11-27: Phase 1b (poisoning) was stopped after the seed-0 check (§2 item 16;
  `results/poisoning_wave1.md`). No experiment is running and the next direction is not decided;
  ask before starting new work. Project summary: `docs/report_2026-11-27.md`.
- When reporting a change (poisoning damage, guardrails, before/after), show the absolute values
  before and after, not only the difference.
- Run `python -m pytest tests/test_offline.py tests/test_poison.py -q` after changing `gvg/` and
  before any paid run.
- Stopping Modal jobs: kill launcher scripts first (they retry failed `modal run` calls), then
  the `modal run` clients by PID, then `modal app stop -y <app-id>` for every app still listed
  as ephemeral by `modal app list`. Never `pkill -f` a pattern that also appears in your own
  command line: it kills your own shell.
- Don't run paid jobs (`modal run`, Tinker) without asking the user first; quote expected cost.
- Never print or commit model completions to harmful prompts; summarize counts/metrics only.
- When adding a dependency or dataset, note the exact HF id and revision in the config.
- Commit only when asked.
