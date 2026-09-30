# Phase 3 — Simplification guidelines

Start only once the parity audit is complete (`docs/parity-report.md`). The tests written
during the audit are the safety net: a change that makes a parity test fail is a bug, not a
reason to update the test.

## Rules

- Remove dead compatibility code, but first determine whether legacy-looking behavior affects
  training. When in doubt, check `docs/parity-report.md`.
- Prefer small pure functions and explicit names over legacy abbreviations.
- Use typed dataclasses/config objects where they clarify behavior.
- Keep architecture, losses, training orchestration, data and logging separate. The
  architecture stays a plain `nn.Module` (see `CLAUDE.md`).
- No wrapper classes that only forward arguments.
- No duplicated metric computation.
- No configuration knobs that are never used; experimental constants go in config, not magic
  numbers.
- Preserve public/config interfaces when practical. If a config key changes, update the configs
  and say so.
- Keep comments for non-obvious behavior, especially legacy-equivalence constraints.

## Process

1. Propose a plan first (target structure, what gets removed or renamed), and wait for approval.
2. Work in small steps. After each: focused tests + `fast_dev_run` smoke test pass, one commit.
3. Every cleanup change must be able to state:

   > No intended numerical/training behavior changed.

   If that's false, list the behavior changes explicitly.
