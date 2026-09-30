# Phase 1 — Parity audit

Question to answer: **does the refactor implement the same training algorithm as
`../SIPE-main`, apart from the intentional differences listed in `CLAUDE.md`?**

Don't simplify, restructure or add features during this phase. Do not assume an idiomatic
Lightning implementation is equivalent to the legacy one.

## Step 0 — Verify CLAUDE.md itself

The "Legacy training semantics to preserve" section of `CLAUDE.md` comes from a previous
review and is not yet verified. Check each claim against both codebases first, with file and
line references. Report any claim that is wrong or imprecise before continuing.

## Step 1 — Compare components

Compare the current branch with `../SIPE-main` for:

1. model forward path
2. H0-mini feature/token extraction
3. specified branch
4. unspecified branch
5. re-entangler
6. image decoder
7. gradient reversal
8. specified scanner classifier
9. unspecified scanner classifier
10. reconstruction loss
11. adversarial classification loss
12. `adverse_alpha`
13. `adverse_norm`
14. reconstruction-only phase
15. adverse phase, if present
16. cycle phase
17. exact detach locations
18. cycle roll/unroll directions
19. loss weights
20. optimizer construction/reset (both moments, LR, weight decay and other options)
21. scheduler construction and step ordering
22. freezing behavior (`requires_grad` and train/eval mode)
23. data augmentation
24. normalization
25. data shuffling
26. batch semantics
27. validation behavior
28. checkpoint resume behavior

## Step 2 — Parity report

Write the result to `docs/parity-report.md`, one row per component, with file/line references:

| Component | Legacy | Refactor | Status | Action |
|---|---|---|---|---|
| cycle image detach | generated image only | generated image only | equivalent | none |
| z-cycle weight | 0.5 | 0.5 | equivalent | none |

Status is one of: `equivalent`, `intentional difference`, `needs verification`,
`behavior-changing difference`, `bug`.

Keep the report concise. Propose fixes for `behavior-changing difference` and `bug` rows, but
don't apply them until they are approved. Apply approved fixes one per commit.

## Step 3 — Focused tests

Prefer small targeted tests over full training runs. Where practical, import the legacy
implementation from `../SIPE-main` and compare numerically (same seed, same weights, same
input, `torch.allclose`). Where a difference is intentional, test the documented new behavior.

- **Cycle graph:** mixed image detached; `s1`/`z1` not detached; roll/unroll restores sample
  correspondence; `z` gets the 0.5 weight.
- **Curriculum boundaries:** exact global steps at which phases change.
- **Alpha schedule:** `adverse_alpha` just before, at, and after each transition.
- **Optimizer reset:** AdamW state cleared at a normal phase boundary, not cleared when resuming
  mid-phase.
- **LR scheduler:** LR at phase start, just before restart, at restart, just after restart, at
  phase transition.
- **Frozen modules:** both `requires_grad` and train/eval mode, including after Lightning's
  epoch-boundary `.train()` calls.
- **Dataset leakage:** no WSI in more than one split.
- **Domain labels:** scanner labels and patch-expanded targets match the input sample.

## Checkpoint resume checks

After resuming mid-phase, verify: `global_step`, active phase, local step within the phase,
LR, scheduler position, optimizer state, frozen/unfrozen modules, `adverse_alpha`.

## Done when

- every component has a status other than `needs verification`, or an explicit "not verified"
  with the reason;
- all approved fixes are committed with their tests;
- the invariants in `CLAUDE.md` are marked verified or corrected;
- `CLAUDE.md` "Current phase" is updated.
