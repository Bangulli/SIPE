# Phase 5 — Baselines

Don't add baselines before the parity audit is complete, unless explicitly asked.

## Rules

- Keep baselines modular; one baseline per branch/PR.
- Reuse the same SCORPION splits, preprocessing and evaluation protocol as CATS.
- Shared evaluation code must not make CATS-specific assumptions.
- Record every baseline-specific choice in configuration.
- Don't modify the main CATS implementation to accommodate a baseline.
- When an official implementation exists, reproduce the documented method rather than inventing
  a protocol, and state any deviation explicitly.

## Likely categories

- raw frozen H0-mini features
- simple scanner/domain adversarial baseline
- scanner-invariant projection / post-hoc baseline
- other representation-disentanglement methods

## Before implementing one

Write a short spec: source paper/implementation, what is reproduced exactly, deviations and
why, config keys, and how it plugs into the shared evaluation.
