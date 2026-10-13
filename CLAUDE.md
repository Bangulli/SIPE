# Project

Acquisition (scanner) harmonization of frozen pathology-foundation-model embeddings, trained on
the **SCORPION multi-scanner dataset** and evaluated on PLISM (robustness) and HEST (utility).
Started as a port of https://github.com/Bangulli/SIPE (SIPE/CATS); that phase is closed (see
"History" below).

**Plan: `docs/roadmap.md`** (target: 4-page ISBI submission, deadline 2026-10-26).
**Core question:** can a scanner-agnostic, statistically constrained *nonlinear* estimator beat
the optimal global linear soft-shrinkage map on the robustness-vs-preservation trade-off, and
still transfer to PLISM? The roadmap fixes the order: finish soft shrinkage → content-dependence
diagnostic → linear controls → Gaussian mixture / quadratic correction → (only if the decision
gates allow) conditional covariance or a constrained VAE. Prefer the simpler model when gains
are small or unstable.

**Scope and contract** (roadmap section 1):
- Paired data may be used at fit time. **Inference takes one embedding, uses global fit
  parameters, and never needs the scanner ID or a paired view.**
- Input: frozen H0-mini GAP over patch tokens (768-d, `backbone_gap`), same features for every
  method.
- Fit everything (means, PCA bases, covariances, clusters, gates) on the train split only; pick
  hyperparameters on SCORPION val only. Don't tune on PLISM; it is an external benchmark.
- State the pairing unit of every method: tile/ROI-level pairs or slide-level scanner means.
- Scanner-probe accuracy is a diagnostic, not the target. Don't claim scanner invariance or
  unseen-scanner generalization from SCORPION WSI-disjoint tests alone.
- Report order: SCORPION val (selection) → SCORPION test → PLISM → HEST.

**Progress log: `docs/progress.md`.** Whenever you check or interpret runs (fits, sweeps,
SCORPION proxy, PLISM/HEST bench runs), update it in the same turn:
- add or update the run's row: run dir, config/tag, key metrics, verdict;
- mark runs still in progress;
- record each roadmap decision (D0–D6) when it is taken, with the evidence;
- adjust "Open points / next steps" and the "as of" date.
Say in your reply that you updated it.

**Never type a metric by hand.** Every value must be read from a file under `runs/` (or from
W&B), and each row must name the run dir it came from.
- Results tables are generated: `uv run python -m sipe.eval.results_tables --write` refreshes
  the `<!-- results:… -->` blocks in `docs/progress.md`, and `--check` exits 1 if they are
  stale. Never edit inside a block. Block options (`exclude=`, `extra=`, `cols=`, `min_steps=`,
  `module=`) are listed in the module docstring.
- A new benchmark output, estimator output or metric goes into a parser in
  `sipe.eval.results_tables`, not into a hand-typed table.
- Numbers in prose must be copied from a generated table or a result file, with the run dir
  cited. For W&B values, use `wandb_history` / `last_val_mean` from that module (it reads the
  local `.wandb` file; `wandb.Api()` has no key here).
- Exception: references outside `runs/` (`../plism-benchmark/output/...`) go through
  `extra=` and are labeled by their path.

## History

- CATS port and feature-space models (ScannerVAE, PairedVAE, MathieuGAN, DistMatchVAE):
  branch `valentin/scorpion`, results in `docs/archive/progress-cats-scanner-removal.md`.
  Outcome: unpaired methods did not remove the scanner (probe at the noise floor).
  CATS and MathieuGAN stay below GAP on PLISM, but the unpaired
  DistMatchVAE `sr40r1h1` beats GAP in every PLISM row (one seed;
  `runs/bench/2026-10-13_20-17-04_lxbelshark_244b_dist_match_vae_sr40r1h1`; `g7fkiwbp` pending,
  see `docs/progress.md`). PairedVAE beats GAP on PLISM, and its removal comes from the
  pair-alignment loss.
- Implementation reference for that code (legacy behavior, curriculum, intentional
  differences, architecture): `docs/cats-reference.md`. Read it before changing anything under
  `src/sipe/training/`, `src/sipe/model/` or the CATS configs.
- `main` = original author code, read-only checkout at `../SIPE-main` (never modify it).
  `*_legacy.py` = copies of external code; don't edit them.

# Data

- SCORPION: 480 H&E regions, 48 WSIs, 5 scanners, 2,400 scanner-paired images, resampled to
  0.5 µm/px, 1568×1568 center crop, 224×224 non-overlapping tiles.
- Filenames encode location + scanner: `slide_1-sample_1-tile_0_0-AT2.jpg`. The shared prefix
  identifies the matched tissue location across scanners. Every slide has the same tile set
  on every scanner.
- P1000 tiles sit about one H0-mini token (~7 µm) off the other scanners: registration noise
  can look like scanner variation (roadmap section 1, "Alignment").
- Splits: seeded (42), grouped by WSI, val/test 0.1 each. All datamodules
  (`src/sipe/data/scorpion.py`) and the soft-shrinkage sweep share this split. Data sanity
  tests: `docs/parity-audit.md`.
- Normalization follows the backbone's timm data config (`src/sipe/model/encoders.py`).
- **Cached features:** `runs/bench/2026-10-13_15-44-35_lxbelshark_d6f7_scorpion_softshrink_h0mini_gap/`
  holds `features_{train,val,test}.npz` + `meta_*.csv` (H0-mini, fp16 autocast, GAP pooling;
  provenance in that dir). Reuse them for post-hoc estimators (`--features-from`) instead of
  re-extracting, so every method sees identical inputs. If you need new features (e.g. CLS,
  token grids), extract once into a new run dir and record it here.

# Post-hoc estimators

- **Soft-shrinkage eraser** (COMPAYL, from `../vfm-geom-xai`): `x' = mu + P (x - mu)`,
  `P = A (A + λB)^-1`, A the shrunk feature covariance, B the second moment of the scanner
  deltas. Deltas are per-slide scanner means minus the slide mean, i.e. **slide-level pairs**
  (on SCORPION equal to the mean of tile-pair deltas). This is the bar to beat.
- The eraser code is vendored verbatim in `src/sipe/baselines/soft_delta_legacy.py` and
  `shrinkage_legacy.py`, with the source commit in the header. Don't edit them; re-vendor
  instead. vfm-geom-xai itself is not a dependency.
- Sweep: `uv run python -m sipe.eval.soft_shrinkage_sweep --tag <tag> [--poolings gap cls]
  [--features-from <run dir>]`.
  - Fits on the train split, sweeps λ on val with the SCORPION-proxy metrics.
  - λ rule (`--select-gain-frac`, default 0.95): the smallest λ reaching 95% of the best val
    top-1 gain. The region probe is the guard (flagged in `selected.json`). The goal is to
    mitigate scanner distances, not to erase the scanner: never select λ by probe removal.
  - Reads test once, for λ=0 and the selected λ, with the proxy protocol (probe trained on
    val). The val sweep trains its probes on the train split, so val and test probe numbers
    are not comparable.
  - Outputs: `runs/bench/<stamp>_<host>_<rand>_scorpion_<tag>/` with `eraser_<pool>_lam<λ>.pt`.
- `--poolings`: `gap` = mean of the patch tokens without prefix tokens (= `backbone_gap`);
  `cls` = token 0. Report which pooling a number uses.
- PLISM/HEST for post-hoc estimators: `--posthoc <map file>` instead of `--ckpt` in
  `sipe.bench.plism` / `sipe.bench.hest` (representation `posthoc_gap` = frozen backbone GAP,
  then the map in float32; `sipe.bench.posthoc`). Map file = one `torch.save` dict in the fit
  run dir: either `P` + `bias` (affine, `bias + (x - bias) P^T`, as the eraser files) or
  `class_path` + `init_args` + `state_dict` (any `nn.Module` [B, 768] -> [B, D']); optional
  `encoder` (default h0-mini) and `pooling` (only `gap`). The fit dir's `git.txt` goes into
  the bench provenance.
- New estimators (roadmap Experiments 1–5): fit on cached features, save the fitted operator
  and a config/provenance in a `runs/bench/...` dir (`sipe.bench.provenance`), and expose the
  same SCORPION-proxy metrics as the sweep, so `results_tables` can parse them. Settle a shared
  layout with the first one and document it here.

# Benchmarks and commands

Always through uv; never call the environment's Python directly (except to debug the env).

```bash
uv run pytest                 # skips tests marked slow (sipe fit roundtrips)
uv run pytest -m slow         # after any change to training code
uv run ruff check <files>     # `ruff check .` already fails on legacy code; keep new files clean
```

Evaluation (each writes `runs/bench/<stamp>_<host>_<rand>_<tag>/` with config, git info and
provenance):

```bash
uv run python -m sipe.eval.scorpion_retrieval --ckpt runs/<run>/checkpoints/last.ckpt --tag <tag>
bench/run python -m sipe.bench.plism --ckpt runs/<run>/checkpoints/last.ckpt --tag <tag>
bench/run python -m sipe.bench.hest  --ckpt runs/<run>/checkpoints/last.ckpt --tag <tag> --datasets '*'
```

- PLISM/HEST run in the separate bench env through `bench/run`, never plain `uv run`
  (`docs/benchmarks.md`). Reference rows: `--representation backbone_gap`.
- PLISM reporting (roadmap section 1): inter-scanner, inter-staining, joint shift; per slide
  pair; the HR stain explicitly; group-aware intervals (pairs share slides).
- Inside claude-box, pass `--workers 0` / `num_workers=0` (shared memory is 64 MB), and check
  `nvidia-smi` before GPU work.
- Don't hard-code a config filename into tooling unless it is the canonical project config.

# Training code (neural branch only)

Only Experiment 6 (constrained VAE) and the archived models train networks. For those:
- Lightning + LightningCLI YAML; one entry point `uv run sipe fit --config <config.yaml>`. A new
  method is a new `SIPEModule` subclass (`src/sipe/training/base.py`) plus a config, never a new
  CLI entry. Feature-space models subclass `FrozenBackboneModule`.
- Each fit writes `runs/<stamp>_<host>_<rand>/` (checkpoints/, config.yaml, git.txt,
  wandb.txt, wandb/); the W&B run name equals the run dir name. Keep this link.
- Smoke test after any training change:
  `uv run sipe fit --config <cfg> --trainer.fast_dev_run=true --trainer.logger=false
  --trainer.enable_checkpointing=false` (add `--trainer.callbacks=[]` if needed). For logger
  code: `--trainer.max_steps=2 --trainer.limit_train_batches=2 --trainer.limit_val_batches=1`
  with `WANDB_MODE=offline`.

# Change reporting

Never change fitting or training behavior silently. For every meaningful change, state:

1. what changed;
2. whether fitting/training or evaluation behavior changed;
3. what reference it was checked against (vendored code, the roadmap's equations, legacy);
4. which tests/commands were run;
5. remaining uncertainty.

Say **not verified** for anything not actually compared. Don't mix unrelated
formatting/refactoring into a fix. Each run must save its config and git commit.
