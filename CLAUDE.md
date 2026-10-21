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

**Progress docs.** Whenever you check or interpret runs (fits, sweeps, SCORPION proxy,
PLISM/HEST bench runs), update them in the same turn:
- `docs/progress.md` = the ISBI paper only: scope, claims with evidence, runs in flight and
  remaining tasks, the paper tables. Update a claim's evidence/status, the in-flight list and
  the "as of" date.
- `docs/experiment-log.md` = everything else: the roadmap status table (one row per step,
  run dirs, verdict), each roadmap decision (D0–D6) with its evidence, per-experiment
  readings, open points outside the paper, the other tables. Mark runs still in progress.
- `docs/findings.md` = lessons that change how to run or read experiments (protocols,
  benchmark quirks, negative results), each with its evidence; not every run.
Say in your reply which of them you updated.

**Never type a metric by hand.** Every value must be read from a file under `runs/` (or from
W&B), and each row must name the run dir it came from.
- Results tables are generated: `uv run python -m sipe.eval.results_tables --write` refreshes
  the `<!-- results:… -->` blocks in `docs/progress.md` and `docs/experiment-log.md`, and
  `--check` exits 1 if one is stale. Never edit inside a block. Block options (`exclude=`, `extra=`, `cols=`, `min_steps=`,
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
  see `docs/experiment-log.md`). PairedVAE beats GAP on PLISM, and its removal comes from the
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
- **Second backbone, UNI2-h** (`uni2-h` in `src/sipe/model/encoders.py`, gated: needs an HF
  token with access; 1536-d, CLS + 8 register tokens). Pooled on its **CLS token** (its
  official embedding), H0-mini on patch-token GAP (inherited from CATS; much better than
  H0-mini CLS on PLISM). Everything runs from `scripts/backbone_run.sh` (named
  `scripts/uni2h_run.sh` until 2026-10-21; tracked despite `*.sh` being ignored; any machine
  with the NAS, env vars in its header; `SMOKE=1` tests it): cache, G0 (`mixture_sweep --ks
  --alphas 1.0`), identity map, V1-S fit (`configs/consensus_gen.yaml` with encoder/pooling
  overridden on the command line; same resolved config as the earlier
  `configs/consensus_gen_uni2h.yaml`; nothing re-selected), `cgm_select --reference-map none`, PLISM, HEST. Run
  dirs go into `runs/uni2h_manifest.env`. `POOLING=gap` runs UNI2-h on patch-token GAP
  (manifest `runs/uni2h_gap_manifest.env`, tags `uni2h_gap_*`, reuses the cls cache).
- **Third backbone, CONCH v1.5** (`conch-v1.5`, `src/sipe/model/conch.py`, gated
  `MahmoodLab/conchv1_5`; `ENCODER=conch-v1.5 scripts/backbone_run.sh`, tags `conch15_*`,
  manifest `runs/conch15_manifest.env`). The repo is not a timm hub model; the authors
  recommend TRIDENT's `encoder_factory("conch_v15")`, which needs transformers<5 (bench env
  only), so `conch.py` rebuilds it in timm and `tests/test_bench_adapters.py` checks it
  against TRIDENT (bench env). As TRIDENT: 448 px input (224 px tiles upsampled inside the
  model, so every pipeline still feeds 224 px tiles), TRIDENT's position embedding, and the
  official embedding = pooling `head` (attentional pooler, 768-d; `pool_backbone` in
  `encoders.py`). The cache also holds `cls` and `gap` (1024-d trunk).
- **Augmented train caches** (CGM augmentation study): `uv run python -m sipe.features.extract
  --tag <tag> --policy none|geo|dihedral|zoom --copies K` (train split by default; meta
  column `aug` = copy; the scanner views of a tile copy share the transform; every view goes
  through the whole encoder). `sipe.features.extract.extract` is the one backbone pass behind
  every cache. Extracted 2026-10-15 (train, GAP, 18,620 tiles x 5 scanners
  per copy, seed 0): `runs/bench/2026-10-15_18-20-26_lxbelshark_cfad_augcache_geo` (geo x4),
  `runs/bench/2026-10-15_18-25-53_lxbelshark_490a_augcache_dihedral` (dihedral x8),
  `runs/bench/2026-10-15_18-37-41_lxbelshark_54ba_augcache_zoom` (zoom x4).

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
  - `--deltas slide|region|tile` (roadmap 2C): pairing unit of the deltas in B. `slide` is
    the default and the vfm-geom deltas; `region` / `tile` = each scanner's mean over the
    region / the tile's view, minus the mean over scanners. Only B changes; reselect λ on
    val (B is trace-normalized, so λ does not carry over between levels).
- `--poolings`: `gap` = mean of the patch tokens without prefix tokens (= `backbone_gap`);
  `cls` = token 0. Report which pooling a number uses.
- PLISM/HEST for post-hoc estimators: `--posthoc <map file>` instead of `--ckpt` in
  `sipe.bench.plism` / `sipe.bench.hest` (representation `posthoc_<pooling>` = frozen
  backbone GAP or CLS, then the map in float32; `sipe.bench.posthoc`). Map file = one
  `torch.save` dict in the fit run dir: either `P` + `bias` (affine, `bias + (x - bias) P^T`,
  as the eraser files) or `class_path` + `init_args` + `state_dict` (any `nn.Module` [B, D] ->
  [B, D']); optional `encoder` (default h0-mini) and `pooling` (`gap` default, or `cls`).
  The drivers write the cache's encoder (`proxy.cache_encoder`) and refuse a map whose
  encoder/pooling differs from the cache (`proxy.check_map_matches_cache`). Raw rows of a
  backbone without a checkpoint: `uv run python -m sipe.bench.identity_map --encoder <e>
  --pooling <p> --tag <tag>` (P = I). The fit dir's `git.txt` goes into the bench
  provenance.
- Experiment 1 (content-dependence diagnostic, no map): `uv run python -m
  sipe.eval.content_dependence --tag <tag> --features-from <feature cache>`; reads train/val
  only, writes `summary.csv`, `likelihood.csv`, `cluster_stats.csv` (table block `content`).
- Scanner-offset diagnostic (exchangeability of the nuisance, no map): `uv run python -m
  sipe.eval.scanner_offsets --tag <tag> --features-from <feature cache>`; tile-level
  pairs, train/val only, writes `summary.csv`, `energy.csv`, `likelihood.csv`,
  `scanner_stats.csv` (table block `offsets`).
- Content-dependent scanner offsets (does δ_k depend on content? no map): `uv run python -m
  sipe.eval.content_offsets --tag <tag> --features-from <feature cache>`; per scanner pair,
  target x_k − x_j, content from the other views (no shared noise); train/val only, writes
  `summary.csv`, `per_scanner.csv`, `per_cluster.csv`, `ridge.csv` (table block
  `content_offsets`).
- PLISM condition-mean oracle (uses the condition label: an oracle row, never a method):
  `bench/run python -m sipe.bench.plism_oracle --features-from <PLISM run> --tag <tag>
  [--center global|zero]`; rewrites that run's features, x - mu_slide + m, then the
  official metrics (`docs/benchmarks.md`).
- PLISM retrieval diagnostics (no new metric; `docs/benchmarks.md`):
  `uv run python -m sipe.eval.plism_pairs --runs <ref run> <method runs> --tag <tag>`
  merges the runs' per-pair `metrics.csv` into `pairs.csv` + `shortlist.csv`; then
  `bench/run python -m sipe.bench.plism_retrieval_viz --pairs-csv <shortlist.csv>
  [--reason worst_ref gain_<label>] [--pair GMH_S60:GMH_P] --tag <tag>` writes per-query
  ranks for every pair (fast) and draws top-10 grids only for `--figure-pairs` pairs
  (default 3) and `--figure-kinds` (default misses, fixed).
- New estimators (roadmap Experiments 2–5) follow the layout of the first one, Experiment 3:
  - fit code + an `nn.Module` map in `src/sipe/estimators/<name>.py` (numpy float64 fit,
    float32 buffers; the module is the map that is evaluated *and* saved, so `--posthoc`
    benchmarks exactly what was screened);
  - driver `sipe.eval.<name>_sweep --features-from <cache>`: hyperparameters on val only
    (reuse `sipe.eval.proxy`: `evaluate` / `summarize` / `select_by_gain`), the eraser λ=0.5 evaluated
    in the same run as `reference` (`--reference-map`), test read once;
  - run dir outputs: `sweep_val.csv`, `summary_val.csv` (probes trained on train),
    `summary_test.csv` (probes trained on val), `selected.json`, `map_<...>.pt` (module
    map file), plus estimator-specific CSVs; a parser + block in `results_tables`.
- Experiment 3 (Gaussian-mixture shrinkage G0/G1/G2-scale/G2, `sipe.estimators.mixture`):
  `uv run python -m sipe.eval.mixture_sweep --tag <tag> --features-from <feature cache>
  [--ks 2 4 8]`. Pairing unit: tile-level pairs (consensus over the 5 scanners). Single-view
  gating in the top-32 PCA subspace of the train views. Table block `mixture`.
- Experiment 4 (quadratic correction Q1/Q2/Q2-rand, `sipe.estimators.quadratic`):
  `uv run python -m sipe.eval.quadratic_sweep --tag <tag> --features-from <feature cache>
  [--ms 8 16]`. Base = consensus ridge regression (Exp 2B), not the eraser objective
  (decision 2026-10-14): γ = inf is exactly G0 (in-run G0 row). Pairing unit: tile-level
  pairs. γ = ridge on the quadratic block, largest γ reaching 95% of the best val gain
  over G0. Table block `quadratic`.
- Experiment 7 CGM (model, arms, recipe, gates, departures from the plan): `docs/experiment7.md`;
  original plan archived in `docs/archive/experiment7-draft.md`.
- Experiment 7 follow-up V1-S+D (content-dependent scanner shift δ_k + D_k z in the
  likelihood, low rank, map unchanged): `sipe.model.consensus_gen.ContentShiftCGM`,
  `configs/consensus_gen_shift.yaml` (= `consensus_gen.yaml` except the network; a test
  enforces it); `cgm_select` labels it `V1-S+D`.
- Experiment 7 CGM on the shared protocol: `uv run python -m sipe.eval.cgm_select --tag <tag>
  --features-from <feature cache> --runs runs/<fit> ... [--test selected|all|none]`. Scores
  each fit's `map_cgm.pt` on clean features next to raw, the eraser and an in-run G0 (α=1);
  fits that differ only by seed form one candidate (mean val top-1); simplest candidate
  (arm, then larger ρ) at 95% of the best gain over G0. Table block `cgm`.
- Experiment 7 LOSO analysis (nothing selected; an analysis of frozen models, not a gate):
  `uv run python -m sipe.eval.loso --tag <tag> --features-from <feature cache>
  [--split val|test] --runs <CGM fit dirs>`; paper numbers on test, read once. For each held-out
  scanner j, G0^-j and CGM-init^-j are fitted in the run on train without j; trained fits
  are `sipe fit --config configs/consensus_gen.yaml` (V0-S: plus `--model.init_args.network.
  init_args.encoder_hidden_dim=0`) with `--data.init_args.hold_out_scanner=<j>
  --data.init_args.views=4`, which drops j from train only (the 10 fits of 2026-10-16 record
  their overrides in their `config.yaml`). Fits without a hold-out are the
  full-data references. Readout: pairs of the split with j (`unseen`), map^-j - G0^-j and
  map^-j - map^full, with slide-bootstrap intervals. `cgm_select` refuses LOSO fits.
  Pairing unit: tile-level pairs over the kept scanners. Table block `loso`.
- CGM augmentation attribution (no training): `uv run python -m sipe.eval.aug_attribution
  --tag <tag> --features-from <feature cache> --aug-caches geo=<cache> ... [--maps
  label=<map>] [--retrieval-only]`. Closed-form {G0, CGM-init (= the V0-S init)} ×
  {clean, augmented caches} × sizes (5120 tiles = the CGM init, `tiles`, `all`); val only,
  nothing selected across arms; PLISM on the saved maps is the readout. Table block `augattr`.

# Benchmarks and commands

Always through uv; never call the environment's Python directly (except to debug the env).

**Long commands are run by the user.** Anything expected to take more than ~5 minutes
(sweeps on the full cache, PLISM/HEST runs, fits) is not run from here: implement and test
it (unit tests, a smoke run on a few slides), then give the exact command and say what to
check afterwards. Short checks and smoke runs are fine to run directly.

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
