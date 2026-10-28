# Estimators, diagnostics and backbones: command reference

Moved out of `CLAUDE.md` on 2026-10-28 (verbatim) to keep the always-loaded context small.
`CLAUDE.md` keeps the rules; this file holds the per-experiment recipes. Read the section you
need before running or changing one of these tools.

## Backbones and feature caches

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

## Soft-shrinkage eraser sweep

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

## PLISM/HEST for post-hoc maps

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
- PLISM for a post-hoc map without the backbone: `bench/run python -m
  sipe.bench.plism_apply --features-from <raw PLISM run> --posthoc <map> --tag <tag>`
  applies the map to that run's stored raw features (encoder/pooling checked), then the
  official metrics (~35 min instead of a full extraction).
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

## Diagnostics (no map)

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

## Experiments 3 and 4 (mixture, quadratic)

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

## Experiment 7 (CGM)

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
- Scanner-count sensitivity (no selection): `uv run python -m sipe.eval.scanner_count --tag <tag>
  --features-from <feature cache> --universe <scanners> [--sizes 2 3] [--split val|test]
  --runs <CGM fit dirs>`. Fits G0 / CGM-init / eraser (λ=0.5) on every subset of the
  universe in the run; trained fits are `sipe fit` with `--data.init_args.scanners=[a,b]
  --data.init_args.views=<n> --data.init_args.batch_size=<512/n>` (keep-list on train only,
  as `hold_out_scanner`; `scripts/scanner_count_run.sh` runs everything). Pairs scored on
  the universe scanners: all / seen / unseen; map[S] − G0[S] per subset and mean per n.
  Writes `plism_jobs.tsv`. Table blocks `scanner_count`, `scanner_count_plism` (PLISM tags
  `sc_<model>_<subset>`). `cgm_select` and `loso` refuse subset fits.
- CGM augmentation attribution (no training): `uv run python -m sipe.eval.aug_attribution
  --tag <tag> --features-from <feature cache> --aug-caches geo=<cache> ... [--maps
  label=<map>] [--retrieval-only]`. Closed-form {G0, CGM-init (= the V0-S init)} ×
  {clean, augmented caches} × sizes (5120 tiles = the CGM init, `tiles`, `all`); val only,
  nothing selected across arms; PLISM on the saved maps is the readout. Table block `augattr`.
