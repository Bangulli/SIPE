# Benchmarks (PLISM, HEST)

Robustness (PLISM) and utility (HEST-Benchmark) of CATS checkpoints, run with the
official code of both benchmarks.

## Bench env

`bench/` is a separate uv project, because HEST/TRIDENT need `transformers<5`. It
installs SIPE, `../HEST` and `../plism-benchmark` as editable packages. Run everything
through `bench/run`, which pins the venv:

```bash
bench/run python ...    # or: bench/run pytest tests/test_bench_adapters.py
```

The `override-dependencies` in `bench/pyproject.toml` deviate on purpose from the
benchmark repos' declared pins. Every run records them in `provenance.json`.

`../HEST` and `../plism-benchmark` are forks (origin = voreille, upstream = official).
Branch `sipe` holds two small opt-in patches:

- plismbench: external extractors via `module:factory`;
- HEST: `skip_download` and `custom_encoder_name`.

Model code stays in SIPE (`src/sipe/bench/`).

## Benchmark a post-hoc estimator

A map fitted on cached SCORPION features (soft-shrinkage eraser, roadmap estimators) is
benchmarked with `--posthoc <map file>` in place of `--ckpt`. The representation is
`posthoc_<pooling>`: the map file's backbone (`encoder`, default H0-mini; timm/HF weights)
pooled as in the SCORPION feature cache it was fitted on (`gap` = mean of the patch tokens,
`cls` = token 0), then the map in float32 (the backbone keeps the benchmark's autocast).
The tables show other backbones in the repr column (e.g. `posthoc_cls[uni2-h]`). A raw
row of a backbone is an identity map file (`sipe.bench.identity_map`). File format:
`sipe.bench.posthoc` docstring / `CLAUDE.md`. UNI2-h end to end: `scripts/uni2h_run.sh`.

```bash
bench/run python -m sipe.bench.plism --posthoc runs/bench/<fit run>/<map>.pt --tag <tag>
bench/run python -m sipe.bench.hest  --posthoc runs/bench/<fit run>/<map>.pt --tag <tag> --datasets '*'
```

`provenance.json` → `checkpoint` then holds the map file (sha256, kind, scalar fields such
as `lambda`) and the fit run dir with its `git.txt`.

## PLISM condition-mean oracle (not a method)

Reference ceiling for per-condition translations. It uses the PLISM condition (slide =
staining x scanner) at inference, so it breaks the inference contract on purpose. No
extraction: it rewrites the features of an existing PLISM run (each tile x becomes
x - mu_slide + m, mu_slide = mean over all 16,278 tiles of its slide) and runs the
official `compute_metrics` (the metric half of a PLISM run; the user runs it):

```bash
bench/run python -m sipe.bench.plism_oracle \
    --features-from runs/bench/2026-10-13_10-02-08_lxbelshark_053f_h0_mini_gap_ref \
    --tag plism_oracle_condmean_gap [--center global|zero]
```

- `--center global` (default): m = mean of the 91 slide means (keeps the origin, as the
  affine maps whose bias is the train mean; cosine depends on the origin). `zero`: m = 0.
- Pairing unit: PLISM slide means. It bounds translations only; a map that also
  rescales can exceed it. Outputs as a PLISM run, plus `condition_means.npz`.

## Inspect PLISM retrieval (which tiles are retrieved)

Diagnostic for the top-k numbers, no new metric. Two steps:

1. **Pick pairs** (main env, seconds; reads the runs' `metrics/8139_tiles/<tag>/metrics.csv`):
   ```bash
   uv run python -m sipe.eval.plism_pairs --runs <gap ref run> <method run> ... --tag <tag>
   ```
   The first run is the reference. `pairs.csv`: one row per slide pair, `shift`
   (`inter-scanner` / `inter-staining` / `joint`), `hr`, per run cosine/top-1/top-10 and
   the top-1 delta vs the reference. `shortlist.csv` (also printed): per shift, the `--k`
   worst/best reference pairs and the largest gains/losses per method (`reason` column).
2. **Draw neighbours** (bench env). Ranks and summaries for every pair take ~1–2 s per
   pair and representation; tile grids take ~7 s per figure, so they are drawn only for
   the first `--figure-pairs` pairs (default 3; `-1` = all, `0` = none) and the
   `--figure-kinds` (default `misses fixed`; also `hits`, `broken`):
   ```bash
   bench/run python -m sipe.bench.plism_retrieval_viz \
       --pairs-csv runs/bench/<plism_pairs run>/shortlist.csv --reason worst_ref gain_<label> \
       [--pair GMH_S60:GMH_P] [--max-pairs N] --tag <tag>
   ```
   Representations default to the runs of the `plism_pairs` run (same reference);
   `--features` overrides. Protocol as plismbench `TopkAccuracy` (candidates = both
   slides minus the query, both directions), cosine in float32 instead of float16.
   Outputs: `queries.csv.gz` (per query `rank_true`, `rank_true_cross` = rank among the
   other slide only, `n_same_top10`, `grid_dist_top1`, cosines), `summary.csv` (per pair
   and direction; misses: rank quartiles, share with a same-slide top-1, share with
   `rank_true_cross == 1` = lost only to same-slide tiles; hit/miss transitions vs the
   reference; plismbench's per-pair top-1/10 next to ours), and `figures/`: query | true
   match | top-10 grids for `hits`, `misses` (stratified by rank) and `fixed_<label>` /
   `broken_<label>`, tagged T (true) / S (same slide as the query) / O (other slide, wrong
   location), plus `rank_curve_<pair>.png` (top-r accuracy for every r).

## Benchmark a checkpoint

The checkpoint must be self-contained, i.e. written at or after `6ccb1e3`. The
benchmarked representation is `z_gap`, the GAP of `z` over the 16×16 patch grid
(704-d). Use `--representation backbone_gap|backbone_cls|s` for the reference rows,
which use the same weights and normalization.

```bash
bench/run python -m sipe.bench.plism --ckpt runs/<run>/checkpoints/last.ckpt --tag cats_<run>
bench/run python -m sipe.bench.hest  --ckpt runs/<run>/checkpoints/last.ckpt --tag cats_<run> --datasets '*'
```

- **claude-box:** `/dev/shm` is 64 MB, so add `--workers 0` (PLISM) or
  `--num-workers 0` (HEST). Pick a free GPU with `CUDA_VISIBLE_DEVICES`.
- **PLISM defaults:** the plismbench defaults (91 slides from `../plism-benchmark/data/plism`,
  metrics on 8,139 tiles/slide, batch 32, autocast). `--smoke` accepts a download dir with
  fewer slides; it patches plismbench's `NUM_SLIDES` and is for testing only. The slides
  must give at least one inter-scanner, one inter-staining and one joint pair (e.g.
  `GIVH_AT2`, `GIVH_GT450`, `GIV_AT2`): with one stain only, plismbench's aggregation fails
  on the empty groups (`Length mismatch: Expected 0 rows`).
- **HEST defaults:** the HEST `BenchmarkConfig` defaults (ridge on 256-d PCA, batch 128),
  float16 autocast as in TRIDENT's h0-mini, and data from `../HEST/data/bench_data`
  (`skip_download`).

Each run writes `runs/bench/<stamp>_<host>_<rand>_<tag>/`:

- `config.yaml`: runner args and the exact kwargs passed to the official code;
- `git.txt`, `git_hest.txt`, `git_plism-benchmark.txt`: commit, dirty flag and diff;
- `provenance.json`, which records:
  - the checkpoint: sha256, step, curriculum phase, and the training run's commit;
  - the representation, normalization and precision;
  - the repo states, package versions and bench overrides;
- outputs:
  - PLISM: `features/<tag>/`, `metrics/<n_tiles>_tiles/<tag>/results.csv`;
  - HEST: `embeddings/`, `results/<tag>::<time>/dataset_results.json`.

The loader warns if the checkpoint is in the reconstruction-only phase, because that
disentangler was never trained adversarially.
