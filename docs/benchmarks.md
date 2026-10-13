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
`posthoc_gap`: frozen H0-mini (timm/HF weights) GAP over the patch tokens, exactly as the
SCORPION feature cache, then the map in float32 (the backbone keeps the benchmark's
autocast). File format: `sipe.bench.posthoc` docstring / `CLAUDE.md`.

```bash
bench/run python -m sipe.bench.plism --posthoc runs/bench/<fit run>/<map>.pt --tag <tag>
bench/run python -m sipe.bench.hest  --posthoc runs/bench/<fit run>/<map>.pt --tag <tag> --datasets '*'
```

`provenance.json` → `checkpoint` then holds the map file (sha256, kind, scalar fields such
as `lambda`) and the fit run dir with its `git.txt`.

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
  fewer slides; it patches plismbench's `NUM_SLIDES` and is for testing only.
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
