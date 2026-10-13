# Project

Port/refactor of https://github.com/Bangulli/SIPE (SIPE/CATS) to the **SCORPION multi-scanner
histopathology dataset**, modernized around PyTorch Lightning / LightningCLI.

**Goal:** a scanner-removal method on frozen H0-mini features that beats the raw backbone
(`backbone_gap`) on PLISM (robustness) and does no harm on HEST (utility), ideally **without
paired data**. CATS was the starting point. It loses to GAP on PLISM, so the work moved to
feature-space models. Each run must save its config and git commit hash.

**Current phase: benchmarking + method search.**
- Screen each run on the SCORPION proxy (`sipe.eval.scorpion_retrieval`), then run PLISM/HEST
  (`docs/benchmarks.md`).
- PairedVAE is the paired reference to approach. DistMatchVAE is the current unpaired
  candidate.
- Data sanity tests: `docs/parity-audit.md`. Baselines: `docs/baselines.md`.

**Progress log: `docs/progress.md`.** Whenever you check or interpret runs (training runs,
SCORPION proxy, PLISM/HEST bench runs), update it in the same turn:
- add or update the run's row: run dir, W&B id, config, key metrics, verdict;
- mark runs still in progress, and the step you read;
- adjust "Open points / next steps";
- update the "as of" date.
Say in your reply that you updated it.

**Never type a metric by hand.** Every value must be read from a file under `runs/` (or from
W&B), and each row must name the run dir it came from.
- Section 2 tables are generated: `uv run python -m sipe.eval.results_tables --write` refreshes
  the `<!-- results:… -->` blocks, and `--check` exits 1 if they are stale. Never edit inside a
  block. Block options (`exclude=`, `extra=`, `cols=`, `min_steps=`, `module=`) are listed in
  the module docstring.
- A new benchmark output or metric goes into a parser in `sipe.eval.results_tables`, not
  into a hand-typed table.
- Numbers in prose must be copied from a generated table or a result file, with the run dir
  cited. For W&B values, use `wandb_history` / `last_val_mean` from that module (it reads the
  local `.wandb` file; `wandb.Api()` has no key here).
- Exception: references outside `runs/` (`../plism-benchmark/output/...`) go through
  `extra=` and are labeled by their path.

## Layout

- `main` = original author code. Read-only checkout at `../SIPE-main` (never modify it).
- Current branch = Lightning refactor. It trains and gives qualitatively reasonable results,
  but parity with the original is not yet established.
- `*_legacy` / `*_legacy.py` = copies of original code. `scripts/` comes from the original
  implementation unless stated otherwise.

# Intentional differences (do not "fix" back)

- **Data:** BPTorch is unavailable; SCORPION uses a native PyTorch/Lightning pipeline. Never
  reintroduce BPTorch.
- **Domains:** 5 scanner domains. Scanner identity replaces the original stain-domain target.
- **Heads:** only the scanner/domain branch is kept. No organ/pathology heads.
- **Framework:** Lightning + LightningCLI + YAML configs, `src/` layout, `uv`. The original
  custom trainer is reference only.
- **Curriculum in optimizer steps**, not epochs (`epochs → steps`, `restarts → restart_steps`).
  Phase semantics otherwise match the original. Absolute step counts are config choices: never
  infer a new schedule without checking the active config and documenting it.

# Data

- SCORPION: 480 H&E regions, 48 WSIs, 5 scanners, 2,400 scanner-paired images, resampled to
  0.5 µm/px, 1568×1568 center crop, 224×224 non-overlapping tiles.
- Filenames encode location + scanner: `slide_1-sample_1-tile_0_0-AT2.jpg`. The shared prefix
  identifies the matched tissue location across scanners.
- Splits must be grouped by WSI (no leakage).
- Datamodules (`src/sipe/data/scorpion.py`):
  - `SCORPIONDataModule`: unpaired tiles. CATS, ScannerVAE.
  - `PairedSCORPIONDataModule`: locations × scanner views via `pair_id`. PairedVAE, MathieuGAN.
  - `UnpairedTrainSCORPIONDataModule`: unpaired training batches, paired validation.
    DistMatchVAE.

  All three share the same seeded, WSI-grouped split. A method that claims "unpaired" may
  use `pair_id` in evaluation only.
- Normalization follows the backbone's timm data config (encoder builder,
  `src/sipe/model/encoders.py`). Runs before `4dfde46` used ImageNet mean/std by mistake.
- **Augmentations** (legacy, `../SIPE-main/pretrain_50k.py:63`): `GaussianBlur(3),
  RandomAffine(3), RandomErasing(p=0.5), RandomGrayscale(p=0.1), RandomInvert(p=0.5)`, then the
  timm H0-mini transform. `train_1M.py` uses no augmentation. `RandomErasing` has always needed
  a tensor, so BPTorch probably fed tensors (timm 1.x `MaybeToTensor` passes them through). The
  refactor (`src/sipe/data/scorpion.py`) applies PIL ops → `ToTensor` → `RandomErasing`, which
  moves Erasing after Grayscale/Invert (an order change). It also adds random flips.

# Legacy reference behavior

Reference, not invariants. Checked against `../SIPE-main` on 2026-10-02 (L = legacy, R = refactor
`src/sipe/training/cats_module.py`). Deviate when it helps convergence; just say so.

- **Curriculum fields:** `type, epochs, lr, adverse_alpha, restarts, adverse_norm,
  freeze_backbone, freeze_tangler` (`add_step` args: `step_type`, `norm`, `freeze_bb`;
  L `trainer/curriculum_trainer.py:28-38`). `adverse_alpha` may be a per-epoch list. pretrain_50k
  ramps 0.1→0.9 over 9 epochs, then 1.0 (`pretrain_50k.py:77-79`). R: list +
  `adverse_alpha_interval_steps`.
- **Optimizer reset:** L builds a new `AdamW(model.parameters(), lr=lr)` per phase with torch
  defaults (wd 0.01). That resets both moments and the per-param `step`. R calls
  `optimizer.state.clear()` (clears all three) and sets the phase LR. It skips the clear when
  resuming mid-phase. L has no mid-phase resume (it saves model weights only).
- **Scheduler:** new `CosineAnnealingWarmRestarts(optimizer, T_0=restarts)` per phase. L steps it
  per epoch after the batch loop. R steps per optimizer step (`restart_steps`). Both call
  `optimizer.step()` before `scheduler.step()`. No Lightning `lr_scheduler` is configured.
- **`adverse_alpha`:** a multiplier on the z-branch (adversarial) CE term only, inside
  `AdversarialClassifLoss` (L `losses/adversarial_classif_loss.py:20-22`). The s-CE is unscaled.
  The GRL alpha is separate and fixed at 1.0. In validation the z term is dropped.
- **`adverse_norm`:** divides both s-CE and z-CE by `log(n_classes)` (log 5 for SCORPION).
- **Reconstruction-only phase:** `L = MSE(reconstruction, image)`, no ×20. L validates this phase
  with the adverse loss (recon + s-CE + …). R validates recon only.
- **Adverse phase** (unused by the legacy scripts): recon (no ×20) + domain losses.
- **Cycle phase** (`SIPE_Loss_Adversarial_Cycle`): `20·MSE + L1(s_cycle, s_orig) +
  0.5·L1(z_cycle, z_orig) + domain losses`. Classification uses first-pass `s1`, `z1`.
  Note: L's cycle path does not run as committed (`curriculum_trainer.py:399-400` call
  `self.transform_organs`, which doesn't exist on the trainer).
- **Cycle graph** (R matches L):

  ```python
  s1, z1 = model(batch1)
  s1_prime = torch.roll(s1, 1, 0)
  mixed_image = model.recon_image(s1_prime, z1).detach()   # only the generated image is detached
  s2, z2 = model(mixed_image)
  s2_prime = torch.roll(s2, -1, 0)                          # roll back before comparison
  s_cycle_loss = L1(s2_prime, s1)                           # s1, z1 NOT detached as targets
  z_cycle_loss = 0.5 * L1(z2, z1)
  ```

- **Domain classifiers:** `s` → linear. `z` → per-patch (16×16) GRL → linear, with the target
  `repeat_interleave`d over patches.
- **Frozen backbone:** L only sets `requires_grad=False` and runs the backbone in train mode. R
  also sets eval mode. For H0-mini (no dropout or drop-path) the two are numerically identical.
  pretrain_50k keeps the backbone frozen. train_1M unfreezes it (lr 1e-5, bs 128).
- **`freeze_tangler`:** L freezes the whole `Entangler`: projectors, all heads and the
  reentangler. The decoder is not frozen. R freezes disentangler, reentangler and both scanner
  classifiers. All legacy curricula use `False`.
- **Optimization setup:** L pretrain_50k runs bs 512, fp32. The R configs use `16-mixed`. State
  any batch size, precision or schedule change when reporting results.

# Architecture intent

- The CATS network is a plain `torch.nn.Module`: no logging, optimizer, curriculum or experiment
  state. Conceptually `s, z = network.encode(images)`, `reconstruction = network.decode(s, z)`
  (or a structured `forward()` returning `s, z, reconstruction`).
- H0-mini dims: backbone 768, `s` 64 (global/sample-level), `z` 704 (spatial).
- The re-entangler combines broadcast `s` with spatial `z` before image reconstruction.
- Feature-space models: ScannerVAE, PairedVAE, MathieuGAN, DistMatchVAE (`src/sipe/model/`,
  `src/sipe/training/*_module.py`, one config each in `configs/`).
  - They encode the frozen backbone patch tokens per token into `z`, with `s` global. There is
    no curriculum, and pixel decoding is for visualization only.
  - They subclass `FrozenBackboneModule` (`src/sipe/training/base.py`).
  - The benchmarked representation is `z_gap`, the GAP of `z`, the posterior mean for VAEs
    (`sipe.bench.encoders.CATSFeatures`).
- Adversaries at chance prove nothing on their own. Judge scanner removal with fresh
  WSI-disjoint probes, comparing `z` against the backbone.

# Post-hoc baselines

- **Soft-shrinkage eraser** (from `../vfm-geom-xai`). It acts on frozen backbone features:
  `x' = mu + P (x - mu)`, `P = A (A + λB)^-1`, where A is the shrunk feature covariance and B
  the second moment of the scanner deltas. Deltas are per-slide scanner means minus the slide
  mean. That needs every slide under several scanners, i.e. **paired data at slide level**
  (on SCORPION the tile sets are identical, so it equals the mean of tile-pair deltas). It is
  a paired baseline, comparable to PairedVAE, not to unpaired methods.
- The eraser code is vendored verbatim in `src/sipe/baselines/soft_delta_legacy.py` and
  `shrinkage_legacy.py`, with the source commit in the header. Don't edit them; re-vendor
  instead. vfm-geom-xai itself is not a dependency.
- Sweep: `uv run python -m sipe.eval.soft_shrinkage_sweep --tag <tag> [--poolings gap cls]`.
  - It fits on SIPE's train split (seed 42, val/test 0.1, grouped by slide).
  - It sweeps λ on val with the SCORPION-proxy metrics and selects by val cross-scanner top-1.
    The region probe is the guard; it is flagged in `selected.json`.
  - It reads test once, for λ=0 and the selected λ, with the proxy protocol (probe trained
    on val).
  - Outputs go to `runs/bench/<stamp>_scorpion_<tag>/`. Use `--features-from <run dir>` to
    skip feature extraction. Most of the run time is the per-λ sklearn probes.
- `--poolings`: `gap` = mean of the patch tokens without prefix tokens (= `backbone_gap`, the
  reference for CATS-style models); `cls` = token 0. Report which pooling a number uses.
- Compare its test row with the SCORPION-proxy table in `docs/progress.md`. PLISM/HEST for the
  eraser is not wired yet (it needs a `backbone_gap` + eraser extractor in `sipe.bench`).

# Logging

- Lightning + W&B. Logging lives in callbacks where practical, not in model/training logic.
- Each `fit` writes `runs/<stamp>_<host>_<rand>/` (checkpoints/, config.yaml, git.txt,
  wandb.txt, wandb/). The W&B run name equals the run dir name, and the W&B config holds
  `run_dir`. Keep this link.
- Key metrics: reconstruction loss, scanner loss, scanner accuracy from `s` and from `z`, cycle
  `s`/`z` losses, LR, curriculum phase, `adverse_alpha`; latent diagnostics (mean |s|, mean |z|,
  std s, std z) for divergence.
- Qualitative images: source, reconstruction, `s`-swapped reconstruction, re-encoded swapped
  reconstruction, optional s-only / z-only. Denormalize to display RGB before sending to W&B.

# Commands

Always through uv; never call the environment's Python directly (except to debug the env).

```bash
uv run pytest
uv run ruff check .
uv run sipe fit --config <config.yaml>
```

`uv run pytest` skips tests marked `slow` (the `sipe fit` checkpoint roundtrips, ~2–3 min each
on CPU). Run them with `uv run pytest -m slow` after any training-related change.

Smoke test after any training-related change (plus the relevant focused tests):

```bash
uv run sipe fit --config <config.yaml> \
  --trainer.fast_dev_run=true --trainer.logger=false \
  --trainer.enable_checkpointing=false        # add --trainer.callbacks=[] if callbacks need a full run
```

`fast_dev_run` replaces loggers with a dummy, so it doesn't test logger or W&B code. For that,
run `--trainer.max_steps=2 --trainer.limit_train_batches=2 --trainer.limit_val_batches=1` with
`WANDB_MODE=offline`.

`uv run ruff check .` already fails on legacy code (`*_legacy.py`, `scripts/`, old eval
scripts). Keep new or changed files clean: `uv run ruff check <files>`.

Evaluation (each writes `runs/bench/<stamp>_<host>_<rand>_<tag>/` with config, git info and
provenance):

```bash
uv run python -m sipe.eval.scorpion_retrieval --ckpt runs/<run>/checkpoints/last.ckpt --tag <tag>
bench/run python -m sipe.bench.plism --ckpt runs/<run>/checkpoints/last.ckpt --tag <tag>
bench/run python -m sipe.bench.hest  --ckpt runs/<run>/checkpoints/last.ckpt --tag <tag> --datasets '*'
```

PLISM/HEST run in the separate bench env through `bench/run`, never plain `uv run`. Inside
claude-box, pass `--workers 0` / `num_workers=0`, because shared memory there is 64 MB.

Don't hard-code a config filename into tooling unless it is the canonical project config.

# Lightning rules

- Configure through LightningCLI-compatible YAML, not custom argument parsing.
- One entry point, `sipe fit`. The config picks the training module (`model.class_path`, a
  `SIPEModule` subclass from `src/sipe/training/base.py`) and the datamodule
  (`data.class_path`). A new method is a new module class plus a config, never a new CLI
  entry. Run configs saved before 2026-10-10 have a flat `model:` section: wrap it in
  `class_path`/`init_args` to replay them.
- Don't bypass Lightning's optimizer/checkpoint lifecycle casually. Custom behavior kept for
  legacy fidelity (AdamW reset, phase-local scheduler, step-based curriculum) must be documented
  in a comment explaining why.
- `global_step` counts optimizer updates. If gradient accumulation is introduced, re-evaluate
  all step-based durations and restart periods.
- Checkpoint resume must restore: `global_step`, active phase, local step in phase, LR,
  scheduler position, optimizer state, frozen modules, `adverse_alpha`.

# Change reporting

Never change training behavior silently. For every meaningful change, state:

1. what changed;
2. whether training behavior changed;
3. which legacy behavior was checked;
4. which tests/commands were run;
5. remaining uncertainty.

Only call something "equivalent to legacy" if that execution path was actually compared.
Otherwise write **not verified**. Don't mix unrelated formatting/refactoring into a bug fix.

Neither codebase is automatically right: the original isn't good because it's old, the
refactor isn't better because it's cleaner.
