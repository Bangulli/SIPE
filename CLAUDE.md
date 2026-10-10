# Project

Port/refactor of https://github.com/Bangulli/SIPE (SIPE/CATS) to the **SCORPION multi-scanner
histopathology dataset**, modernized around PyTorch Lightning / LightningCLI.

**Goal:** get CATS converging on SCORPION and benchmark it against baselines, fast. Fix only
what blocks convergence or makes results invalid. Each run must save its config and git commit
hash.

**Current phase: convergence + baselines.** Data sanity tests: `docs/parity-audit.md`.
Baselines: `docs/baselines.md`.

**Progress log: `docs/progress.md`.** Whenever you check or interpret runs (training runs,
SCORPION proxy, PLISM/HEST bench runs), update it in the same turn:
- add or update the run's row: run dir, W&B id, config, key metrics, verdict;
- mark runs still in progress, and the step you read;
- adjust "Open points / next steps";
- update the "as of" date.
Say in your reply that you updated it.

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
- The CATS cycle objective does not need a paired sampler; pairing info may be exposed for
  evaluation or future paired objectives.
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

Smoke test after any training-related change (plus the relevant focused tests):

```bash
uv run sipe fit --config <config.yaml> \
  --trainer.fast_dev_run=true --trainer.logger=false \
  --trainer.enable_checkpointing=false        # add --trainer.callbacks=[] if callbacks need a full run
```

`fast_dev_run` replaces loggers with a dummy, so it doesn't test logger or W&B code. For that,
run `--trainer.max_steps=2 --trainer.limit_train_batches=2 --trainer.limit_val_batches=1` with
`WANDB_MODE=offline`.

Don't hard-code a config filename into tooling unless it is the canonical project config.

# Lightning rules

- Configure through LightningCLI-compatible YAML, not custom argument parsing.
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
