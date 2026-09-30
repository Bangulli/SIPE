# Project

Port/refactor of https://github.com/Bangulli/SIPE (SIPE/CATS) to the **SCORPION multi-scanner
histopathology dataset**, modernized around PyTorch Lightning / LightningCLI.

**Current phase: 1 — parity audit.** Follow `docs/parity-audit.md`.
Later phases: `docs/refactor-guidelines.md` (simplification), `docs/baselines.md` (baselines).
Order: behavioral fidelity → tests → simplification → baselines.

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
- **Augmentations** (legacy): GaussianBlur, RandomAffine, RandomErasing, RandomGrayscale,
  RandomInvert. Current torchvision's `RandomErasing` needs a tensor: PIL transforms before
  `ToTensor`, `RandomErasing` after. When changing augmentation code, separate (1) the intended
  augmentation, (2) library-compatibility changes, (3) real changes to probability/order.

# Legacy training semantics to preserve

Believed correct from a previous review, **not yet verified in this repo**: confirm each one
during the parity audit, then mark it verified here.

- **Curriculum fields:** `type, epochs, lr, adverse_alpha, restarts, norm/adverse_norm,
  freeze_backbone, freeze_tangler`.
- **Optimizer reset:** legacy builds a new AdamW per phase. Refactor reproduces it by clearing
  AdamW state (both moments) and applying the phase LR at the transition, without replacing
  Lightning's optimizer. Resuming mid-phase must NOT clear state (don't reset just because
  `on_train_batch_start` first sees a phase after resume).
- **Scheduler:** new `CosineAnnealingWarmRestarts(optimizer, T_0=restarts)` per phase. Legacy
  steps it per epoch; refactor per step (`restart_steps`). Order: `optimizer.step()` then
  `scheduler.step()`. No second Lightning scheduler on the same optimizer.
- **`adverse_alpha` ≠ GRL alpha.** Legacy calls `loss.set_adverse_alpha(alpha)`, consumed inside
  the original `AdversarialClassifLoss`. The GRL has its own alpha (normally 1.0). Don't replace
  `adverse_alpha` by scaling the domain loss unless proven equivalent. Prefer reusing the
  original loss.
- **`adverse_norm`:** forwarded via `loss.set_adverse_norm(norm)`. Keep it until its effect is
  inspected.
- **Reconstruction-only phase:** `L = ImageReconLoss(reconstruction, image)`, no ×20.
- **Cycle phase** (legacy `SIPE_Loss_Adversarial_Cycle` in `../SIPE-main`):
  `20·ImageReconLoss + 1.0·L1(s_cycle, s_orig) + 0.5·L1(z_cycle, z_orig) + adversarial/domain
  losses`. The 0.5 is intentional. No renormalizing, no L1→MSE, no extra detaches.
- **Cycle graph** (easy to break during cleanup; protect with a test):

  ```python
  s1, z1 = model(batch1)
  s1_prime = torch.roll(s1, 1, 0)
  mixed_image = model.recon_image(s1_prime, z1).detach()   # only the generated image is detached
  s2, z2 = model(mixed_image)
  s2_prime = torch.roll(s2, -1, 0)                          # roll back before comparison
  s_cycle_loss = L1(s2_prime, s1)                           # s1, z1 NOT detached as targets
  z_cycle_loss = 0.5 * L1(z2, z1)
  ```

- **Domain classifiers:** `s` predicts the scanner normally; `z` predicts it through gradient
  reversal. Classification on `z` is patch-wise/spatial if the architecture is spatial, with the
  scanner target expanded to spatial positions.
- **Frozen backbone:** H0-mini is frozen unless a config says otherwise. Frozen means
  `requires_grad=False` AND eval mode (Lightning may call `.train()` on the whole module).
- **`freeze_tangler`:** legacy freezes the components producing/recombining the latents and their
  classifiers. Check exactly which current modules are affected; don't rely on similar names.
- **Optimization setup:** batch size 512 (close to the original author). Never silently change
  batch size, optimizer, augmentation, precision, effective batch size or curriculum length while
  claiming parity. If memory/runtime forces a change, state it and its likely consequences.

# Architecture intent

- The CATS network is a plain `torch.nn.Module`: no logging, optimizer, curriculum or experiment
  state. Conceptually `s, z = network.encode(images)`, `reconstruction = network.decode(s, z)`
  (or a structured `forward()` returning `s, z, reconstruction`).
- H0-mini dims: backbone 768, `s` 64 (global/sample-level), `z` 704 (spatial).
- The re-entangler combines broadcast `s` with spatial `z` before image reconstruction.

# Logging

- Lightning + W&B. Logging lives in callbacks where practical, not in model/training logic.
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
