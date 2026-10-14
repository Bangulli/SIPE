# CATS / feature-space-model reference (archived phase)

Moved out of `CLAUDE.md` on 2026-10-13 when the project switched to the acquisition-shrinkage
roadmap (`docs/roadmap.md`). Read this before touching the CATS port, the curriculum, or the
feature-space models (ScannerVAE, PairedVAE, MathieuGAN, DistMatchVAE). Results of that phase:
`docs/archive/progress-cats-scanner-removal.md` (tag/branch `valentin/scorpion`).

## Layout

- `main` = original author code. Read-only checkout at `../SIPE-main` (never modify it).
- The Lightning refactor trains and gives qualitatively reasonable results, but parity with the
  original is not established.
- `*_legacy` / `*_legacy.py` = copies of original code (the vendored soft-shrinkage files are
  the exception: they come from vfm-geom-xai). `scripts/` comes from the original
  implementation unless stated otherwise.

## Intentional differences (do not "fix" back)

- **Data:** BPTorch is unavailable; SCORPION uses a native PyTorch/Lightning pipeline. Never
  reintroduce BPTorch.
- **Domains:** 5 scanner domains. Scanner identity replaces the original stain-domain target.
- **Heads:** only the scanner/domain branch is kept. No organ/pathology heads.
- **Framework:** Lightning + LightningCLI + YAML configs, `src/` layout, `uv`. The original
  custom trainer is reference only.
- **Curriculum in optimizer steps**, not epochs (`epochs → steps`, `restarts → restart_steps`).
  Phase semantics otherwise match the original. Absolute step counts are config choices: never
  infer a new schedule without checking the active config and documenting it.

## Datamodules and augmentations

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
  The set is configurable via `data.init_args.augmentations` (names `blur`, `affine`,
  `grayscale`, `invert`, `erasing`, plus the new `rot90` = exact 90° rotations and `zoom` =
  80–100 % area crop resized back; unset = this legacy set, unchanged); flips stay separate
  flags. `configs/paired_vae_geometric_aug.yaml` uses `[rot90, zoom]` + flips. All ops draw
  from the torch RNG, so the paired views of a location share them (tested).

## Legacy reference behavior

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

## Architecture intent

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

## Logging (CATS metrics)

- Key metrics: reconstruction loss, scanner loss, scanner accuracy from `s` and from `z`, cycle
  `s`/`z` losses, LR, curriculum phase, `adverse_alpha`; latent diagnostics (mean |s|, mean |z|,
  std s, std z) for divergence.
- Qualitative images: source, reconstruction, `s`-swapped reconstruction, re-encoded swapped
  reconstruction, optional s-only / z-only. Denormalize to display RGB before sending to W&B.

## Lightning rules specific to the curriculum

- Don't bypass Lightning's optimizer/checkpoint lifecycle casually. Custom behavior kept for
  legacy fidelity (AdamW reset, phase-local scheduler, step-based curriculum) must be documented
  in a comment explaining why.
- `global_step` counts optimizer updates. If gradient accumulation is introduced, re-evaluate
  all step-based durations and restart periods.
- Checkpoint resume must restore: `global_step`, active phase, local step in phase, LR,
  scheduler position, optimizer state, frozen modules, `adverse_alpha`.

## Change reporting against legacy

When a change touches the CATS port, also state which legacy behavior was checked. Only call
something "equivalent to legacy" if that execution path was actually compared; otherwise write
**not verified**. Neither codebase is automatically right: the original isn't good because it's
old, the refactor isn't better because it's cleaner.
