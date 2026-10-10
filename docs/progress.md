# Progress summary (as of 2026-10-10)

This file tracks what has been tried on SCORPION, what each attempt gave, and what comes
next. Setup details are in `docs/benchmarks.md` (PLISM/HEST),
`docs/parity-audit.md` (data sanity tests) and `docs/baselines.md`.

## 1. Infrastructure

| commit | what |
|---|---|
| `3bfc343`, `774b0f7` | Lightning/LightningCLI port of CATS to SCORPION (5 scanners, WSI-grouped splits, data sanity tests). |
| `2e1938e`, `b04664e` | Each fit writes `runs/<stamp>_<host>_<rand>/` (config, git info, W&B link). |
| `4dfde46` | Encoder builder: backbone and its timm transform from one source. **All earlier runs normalized with ImageNet mean/std instead of the H0-mini stats.** |
| `6ccb1e3` | Checkpoints are self-contained (`load_from_checkpoint` works); only `last.ckpt` is kept. |
| `92bfa94`, `bffab23` | `bench/` env and `sipe.bench.{plism,hest}`: official PLISM and HEST code on forks (`../plism-benchmark`, `../HEST`, branch `sipe`). Provenance is recorded for every bench run. |
| `82b61c5`, `4fd9c05` | Backbone tokens via `forward_features` (required for H-optimus-1); H-optimus-1 config. The H-optimus-1 config has only been smoke-tested (recon phase), not trained. |
| `fe1c930` | SCORPION proxy `sipe.eval.scorpion_retrieval`: cross-scanner retrieval of the same region on the held-out split, plus post-hoc scanner probes (logreg/MLP). |
| `5a77728` | A SCORPION region probe was added to the proxy. |

Benchmarked representation: GAP over the 16×16 patch grid of the scanner-free latent (`z_gap`).
Reference: GAP of the backbone patch tokens (`backbone_gap`).

## 2. Results

### PLISM (official code, 91 slides, 8,139 tiles/slide), top-1 accuracy

| row | H0-mini GAP | H0-mini CLS | legacy-step CATS z_gap (`ce50`) | CATSv2 z_gap (`9lctrbqp`) | original SIPE |
|---|---|---|---|---|---|
| inter-scanner | **0.810** | 0.600 | 0.688 | 0.751 | 0.666 |
| inter-staining | **0.259** | 0.120 | 0.239 | 0.255 | 0.237 |
| scanner + staining | **0.173** | 0.055 | 0.154 | 0.165 | 0.158 |
| all | **0.227** | 0.100 | 0.201 | 0.216 | 0.203 |

Bench runs: `runs/bench/2026-10-08_10-18-32_lxbelshark_6ee6_cats_s2bqyu2g_z_gap` (legacy-step CATS)
and `runs/bench/2026-10-08_20-42-51_lxbelshark_de75_cats_9lctrbqp_z_gap` (CATSv2). The other
columns come from earlier comparisons. **No method beats raw H0-mini GAP on PLISM so far.**

### SCORPION proxy (held-out split)

| representation | cross-scanner top-1 | scanner probe (logreg) |
|---|---|---|
| backbone GAP | 0.802 | 0.956 |
| legacy-step CATS z_gap (`ce50`) | 0.668 | 0.908 |
| CATSv2 z_gap (`9lctrbqp`) | **0.844** | 0.917 |

CATSv2 beats GAP on SCORPION but loses on PLISM, and a fresh probe still reads the scanner
from its z. The gain is a SCORPION-specific alignment, not scanner removal.

## 3. Attempts and what we learned

1. **Legacy-step CATS** (`configs/cats_legacy_steps.yaml`; run
   `runs/2026-10-07_14-11-04_lxbelshark_ce50`; 12,250 steps: recon → adverse → cycle;
   frozen H0-mini, bs 512, 16-mixed). Below GAP everywhere and close to the original
   author's SIPE result, so the weakness is the method, not the port. Diagnosis:
   - z is a randomly initialized ReLU projection learned through a weak pixel decoder;
   - the linear per-token GRL adversary sits at chance while pooled z still encodes the scanner;
   - the cycle mostly teaches robustness to decoder blur.
2. **CATSv2** (`configs/cats_v2_unpaired.yaml`; commits `fe1c930`, `d0c7a1a`; W&B `9lctrbqp`).
   - Reconstruction in feature space, z initialized as identity on the backbone tokens.
   - Pooled GRL adversaries; LayerNorm before the GRL fixed a |z| blow-up.
   - Result: PLISM 0.751. The GRL adversaries sit at chance, but a fresh probe still gets
     92% (backbone 96%). **The GRL was fooled.**
3. **ScannerVAE** (`uv run sipe-vae fit`; `src/sipe/model/scanner_vae.py`,
   `src/sipe/training/scanner_vae_module.py`; commits `5a77728`, `ff26970`). Design:
   - Fader / Mathieu-style VAE on frozen H0-mini tokens: per-token Gaussian z, identity
     init, and a decoder conditioned on a learned scanner embedding.
   - KL with β warm-up; an adversary on LayerNorm(GAP(μ)). No cycle, no curriculum.
   - Every validation trains a fresh linear probe on WSI-disjoint halves of val. It logs
     `val/probe_zgap_acc` next to `val/probe_backbone_acc` (about 0.80 on this small split).
     This is the removal signal; the adversary's accuracy alone is not.

   All runs: 6,000 steps, frozen H0-mini, bs 512, 16-mixed. Results (val, last available step):

   | run | config | probe z / backbone | feature recon | KL | σ | adversary acc |
   |---|---|---|---|---|---|---|
   | `qvc6voso` (`runs/2026-10-09_23-37-17_lxbelshark_f08b`) | GRL, β 0.1 | 0.74 / 0.80 | 0.051 | 2.04 | 0.24 | 0.23 |
   | `o375g4md` (`runs/2026-10-10_12-05-05_lxbelshark_9597`) | GRL, β 1 | 0.78 / 0.80 | 0.244 | 0.52 | 0.86 | 0.25 |
   | `da7vqx1r` (`runs/2026-10-10_13-16-50_lxbelshark_ca85`) | Fader, β 0.1, 5 adversary steps | 0.75 / 0.80 (min 0.69 at step 2000) | 0.080 | 3.0 | 0.23 | 0.47 |

   - β 0.1 GRL: stable, but the KL barely compresses and the GRL is fooled again.
   - β 1: reconstruction is 5× worse and σ is near the prior, yet the scanner is still
     readable. A stronger bottleneck removes content, not scanner. **Dead end.**
   - Fader: **still running** at the last check (step 4,300/6,000).
     - Its adversary is honest (training fit accuracy about 0.54, chance 0.2), but the
       encoder is losing (confusion about 1.24; 1.0 = uniform).
     - Scanner removal is at most slightly better than GRL; probe values move by about
       ±0.03 between validations.
     - Its final numbers still need to be checked.
   - The β=1 run ran on `5a77728` with the uncommitted Fader diff (GRL mode); that code is
     the same as what was committed in `ff26970`.

4. **PairedVAE**, stage 1 toward Mathieu et al. (`uv run sipe-paired fit --config
   configs/paired_vae.yaml`; `src/sipe/model/paired_vae.py`,
   `src/sipe/training/paired_vae_module.py`, `PairedSCORPIONDataModule`). Implemented
   2026-10-10, not trained yet. Design:
   - s is encoded from the image (MLP on GAP); z is a per-token VAE latent. No adversary.
   - Batches: 256 locations × 2 scanners, with the same augmentation on both views.
   - Losses:
     - Mathieu's same-scanner swap reconstruction (s from another tile of the same scanner);
     - cross-scanner translation `Dec(s_donor(b), z_a) ≈ x_b` on 4×4-pooled features,
       because P1000 tiles sit about 1 token (~7 µm) off the other scanners;
     - batch-normalized GAP(μ) pair alignment;
     - KL (β 0.1).
   - Validation:
     - fresh probes on z, backbone and s;
     - PLISM-style cross-scanner retrieval top-1 on val, as in the SCORPION proxy
       (backbone about 0.89).
   - Stage 2 (next): replace the translation with a class-conditional GAN in embedding space.

## 4. Open points / next steps

- Check the final Fader numbers. If the probe gap improves, run the SCORPION proxy and
  then PLISM.
- Fader knobs: a larger `adversary_weight` (3–5) and fewer `adversary_steps` (1–2), so the
  encoder can keep up.
- Next ideas from the VAE plan: a conditional discriminator on generated features (Mathieu
  `L_adv`), then a paired translation loss using SCORPION `pair_id` (allowed if unpaired
  stalls).
- Not started: H-optimus-1 training (memory at bs 512 untested; `unspecified_dim` choice
  open), HEST runs on a winning model ("does no harm" check), and baselines beyond raw GAP.
- Each run (and the H0-mini GAP baseline) has one seed, so small differences
  (about ±0.03 on the val probe) are within noise.
