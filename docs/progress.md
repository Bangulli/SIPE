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
3. **ScannerVAE** (`uv run sipe fit --config configs/scanner_vae*.yaml`; `src/sipe/model/scanner_vae.py`,
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
   | `da7vqx1r` (`runs/2026-10-10_13-16-50_lxbelshark_ca85`) | Fader, β 0.1, 5 adversary steps | 0.72 / 0.80 (mean of last 10 vals 0.72; min 0.69 at step 2000) | 0.095 | 3.7 | 0.23 | 0.60 |

   - β 0.1 GRL: stable, but the KL barely compresses and the GRL is fooled again.
   - β 1: reconstruction is 5× worse and σ is near the prior, yet the scanner is still
     readable. A stronger bottleneck removes content, not scanner. **Dead end.**
   - Fader (finished, 6,000 steps): **no real gain over GRL.**
     - Probe gap 0.08 (z 0.72 vs backbone 0.80) vs 0.06 for GRL, within the ±0.03 noise
       between validations.
     - The adversary is honest and wins at the end. While the encoder's cosine LR decays
       (the adversary's LR is constant), its training fit accuracy rises from 0.54 to 0.78,
       val adversary accuracy from 0.42 to 0.60, and confusion from 1.24 to 1.84.
       Meanwhile recon worsens (0.080 → 0.095), KL rises (3.0 → 3.7) and |z| grows.
     - Conclusion: in a fair game, an adversary on z can't remove the scanner at this
       weight. No SCORPION proxy or PLISM run.
   - The β=1 run ran on `5a77728` with the uncommitted Fader diff (GRL mode); that code is
     the same as what was committed in `ff26970`.

4. **PairedVAE**, stage 1 toward Mathieu et al. (`uv run sipe fit --config
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
   - Stage 2: replace the translation with a class-conditional GAN in embedding space (item 5).

5. **MathieuGAN**, stage 2 toward Mathieu et al. (`uv run sipe fit --config
   configs/mathieu_gan.yaml`; `src/sipe/training/mathieu_gan_module.py`). Implemented
   2026-10-10, not trained yet (only a 300-step smoke run in the box). Design:
   - Same network as PairedVAE (`PairedVAE`). Training is **unpaired** (scanner labels
     only); paired batches are kept so validation matches PairedVAE.
   - Losses: same-scanner swap reconstruction + KL (β 0.1) + class-conditional GAN.
     Fake = `Dec(s(x_other), z_i)` with x_other from another scanner *and* another
     location (never i's pair partner), labeled with x_other's scanner; real = backbone
     tokens with their own scanner.
   - D: projection discriminator with spectral norm on the 16×16 token grid (1×1 conv,
     two stride-2 3×3 convs, mean pool), hinge loss; Adam(0.5, 0.999), same cosine LR
     decay as the generator (Fader lesson). Manual optimization: 1 G step (= 1 global
     step) then 1 fp32 D step. Adversarial weight 1.0, warm-up 1,000 steps.
   - No adversary on z, no translation or alignment loss in training.
   - Validation: everything PairedVAE logs (probes on z/backbone/s, retrieval top-1,
     paired translation loss as evaluation only), plus D accuracies and
     `val/fake_{target,source}_acc`: a probe trained on real backbone GAP reads the
     scanner of GAP(fake). Target should rise, source should fall.
   - Known limitation (also in Mathieu): the decoder can overwrite a scanner signature
     left in z, so a fooled D doesn't prove z is scanner-free. Read the z probe.
   - Box smoke run (`runs/2026-10-10_13-23-13_claude-box_103a`, W&B offline `7zo0kmfr`;
     300 steps, adversarial warm-up 50 instead of 1,000, 8 val batches only, so absolute
     numbers are not comparable with full runs). Finished; mechanics OK, too short to judge:
     - D dominates in training (real/fake acc ~0.95–1.0, d_loss 0.05–0.5).
     - The GAN costs a lot of reconstruction: feature recon 0.025 before the GAN starts, then
       0.6 at its peak, 0.12 at step 300 (stage-0 runs end at ~0.05).
     - Fakes don't look like the target scanner yet: `fake_target_acc` 0.18 → 0.22 (chance
       0.2), while `fake_source_acc` 0.37 → 0.20. The decoder moves features off the real
       distribution rather than toward the target scanner.
     - Probe z 0.69 vs backbone 0.73; retrieval z 0.860 vs backbone 0.865; s probe 0.52.
     - Verdict: works mechanically. Watch whether recon recovers and `fake_target_acc` rises.
       Adversarial weight 1.0 may be too high; 0.1 is the first knob to try.

## 4. Open points / next steps

- Launch PairedVAE (`uv run sipe fit --config configs/paired_vae.yaml`), then check it
  against the backbone: probe gap, `val/retrieval_top1_zgap` vs `_backbone`, and `s` probe
  (should be high).
- Launch MathieuGAN (`uv run sipe fit --config configs/mathieu_gan.yaml`) and compare with
  PairedVAE on the same val metrics; watch `train/d_{real,fake}_acc` for D domination or
  collapse and `val/fake_target_acc` vs `val/fake_source_acc`. If the second GPU is
  free, run `--model.init_args.adversarial_weight=0.1` in parallel (smoke run: recon
  suffers at 1.0).
- If adversaries on z come back: the Fader knobs not tried are a larger
  `adversary_weight` (3–5), fewer `adversary_steps`, and decaying the adversary's LR with
  the encoder's (the end of the Fader run was dominated by the adversary).
- Not started: H-optimus-1 training (memory at bs 512 untested; `unspecified_dim` choice
  open), HEST runs on a winning model ("does no harm" check), and baselines beyond raw GAP.
- Each run (and the H0-mini GAP baseline) has one seed, so small differences
  (about ±0.03 on the val probe) are within noise.
