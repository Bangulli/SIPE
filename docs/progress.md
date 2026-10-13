# Progress summary (as of 2026-10-13)

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

| row | H0-mini GAP | H0-mini CLS | legacy-step CATS z_gap (`ce50`) | CATSv2 z_gap (`9lctrbqp`) | original SIPE | PairedVAE z_gap (`ysltwjoj`) |
|---|---|---|---|---|---|---|
| inter-scanner | 0.810 | 0.600 | 0.688 | 0.751 | 0.666 | **0.839** |
| inter-staining | 0.259 | 0.120 | 0.239 | 0.255 | 0.237 | **0.293** |
| scanner + staining | 0.173 | 0.055 | 0.154 | 0.165 | 0.158 | **0.215** |
| all | 0.227 | 0.100 | 0.201 | 0.216 | 0.203 | **0.267** |

Bench runs: `runs/bench/2026-10-08_10-18-32_lxbelshark_6ee6_cats_s2bqyu2g_z_gap` (legacy-step CATS)
and `runs/bench/2026-10-08_20-42-51_lxbelshark_de75_cats_9lctrbqp_z_gap` (CATSv2). The other
columns come from earlier comparisons (H0-mini GAP: `../plism-benchmark/output/metrics/8139_tiles/h0_mini_gap`,
2026-10-07). H0-mini GAP was re-run with the current tooling
(`sipe.bench.plism --representation backbone_gap`,
`runs/bench/2026-10-13_10-02-08_lxbelshark_053f_h0_mini_gap_ref`, SIPE `06257ff`; dirty only in
`docs/progress.md`). It reproduces the old row: inter-scanner 0.811 (0.810 before),
the other rows are identical, and the mean per-pair difference is 0.000 on every row.
So both GAP rows and PairedVAE share the same preprocessing. PairedVAE: `runs/bench/2026-10-10_17-32-32_lxbelshark_95c5_paired_vae_ysltwjoj`
(SIPE `06257ff` clean, checkpoint `last.ckpt` step 6,000).

**PairedVAE is the first method to beat raw H0-mini GAP on PLISM, on every row.** Paired
comparison per slide pair against GAP (same numbers against the old and the re-run GAP) (top-1, mean difference / share of pairs improved):
inter-scanner +0.028 / 88% (273 pairs), inter-staining +0.034 / 90% (546),
scanner + staining +0.042 / 89% (3,276), all +0.040 / 89% (4,095). Every scanner gains on
average (P the most: +0.072, GAP's weakest scanner at 0.655). It also gains on
inter-staining, which it never trained on. One failure: the **HR** stain loses −0.179 on
inter-scanner pairs (e.g. AT2↔P 0.33 → 0.07, GT450↔S360 0.41 → 0.10; slide-pair cosine
drops from ~0.89 to ~0.66–0.71), while HRH and every other stain gain. HR is already hard for
GAP; the latent seems to break on it (out-of-distribution stain for SCORPION?).
Caveat: one PairedVAE seed.

### HEST (official code, ridge on PCA-256, Pearson over 50 genes)

| task | H0-mini GAP | PairedVAE z_gap (`ysltwjoj`) |
|---|---|---|
| CCRCC | 0.267 (0.045) | in progress |
| COAD | 0.256 (0.025) | |
| HCC | 0.074 (0.020) | |
| IDC | 0.578 (0.078) | 0.577 (0.078) (IDC-only run) |
| LUNG | 0.535 (0.051) | |
| LYMPH_IDC | 0.267 (0.033) | |
| PAAD | 0.497 (0.044) | |
| PRAD | 0.371 (0.032) | |
| READ | 0.202 (0.049) | |
| SKCM | 0.594 (0.067) | |
| **average** | **0.364** | |

GAP, all tasks: `runs/bench/2026-10-13_11-03-53_lxbelshark_8f0f_h0_mini_gap_ref` (finished; its
IDC equals the IDC-only run `..._df55_h0_mini_gap_ref`). PairedVAE IDC only:
`runs/bench/2026-10-13_10-12-16_lxbelshark_b34d_paired_vae_ysltwjoj`; no cost on IDC. PairedVAE
all tasks: `runs/bench/2026-10-13_13-20-25_lxbelshark_2cf4_paired_vae_ysltwjoj`, **in progress**
(read at CCRCC). SIPE `06257ff`/`5da426e`, dirty only in `docs/progress.md`.

### SCORPION proxy (held-out split)

| representation | cross-scanner top-1 | scanner probe (logreg) |
|---|---|---|
| backbone GAP | 0.802 | 0.956 |
| legacy-step CATS z_gap (`ce50`) | 0.668 | 0.908 |
| CATSv2 z_gap (`9lctrbqp`) | 0.844 | 0.917 |
| PairedVAE z_gap (`ysltwjoj`) | **0.962** | **0.850** |

CATSv2 beats GAP on SCORPION but loses on PLISM, and a fresh probe still reads the scanner
from its z. The gain is a SCORPION-specific alignment, not scanner removal.

PairedVAE (`runs/bench/2026-10-10_14-52-05_claude-box_807c_scorpion_paired_vae_ysltwjoj`):
top-1 0.962 (top-5 0.993), and it has the lowest scanner probe so far (logreg 0.850, MLP 0.831 vs
0.956 / 0.939 for GAP). Content is kept: region probe 0.985 vs 0.975 for GAP. Its `s`
reads the scanner (logreg 0.966) but carries little location (region probe 0.36, retrieval
0.003). The alignment loss trains on SCORPION pairs, so PLISM decides.

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
   2026-10-10. Design:
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
   - Run `ysltwjoj` (`runs/2026-10-10_15-18-38_lxbelshark_20c2`, commit `f48c72a`, clean;
     `configs/paired_vae.yaml` defaults: 6,000 steps, frozen H0-mini, 256 pairs/batch,
     16-mixed, lr 3e-4). **Finished** (6,000 steps; table read at step 3,899, final
     mean of last 10 vals: probe z / backbone 0.689 / 0.802, s 0.93, retrieval 0.977 /
     0.887, recon 0.047, KL 1.64):

     | metric (val) | step 99 | step 3,899 | mean of vals 3,000–3,899 |
     |---|---|---|---|
     | probe z / backbone | 0.79 / 0.79 | 0.69 / 0.81 | 0.695 / 0.801 |
     | probe s | 0.73 | 0.93 | 0.93 |
     | retrieval top-1 z / backbone | 0.951 / 0.887 | 0.977 / 0.887 | 0.977 / 0.887 |
     | pair cos z / backbone | 0.94 / 0.92 | 0.98 / 0.92 | |
     | translation / align (backbone align 0.299) | 0.252 / 0.220 | 0.238 / 0.239 | |
     | feature recon / KL / σ / \|z\| | 0.019 / 2.64 / 0.08 / 0.81 | 0.048 / 1.66 / 0.23 / 0.83 | |

     - Best removal so far: probe gap ~0.11 (vs 0.06 GRL, 0.08 Fader), still drifting down
       slowly; s carries the scanner (0.93), as intended. All 768 units active, no collapse.
     - Retrieval z 0.977 vs backbone 0.887 on val (WSI-disjoint from train). Caveat: the
       alignment loss trains exactly this pairing, so PLISM is the real test.
     - Feature recon rose 0.019 → 0.047 and is flat since step ~1,500 (GRL β0.1 ends at 0.051).
     - Metrics were flat after step ~4,000.
     - SCORPION proxy (held-out test split, section 2): top-1 0.962 vs 0.802 GAP; scanner
       probe 0.850 vs 0.956.
     - PLISM (section 2): inter-scanner 0.839 vs 0.810 GAP, all 0.267 vs 0.227; gains on
       ~89% of slide pairs; fails on the HR stain.
     - Verdict: **best so far on SCORPION and PLISM**; first method to beat GAP on PLISM.

5. **MathieuGAN**, stage 2 toward Mathieu et al. (`uv run sipe fit --config
   configs/mathieu_gan.yaml`; `src/sipe/training/mathieu_gan_module.py`). Implemented
   2026-10-10; full run `h65a5z5x` 2026-10-13 (results at the end of this item). Design:
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
   - Full run `h65a5z5x` (`runs/2026-10-13_11-54-38_lxbelshark_36b0`, commit `5da426e`, dirty
     only in `docs/progress.md`; `configs/mathieu_gan.yaml` defaults: adversarial weight 1.0,
     warm-up 1,000, D LR decay on, 6,000 steps, frozen H0-mini, 16-mixed). **Finished.**
     Mean of the last 10 validations, PairedVAE `ysltwjoj` in brackets:

     | metric (val) | step 99 | step 5,999 | last-10 mean | PairedVAE |
     |---|---|---|---|---|
     | probe z / backbone | 0.79 / 0.80 | 0.78 / 0.80 | 0.783 / 0.802 | 0.689 / 0.802 |
     | probe s | 0.66 | 0.83 | 0.83 | 0.93 |
     | retrieval top-1 z / backbone | 0.896 / 0.887 | 0.921 / 0.887 | 0.921 / 0.887 | 0.977 / 0.887 |
     | feature recon / KL | 0.098 / 2.66 | 0.197 / 2.23 | 0.195 / 2.23 | 0.047 / 1.64 |
     | translation (eval only) | 0.41 | 0.44 | 0.44 | 0.24 (trained) |
     | fake target / source acc (chance 0.2) | 0.18 / 0.24 | 0.34 / 0.25 | 0.37 / 0.25 | |
     | D real / fake acc (val) | 0.99 / 0.89 | 0.26 / 0.74 | 0.30 / 0.71 | |

     - Almost no removal: probe gap 0.02 (PairedVAE 0.11, GRL 0.06), within noise.
     - The GAN does move fakes toward the target scanner (fake_target 0.18 → ~0.37), but
       z keeps the scanner, so the decoder rewrites the scanner signature on top of z
       (the known limitation above).
     - Reconstruction gets worse throughout (0.07 at step 499 → 0.197, still rising); no
       recovery after the warm-up.
     - D does not dominate: training D accuracies fall from ~0.90/0.86 to ~0.79/0.77 as the
       LRs decay.
     - Small retrieval gain on val (0.921 vs 0.887), well below PairedVAE.
     - Verdict: **unpaired GAN alone does not remove the scanner from z**; worse than
       PairedVAE on every val metric. PLISM/HEST still to run for the record.

6. **DistMatchVAE**: unpaired scanner removal by distribution matching (`uv run sipe fit
   --config configs/dist_match_vae.yaml`; `src/sipe/training/dist_match_vae_module.py`,
   `UnpairedTrainSCORPIONDataModule`). Implemented 2026-10-13. Goal (user, 2026-10-13): a
   method that needs **no paired data**. PairedVAE stays the paired upper reference.
   - Why: every unpaired attempt so far relied on a learned adversary that the
     encoder or decoder could satisfy without z losing the scanner (GRL fooled, Fader
     adversary wins, MathieuGAN decoder overwrites). Fixed statistics can't be fooled.
   - Same network as PairedVAE. Losses: same-scanner swap recon + KL (β 0.1) +
     matching on g = GAP(μ), standardized per dimension over the batch (statistics not
     detached, so shrinking z doesn't help):
     - MMD² between every pair of scanners (unbiased; mean of RBF kernels at
       0.25–4 × the median squared distance);
     - per-scanner mean (‖mean_a‖²/D) and covariance (‖Cov_a − Cov‖²_F/D²) matching:
       what a linear probe reads.
     - Weights 1.0 each, warm-up 1,000 steps. No adversary, no pairs, no translation.
   - Data: training batches are 512 **shuffled tiles** (no location key), so each scanner
     in a batch sees random, mostly different locations. Paired batches would leak
     pairing into MMD, since both views of a location share tissue. Validation stays
     paired (all 5 scanners per location) and logs the PairedVAE metrics. On those
     batches every scanner sees the same tissue, so `val/mmd`, `val/mean_loss` and
     `val/cov_loss` measure the scanner effect alone. `*_backbone` versions give the scale.
   - Assumption: every scanner sees the same tissue distribution. True for SCORPION by
     construction; it breaks on data where tissue differs by scanner.
   - Box smoke run (`runs/2026-10-13_12-27-13_claude-box_bc5a`, W&B offline `klr6rd9h`;
     300 steps, warm-ups 50 instead of 1,000, 8 val batches only, so the val probes are
     noisy and not comparable with full runs). Finished; mechanics OK:

     | val (paired batches) | step 99 | step 299 | backbone |
     |---|---|---|---|
     | MMD / mean / cov term | 0.0200 / 0.0377 / 0.0075 | 0.0194 / 0.0372 / 0.0073 | 0.0317 / 0.0536 / 0.0084 |
     | probe z (backbone) | 0.79 (0.71) | 0.80 (0.71) | |
     | retrieval top-1 z (backbone) | 0.921 (0.865) | 0.927 (0.865) | |
     | feature recon / KL / \|z\| | 0.012 / 1.67 / 0.73 | 0.015 / 1.56 / 0.71 | |

     - On val, z shows less scanner difference than the backbone (MMD −39%, mean −31%,
       cov −13%), but the terms are flat after step 99.
     - Training batches: MMD drops to ~0.001 (backbone ~0.02). The mean and cov terms sit
       at their **sampling-noise floor**: for 512 i.i.d. 768-d samples over 5 scanners the
       floor is 0.0078 for each; train reads 0.008–0.009 and 0.010. So with
       ~100 random tiles per scanner, the per-batch moment estimates can't see the
       remaining scanner difference. Fixes if the full run stalls: running (EMA)
       per-scanner statistics across batches, or larger matching weights.
     - Probe z ≥ probe backbone here is a warning sign but uses 8 val batches only; the
       full run's probes (whole val split) decide.
     - Reconstruction is fine (0.015; PairedVAE ends at 0.047). All 768 units active.

## 4. Open points / next steps

- PairedVAE `ysltwjoj`: done. PLISM beats GAP on all rows (inter-scanner 0.839 vs 0.810,
  all 0.267 vs 0.227). To make the claim solid:
  - ~~re-run the GAP baseline with the current tooling~~ done: it reproduces 0.810/0.811;
  - a second seed of PairedVAE (gain +0.03 inter-scanner vs ±? seed noise);
  - look into the HR stain failure;
  - HEST "does no harm" check: IDC is on par (0.577 vs 0.578). GAP on all tasks is running;
    then run `bench/run python -m sipe.bench.hest --ckpt
    runs/2026-10-10_15-18-38_lxbelshark_20c2/checkpoints/last.ckpt --tag paired_vae_ysltwjoj
    --datasets '*'` (needs `8a28a6a` for `'*'`).
- DistMatchVAE: launch `uv run sipe fit --config configs/dist_match_vae.yaml` on the host
  (smoke run OK, but the moment terms hit their noise floor on 512-tile batches). In
  parallel, a stronger run: `--model.init_args.mmd_weight=10 --model.init_args.mean_weight=10
  --model.init_args.cov_weight=10`.
  Compare with PairedVAE on the val probe (z vs backbone) and retrieval. Watch
  `val/mmd` vs `val/mmd_backbone`, recon, active units and |z| for collapse.
- MathieuGAN `h65a5z5x`: finished, weak on val (probe gap 0.02, recon 0.195). PLISM/HEST:
  `bench/run python -m sipe.bench.plism --ckpt
  runs/2026-10-13_11-54-38_lxbelshark_36b0/checkpoints/last.ckpt --tag mathieu_gan_h65a5z5x`
  and the same with `sipe.bench.hest ... --datasets '*'`. An `adversarial_weight=0.1` run
  would likely trade even less removal for better recon; not a priority given the z probe.
- If adversaries on z come back: the Fader knobs not tried are a larger
  `adversary_weight` (3–5), fewer `adversary_steps`, and decaying the adversary's LR with
  the encoder's (the end of the Fader run was dominated by the adversary).
- Not started: H-optimus-1 training (memory at bs 512 untested; `unspecified_dim` choice
  open), and baselines beyond raw GAP.
- Each run (and the H0-mini GAP baseline) has one seed, so small differences
  (about ±0.03 on the val probe) are within noise.
