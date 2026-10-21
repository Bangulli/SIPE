# Findings: what the acquisition-shrinkage phase taught us (as of 2026-10-21)

Lessons that outlive single runs: about the benchmarks, the protocols, the data and what
did or did not work. Each one names its evidence (run dir, or the table in
`docs/progress.md` / `docs/experiment-log.md` it comes from). Paper status:
`docs/progress.md`; full experiment record: `docs/experiment-log.md`. Add a finding here
when a result changes how we should run or read experiments, not for every run.

## Evaluation and benchmarks

**1. SCORPION val is near ceiling and can disagree with PLISM, in both directions.** Val
top-1 differences of a few thousandths decided nothing that held on PLISM:
- Q2 (quadratic correction) passed its val gate, +0.0043 over G0 vs +0.0019 for the
  random-direction control, and was *below* G0 on PLISM, all pairs −0.005 [−0.006, −0.004]
  (`runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap`,
  `runs/bench/2026-10-15_00-34-59_lxbelshark_01e9_quadratic_Q2_m16`).
- G2 (mixture with cluster covariances) was +0.001 to +0.002 over G0 on val and below G0 in
  every PLISM row (`runs/bench/2026-10-14_08-02-19_lxbelshark_2d27_mixture_G2_K8`).
- The other way round: the untrained CGM init equals the trained V0-S on val (0.9808 vs
  0.9807, `runs/bench/2026-10-15_19-36-05_lxbelshark_eda5_scorpion_aug_attribution`), but is
  −0.020 [−0.023, −0.017] below it on PLISM
  (`runs/bench/2026-10-16_11-55-56_lxbelshark_d262_aug_cgminit_geo_5120`).
- Augmentation: CGM fits without augmentation are higher on SCORPION val and test (V1-S
  7e3c 0.9876 / 0.9780 vs 7b71 0.9852 / 0.9756; V0-S 3fa4 0.9832 vs a602 0.9807,
  `runs/bench/2026-10-17_11-24-40_lxbelshark_e198_scorpion_cgm_select_all`) and lower on PLISM (−0.002 [−0.002, −0.001] and −0.001 [−0.002, −0.000]). The
  clean-feature SCORPION protocol matches the no-aug training distribution.
- Use SCORPION val to select within one method family and as a sanity check; claim gains
  on PLISM per slide pair with the slide-bootstrap intervals. 5 val and 5 test WSIs.

**2. PLISM's HR stain is a protocol outlier.** HR is Harris hematoxylin applied overnight
(more than 24 h; checked on the images by the user, 2026-10-15). Its cross-stain pairs sit
near zero for every method (G0 0.007 vs 0.357 for the other inter-staining pairs,
`runs/bench/2026-10-15_05-38-58_claude-box_6ed5_plism_pairs_gap_g0_cgm_q2_vae/by_stain.csv`);
HRH (not overnight) behaves like the other stains. HR is in 609 of 4,095 pairs, so it
dilutes the inter-staining / joint / all means. Every PLISM table has a `drop_stain=HR` twin;
report it next to the full table, never instead. HR is mostly a slide-wide translation: the PLISM
condition-mean oracle (each slide's mean removed) lifts the inter-scanner HR pairs by
+0.086 [+0.013, +0.156] over V1-S and the HR pairs overall by +0.036 [+0.028, +0.043], while it
is at V1-S's level elsewhere (`runs/bench/2026-10-20_14-40-47_lxbelshark_8d2c_plism_oracle_condmean_gap`). "inter-scanner, HR" (21 pairs) is where
learned maps that break on an unusual stain show it (PairedVAE `ysltwjoj`).

**3. Protocols are not interchangeable.** (a) In-loop validation of a training module
(probes on WSI halves of val) is not the shared SCORPION-proxy protocol of `sipe.eval.proxy`
(probes trained on train, scored on val); compare CGM with G0 / the eraser only through
`cgm_select`. (b) Val probes are trained on train, test probes on val: probe numbers are
not comparable between val and test, retrieval is. (c) The PLISM top-1 is plismbench's
official metric; per-pair tables are merged from its `metrics.csv`, not recomputed.

**4. Scanner-probe accuracy is a diagnostic, not the target.** Lower probes did not mean
better retrieval: G2 lowers the test scanner probe more than G0 (logreg 0.516–0.558 vs 0.617)
without a PLISM gain; PairedVAE with geometric augmentation drops its z-probe from 0.689 to
0.362 and stays below the eraser and G0 on PLISM (`runs/2026-10-14_22-16-15_lxbelshark_44e5`,
`runs/bench/2026-10-15_02-56-10_lxbelshark_e50a_paired_vae_geometric_aug_lazztzda`). The goal
is to shrink scanner distances while keeping content, not to erase the scanner.

**14. HEST per-task changes come with tile-level pairing, and are not testable over
patients.** HEST has 2 to 6 patient folds per task, so the only per-task intervals are over
the 50 genes with the test patients fixed (`sipe.eval.hest_pairs`, block `hest_pairs`).
With that caveat, the tile-level maps (tile eraser, V0-S, V1-S) all move the same tasks
the same way vs GAP: READ −0.034 / −0.031 / −0.035, COAD +0.014 / +0.025 / +0.029, PRAD
+0.014 / +0.013 / +0.018, while the 10-task average stays within [−0.013, +0.012] of GAP
(`runs/bench/2026-10-17_11-26-14_lxbelshark_65e7_softshrink_gap_tile_lam0.4`,
`runs/bench/2026-10-15_15-08-23_lxbelshark_78e1_cgm_v0s_a602`,
`runs/bench/2026-10-15_07-47-25_lxbelshark_8823_cgm_v1s_7b71`). Read HEST as average
preserved plus a task trade-off of the pairing unit; don't attribute it to a model class.

**15. Backbones differ far more in PLISM robustness than maps do; compare within a
backbone.** Raw UNI2-h CLS is at 0.043 on all PLISM pairs vs 0.227 for H0-mini GAP, and our
`--posthoc` identity row reproduces the official plismbench UNI2-h output exactly
(0.317 / 0.073 / 0.015 / 0.043; `runs/bench/2026-10-21_02-32-59_lxbelshark_2f49_uni2h_cls_ref`
vs `../plism-benchmark/output/metrics/8139_tiles/uni2h`), so the gap is the backbone, not the
pipeline. With joint-shift pairs at the floor (0.015), CGM's gain over G0 on UNI2-h is mostly
inter-scanner (+0.041 vs +0.011 joint), the reverse of H0-mini (`docs/progress.md`, claim
10). Keep one block set per backbone (`include=` / `exclude=*uni2h*`) and report gains as
differences to that backbone's own raw and G0 rows. For a PLISM `--smoke` run, the slides
must give scanner, stain and joint pairs, or plismbench's aggregation fails
(`docs/benchmarks.md`).

**16. Load a backbone the way its authors do; resolution and position embedding change the
embedding.** CONCH v1.5 (`MahmoodLab/conchv1_5`) has no timm config, and its loaders
disagree: TRIDENT's `encoder_factory("conch_v15")` (recommended by the authors) resizes
224 px tiles to 448 px, while plismbench's `CONCHv15` feeds 224 px tiles. On a check of
SCORPION tiles (2026-10-21, not saved as a run), the 224 px and 448 px embeddings of the same
tile were far apart, and TRIDENT's position embedding (the checkpoint's 28x28 grid resampled
to 14x14 at load, interpolated back at each 448 px forward) changed nearest neighbours
against the checkpoint's own grid. We follow TRIDENT exactly (`src/sipe/model/conch.py`)
and test the match (`tests/test_bench_adapters.py::test_conch_v15_matches_trident`, bench
env). A plismbench leaderboard row of CONCH v1.5 is therefore not directly comparable to ours.

## Data and the acquisition nuisance

**5. The pairing unit matters more than the objective.** On PLISM all pairs: slide-level
eraser 0.281 < G0 (tile-level consensus) 0.294 < tile-level eraser 0.304 < region-level eraser
0.311 (PLISM table; `runs/bench/2026-10-13_21-54-02_lxbelshark_e26b_softshrink_gap_lam0.5`,
`runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1`,
`runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4`,
`runs/bench/2026-10-15_09-13-06_lxbelshark_0810_softshrink_gap_region_lam0.7`). Most of G0's
lead over the published eraser is pairing scale, not the consensus objective. Always state
the pairing unit; compare methods at the same unit. (ISBI fixes tile level, D2.)

**6. The scanner nuisance is not exchangeable, and its mean part is low-rank.** Per-scanner
offsets carry 0.295 of the train residual energy and transfer to unseen WSIs (they remove
0.259 of the val residual energy); they span 4 directions holding 0.849 of the residual
energy but only 0.042 of the content variance. Scanner-aware residual models beat the pooled
one on val by 0.157 [0.140, 0.174] nats/dim; not driven by P1000
(`runs/bench/2026-10-14_11-45-49_claude-box_f4c9_scorpion_scanner_offsets_gap`). This is why
the CGM's per-scanner likelihood (offsets as a random effect at inference) helps: the
CGM-init estimator is above G0 on the same data in every arm (+0.0019 to +0.0022 val).
The offsets also depend on the content: per scanner pair, a linear function of the other
views' content (top-128 PCs) predicts 0.158 raw / 0.057 whitened of the val energy that the
constant offsets leave (Δll +0.123 [0.114, 0.133] nats/dim, 5/5 WSIs; clusters reach only
0.054; not P1000; largest for AT2 and Philips;
`runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap`). Protocol:
test this with pair differences and content from the *other* views; regressing residuals
to the full consensus on the consensus finds content dependence where there is none when
the scanners' noise covariances differ.

**7. Content dependence is real on SCORPION but did not transfer.** Content-conditioned
residual models beat the global one on held-out WSIs (tile_all K=4 val +0.113 nats/dim,
mostly covariance shape; `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap`),
but the highest-energy cluster is mostly blank glass and tissue edges, and the content-aware
maps (G2, Q2) did not beat G0 on PLISM (finding 1). Background tiles are not quantified (no
tissue mask).

**8. P1000 registration offset (~1 H0-mini token) does not drive the diagnostics.** Dropping
P1000 leaves the content-dependence and offset results unchanged (finding 6–7 run dirs). In
LOSO, holding out AT2 or Philips costs the most (G0 unseen-pair top-1 −0.0113 / −0.0090 vs
the full fit), not P1000 (−0.0029)
(`runs/bench/2026-10-16_11-34-36_claude-box_bf0f_scorpion_loso_closed_form`).

## Models

**9. CGM's gain is the trained linear map, not augmentation or the nonlinearity.** On PLISM,
V0-S (linear encoder) has +0.025 of V1-S's +0.028 over G0; the nonlinear encoder adds
+0.003 / +0.004 (two seeds); a nonlinear generator (V2-S) adds +0.002, about a seed change
(+0.001), while moving the map more (nonlinear_rel 0.192 vs 0.072). Without augmentation,
V0-S loses −0.001 [−0.002, −0.000]; G0 fitted on augmented moments is −0.001
[−0.002, −0.001] vs G0 (`runs/bench/2026-10-16_13-09-01_lxbelshark_9a38_aug_g0_geo_all`); the
untrained init is −0.020 below trained V0-S (augmented, 5120 tiles), and −0.019 [−0.021, −0.016] on clean data with all train tiles (`runs/bench/2026-10-16_14-32-16_lxbelshark_8ef2_aug_cgminit_clean_all`). V1-S without augmentation: −0.002 [−0.002, −0.001] vs V1-S
(`runs/bench/2026-10-16_14-07-40_lxbelshark_3c28_cgm_v1s_noaug_7e3c`), a seed-sized effect. Geo and zoom
augmentation even lower the closed-form estimators on clean SCORPION val (G0 0.9765 vs
0.9787); dihedral is neutral (H0-mini GAP close to dihedral-invariant). Evidence: ablation
blocks and augmentation-attribution table in `docs/progress.md`. The exchangeable linear V0
(no scanner offsets or per-scanner noise) is +0.022 [+0.018, +0.025] over G0 and only
−0.003 [−0.003, −0.002] below V0-S (`runs/bench/2026-10-16_22-17-16_lxbelshark_d926_cgm_v0_5ee4`),
and the exchangeable nonlinear V1 is −0.001 [−0.002, −0.000] below V1-S, about the seed
spread (`runs/bench/2026-10-17_00-15-28_lxbelshark_7a78_cgm_v1_73af`):
the training, not the per-scanner likelihood, carries the gain.

**10b. A better-specified nuisance likelihood can make the map worse.** Adding a
content-dependent scanner shift δ_k + D_k z (V1-S+D, rank 128) to the CGM likelihood lowers
val NLL (0.0867 vs 0.0985) but drops PLISM by −0.005 [−0.006, −0.004] vs V1-S and
−0.002 [−0.003, −0.001] vs V0-S on all pairs; only the inter-scanner pairs stay at V0-S level
(+0.001 [+0.000, +0.002]) (`runs/bench/2026-10-16_20-17-12_lxbelshark_d531_cgm_v1s_shift_e3c7`), although the
diagnostic found real content dependence (finding 6). Plausible, not verified: what the
likelihood explains as nuisance, the scanner-blind map no longer has to remove. Rank 32
does not fix it: −0.004 [−0.004, −0.003] vs V1-S
(`runs/bench/2026-10-17_01-16-13_lxbelshark_1a37_cgm_v1s_shift_aa1c`). Judge a
likelihood term by the map it trains, not by the NLL.

**10. Unpaired methods did not remove the scanner; paired supervision did.** CATS-phase
result (archive): unpaired models stayed at the probe noise floor; PairedVAE's removal comes
from its pair-alignment loss (translation-only ablation `runs/2026-10-13_19-12-17_lxbelshark_5465`),
and its SCORPION val gain appears almost before training. Closed-form paired estimators
(eraser, G0) beat PairedVAE on PLISM in every row.

## Engineering

**11. Paired augmentation relies on seed replay.** The scanner views of a tile share their
transform because the same torch seed is set before each view (`PairedSCORPIONDataset`,
feature caches: seed = crc32(pair_id|copy|seed)). Correct only while every random op draws
from the torch RNG the same number of times; albumentations / numpy / `random` ops would
desync the views silently (test: `test_paired_views_share_geometric_augmentation`).

**12. Host scripts that run fits concurrently must name their run dir.** Taking "the newest
new dir in runs/" picked a concurrent fit's dir (2026-10-15, no-aug V0-S vs V2-S). Fit
scripts create the run dir with `sipe.cli.run_dir_name` / `write_git_info` and pass it via
`SIPE_RUN_DIR`.

**13. In claude-box, full probe evaluation is slow.** `sipe.eval.proxy.summarize` on the
full cache took more than 20 min for 2 maps under CPU load; run `cgm_select` / sweeps on the
host, use `--retrieval-only` or smoke caches here. Shared memory is 64 MB (`--workers 0`).
