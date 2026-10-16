# Experiment log: acquisition-shrinkage roadmap (as of 2026-10-16)

Full record of the roadmap experiments (`docs/roadmap.md`), moved out of `docs/progress.md` on
2026-10-16 so that progress.md can focus on the ISBI paper. Sections 1–3 were moved verbatim
(their "above" / "below" references point to the tables in section 4 of this file or to the
paper tables in `docs/progress.md`). Lessons that outlive single runs: `docs/findings.md`.
Generated tables: `uv run python -m sipe.eval.results_tables --write` refreshes the blocks in
this file and in progress.md; never edit inside a block.

Where to write what:
- a run or decision that matters for the ISBI paper → `docs/progress.md` (claims, runs in
  flight, paper tables);
- any other run, reading or decision (D0–D6) → here: a row or reading in section 1–2, an
  open point in section 3, a table block in section 4;
- a general lesson (protocol, benchmark quirk, negative result) → `docs/findings.md`.

## 1. Roadmap status and decisions

| step (roadmap) | status | evidence (run dir) | decision |
|---|---|---|---|
| Exp 0: soft shrinkage, new λ rule, SCORPION test | done: λ=0.5 | `runs/bench/2026-10-13_17-45-08_lxbelshark_db10_scorpion_softshrink_h0mini_gap_gain95` | – |
| Exp 0: PLISM / HEST for the eraser | done (λ=0.5, `--posthoc`) | PLISM `runs/bench/2026-10-13_21-54-02_lxbelshark_e26b_softshrink_gap_lam0.5`, HEST `runs/bench/2026-10-13_23-15-00_lxbelshark_c5a0_softshrink_gap_lam0.5` | D0 taken 2026-10-13: shrinkage beats PairedVAE on PLISM (every row) and keeps the HEST average → main bar to beat (one fit; per-pair/HR breakdown pending) |
| Exp 1: content-dependence diagnostic | done (`sipe.eval.content_dependence`); both D1 checks done 2026-10-13 (see Exp 3 reading) | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` (content table; `cluster_stats.csv`); tiles: `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap/inspection/` | D1 taken (user asked for Exp 3): clear, stable content dependence → Exp 3 first, Exp 4 as competitor |
| Exp 2: consensus covariance (2A), affine regression (2B), pairing-scale control (2C) | 2A covered by G0 of Exp 3 (α grid; 2A=2B at α=1 is a unit test); 2C implemented (`soft_shrinkage_sweep --deltas slide|region|tile`, 2026-10-14); tile sweep done (`runs/bench/2026-10-14_09-21-09_lxbelshark_c54b_scorpion_softshrink_gap_tile`, λ=0.4) and region sweep done (`runs/bench/2026-10-14_19-02-29_lxbelshark_71bb_scorpion_softshrink_gap_region`, λ=0.7), both with val selection + test (softshrink table, refreshed 2026-10-14); not yet read against G0 | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` (mixture table, G0 row) | D2 pending: G0 (tile-level consensus target) beats the eraser on SCORPION and on PLISM (all four rows; PLISM `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1`, one fit, no intervals yet) and keeps the HEST average (0.363 vs eraser 0.364 and GAP 0.364, `runs/bench/2026-10-14_10-08-10_lxbelshark_41cf_mixture_G0_K1`); needs 2C before calling it 2026-10-15: on SCORPION (val selection, `selected.json` of each sweep) the tile eraser almost matches G0 (val top-1 0.978 vs 0.979) and the region eraser is above it (0.981), so most of G0's lead over the slide eraser is pairing scale; D2 is defined on external retrieval → waiting for PLISM of the tile (λ=0.4) and region (λ=0.7) eraser maps (user, `scripts/bench_d2_pairing_scale.sh`, gitignored; tags `softshrink_gap_tile_lam0.4`, `softshrink_gap_region_lam0.7`), then add them to the `plism_pairs` block. PLISM done 2026-10-15: region eraser (`runs/bench/2026-10-15_09-13-06_lxbelshark_0810_softshrink_gap_region_lam0.7`) 0.896 / 0.321 / 0.261 / 0.311, tile eraser (`runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4`) 0.889 / 0.310 / 0.254 / 0.304 (PLISM table). Per pair vs the region eraser (generated block): G0 −0.017 [−0.020, −0.015] on all pairs and below in every group; tile eraser −0.007 [−0.009, −0.005]. Order on PLISM: region eraser > tile eraser > G0 > slide eraser; the eraser objective at tile level is not worse than the consensus objective at the same level | D2 TAKEN 2026-10-15 (user), for the ISBI paper: pairing is fixed at tile level (the pairing unit of G0 and CGM). At that level the eraser objective is about equal to G0 on PLISM (tile eraser +0.010 over G0 on all pairs), so G0 stays the linear base because it is CGM's linear limit; the original eraser is reported as published (slide-level pairing) as the external baseline. The pairing-scale result (region eraser 0.311 on all pairs, above G0 and the tile eraser) is out of the ISBI scope and will be re-evaluated in the broader experimental setup of the follow-up paper |
| Exp 3: Gaussian mixture G0/G1/G2-scale/G2, K=2/4/8 | SCORPION val + test done (`sipe.eval.mixture_sweep`); PLISM G0 done (`runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1`; its git.txt says 9ee82ff dirty with the mixture code untracked = code of 099e76e): above the eraser in all four PLISM rows (PLISM table); PLISM G2_K8 done (`runs/bench/2026-10-14_08-02-19_lxbelshark_2d27_mixture_G2_K8`): below G0 in all four PLISM rows, about at the eraser's level; PLISM G2_K4 done (`runs/bench/2026-10-14_08-54-57_lxbelshark_38d3_mixture_G2_K4`): also below G0 in all four rows; HEST G0 done (`runs/bench/2026-10-14_10-08-10_lxbelshark_41cf_mixture_G0_K1`): average 0.363, the eraser and GAP 0.364; HEST G2_K8 done (`runs/bench/2026-10-14_11-53-10_lxbelshark_d3bf_mixture_G2_K8`): average 0.364 | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` (mixture table) | D3 taken 2026-10-14 (user): G2 > G1 ≈ G2-scale on SCORPION, but G2 − G0 is small there and G2_K8 and G2_K4 are both below G0 on PLISM → don't promote G2, no Exp 5; G0 is the candidate linear bar (HEST done 2026-10-14; intervals pending); next = Exp 4 |
| Diagnostic: scanner offsets / exchangeability (for Exp 6 design) | done 2026-10-14 (`sipe.eval.scanner_offsets`) | `runs/bench/2026-10-14_11-45-49_claude-box_f4c9_scorpion_scanner_offsets_gap` (offsets table) | – (informs Exp 6; no gate) |
| Diagnostic: content-dependent scanner offsets (does δ_k depend on the tissue? for a per-scanner generator L_k = L + D_k in the CGM) | done 2026-10-16 (`sipe.eval.content_offsets`, full grid, tile_all + tile_noP1000; reading in section 3) | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` (content_offsets table, `per_scanner.csv`, `per_cluster.csv`); smoke `runs/bench/2026-10-16_13-59-16_claude-box_42ad_scorpion_content_offsets_smoke` | – (gate for a "V0-S + D_k" CGM arm: passes on SCORPION; no ISBI decision) |
| Exp 7 follow-up: V1-S+D, content-dependent scanner shift δ_k + D_k z in the CGM likelihood | implemented 2026-10-16 (`ContentShiftCGM`, `configs/consensus_gen_shift.yaml`, rank 128); done: fit `runs/2026-10-16_19-27-29_lxbelshark_e3c7`, PLISM `runs/bench/2026-10-16_20-17-12_lxbelshark_d531_cgm_v1s_shift_e3c7` | below V1-S on PLISM (all pairs −0.005 [−0.006, −0.004]); not pursued | decides whether content-dependent shifts are pursued (vs V1-S 7b71 on PLISM); not in the ISBI scope |
| Exp 4: quadratic correction Q1/Q2 | full sweep done (`runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap`; val selection, probes, test; quadratic table). Val gate passed for Q2: Q2 m=16 val Δtop-1 vs G0 +0.0043 vs Q2-rand m=16 +0.0019 (Q2 m=8 +0.0024 vs rand +0.0008); Q1 ≈ G0 (+0.0001/+0.0002); selected γ=0.01 not at the grid edge (`selected.json`). Test agrees in order: Q2 m=16 0.971, Q2-rand m=16 0.968, G0 0.965 (quadratic table). PLISM done 2026-10-15: Q2 m=16 (`runs/bench/2026-10-15_00-34-59_lxbelshark_01e9_quadratic_Q2_m16`) 0.883 / 0.299 / 0.237 / 0.288, *below* G0 (0.887 / 0.303 / 0.243 / 0.294) in all four rows; Q2-rand m=16 (`runs/bench/2026-10-15_01-47-42_lxbelshark_5dd4_quadratic_Q2-rand_m16`) 0.888 / 0.306 / 0.246 / 0.297, slightly above G0 (PLISM table). Per pair vs G0 (PLISM per-pair table, slide-bootstrap 95% CI): Q2 all-pairs Δtop-1 −0.005 [−0.006, −0.004], better on 4% of pairs; Q2-rand +0.004 [+0.003, +0.004]. Base = consensus regression (user decision 2026-10-14). One fit, one split | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` (quadratic table) | D4 TAKEN 2026-10-15 (user): Q2 below G0 on PLISM in every row → the quadratic SCORPION gain does not transfer; keep G0, Q2 not promoted, no HEST for Q2. To be re-evaluated in the broader experimental setup of the follow-up paper |
| Exp 5/6: conditional covariance / constrained VAE | gated by D3/D4 | – | – |
| Exp 7: consensus generative model (CGM, `docs/archive/experiment7-draft.md` Step 3), V1-S arm, backbone in the loop | implemented 2026-10-14 (`sipe.model.consensus_gen`, `sipe.training.consensus_gen_module`, `configs/consensus_gen.yaml`); first full fit done 2026-10-14 (`runs/2026-10-14_22-30-38_lxbelshark_7b71`, W&B o2snd9bm, 6000 steps, commit dd32d0a clean; CGM table). In-loop val top-1 0.985 (last-10 mean) vs backbone 0.887; first validation (step 99, `wandb_history`) 0.9795, so training added about +0.006 over the near-linear start. For orientation only (different protocol): G0 val 0.979 and Q2 m=16 0.983 in the quadratic table. `val/nll` rose from 0.040 (step 99) to 0.0985 while `val/kl` fell; train nll steps 0.02–0.08 → likelihood generalization gap. hyperparameters not selected on val; one seed. PLISM done 2026-10-15 (`runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71`, `--posthoc map_cgm.pt`): 0.899 / 0.321 / 0.273 / 0.321, above G0 (0.887 / 0.303 / 0.243 / 0.294) in all four rows and the best PLISM row so far (PLISM table). Per pair vs G0 (PLISM per-pair table, slide-bootstrap 95% CI): all pairs Δtop-1 +0.028 [+0.024, +0.032], better on 98% of pairs; inter-scanner +0.012 [+0.007, +0.020]; inter-staining +0.018 [+0.015, +0.022]; joint +0.031 [+0.026, +0.035]; HR inter-scanner +0.001 [−0.007, +0.012] (21 pairs). Caveats: val selection done 2026-10-15 (`cgm_select`, V1-S selected; CGM selection table), trained on augmented images rather than the shared feature cache, one seed. HEST done 2026-10-15 (`runs/bench/2026-10-15_07-47-25_lxbelshark_8823_cgm_v1s_7b71`; git.txt: only docs/progress.md modified, code = dd7e212): average 0.363, same as G0 (0.363) and the eraser / GAP (0.364) → preservation holds on average (HEST table). Per task it moves more than G0: COAD 0.285 vs GAP 0.256, PRAD 0.389 vs 0.371, but READ 0.167 vs 0.202, SKCM 0.573 vs 0.594, HCC 0.060 vs 0.074 (fold stds 0.01–0.09, one fit). Ablation fits done 2026-10-15 (6000 steps each; git.txt dd7e212 with only docs/progress.md and the eval module `plism_pairs.py` modified, training code unchanged; CGM table): V0-S linear encoder `runs/2026-10-15_08-50-48_lxbelshark_a602` (nbeqfgol), in-loop val top-1 0.981 (last-10 mean), nonlinear_rel 0.000; V1-S seed 43 `runs/2026-10-15_08-51-41_lxbelshark_8ed0` (prb6pqng), 0.985, the same as 7b71 (0.985). First validation (step 99, `wandb_history`): 0.9796 (V0-S), 0.9795 (seed 43), 0.9795 (7b71) → training adds about +0.006 with the nonlinear encoder (both seeds) and about +0.001 with the linear one. PLISM done 2026-10-15 (PLISM table; per-pair blocks vs G0, vs the region eraser and V1-S vs V0-S): seed 43 (`runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0`) 0.899 / 0.322 / 0.274 / 0.322 reproduces 7b71 (0.899 / 0.321 / 0.273 / 0.321). V0-S (`runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602`) 0.897 / 0.319 / 0.270 / 0.318: the linear CGM already gets +0.025 [+0.021, +0.028] of the +0.028 over G0; the nonlinear encoder adds +0.003 [+0.003, +0.004] (7b71) and +0.004 [+0.003, +0.004] (seed 43) on all pairs vs V0-S, small but positive in every group for both seeds. Against the region eraser (the D2 anchor candidate): 7b71 +0.011 [+0.008, +0.013] on all pairs, mostly from scanner + staining pairs (+0.013); inter-scanner +0.004 [−0.000, +0.007]; inter-staining +0.001 [−0.002, +0.003]; HR inter-scanner −0.013 [−0.030, −0.000] (21 pairs). V0-S vs the region eraser: +0.007 [+0.005, +0.010] on all pairs. Reading: most of CGM's PLISM gain is a better *linear* map (trained on augmented images through the backbone with the per-scanner likelihood), not the nonlinearity; the gain over the strongest linear eraser sits in the joint scanner + staining shift. Draft Steps 0–2 skipped | `runs/2026-10-14_22-30-38_lxbelshark_7b71` (CGM table) | – |

## 2. Experiment readings

Notes carried over from the CATS phase (details and run dirs in the archive):
- Soft shrinkage at λ=0.5 nearly matches PairedVAE on the SCORPION proxy (see the
  softshrink and scorpion tables). Val and test scanner probes use different training sets
  (train vs val), so compare probe numbers within one protocol only.
- PairedVAE's scanner removal comes from its pair-alignment loss (translation-only ablation
  `runs/2026-10-13_19-12-17_lxbelshark_5465`). Its val retrieval gain over the backbone
  appears almost before training, so SCORPION val retrieval alone says little.

Experiment 0, external (D0; PLISM and HEST tables):
- The λ=0.5 eraser is above PairedVAE and GAP in every PLISM row (inter-scanner 0.873 vs
  0.839 and 0.811; all 0.281 vs 0.267 and 0.227;
  `runs/bench/2026-10-13_21-54-02_lxbelshark_e26b_softshrink_gap_lam0.5`). HEST average
  0.364, the same as GAP and PairedVAE (`runs/bench/2026-10-13_23-15-00_lxbelshark_c5a0_softshrink_gap_lam0.5`);
  per task within 0.012 of GAP (largest: COAD 0.268 vs 0.256, PRAD 0.382 vs 0.371).
- Not yet done: the per-slide-pair / HR-stain breakdown and group-aware intervals; one fit.

HEST reading for G0 and G2_K8 (2026-10-14; HEST table, `runs/bench/2026-10-14_10-08-10_lxbelshark_41cf_mixture_G0_K1`, `runs/bench/2026-10-14_11-53-10_lxbelshark_d3bf_mixture_G2_K8`):
- G0 keeps the HEST average: 0.363, against 0.364 for the eraser
  (`runs/bench/2026-10-13_23-15-00_lxbelshark_c5a0_softshrink_gap_lam0.5`) and for GAP
  (`runs/bench/2026-10-13_11-03-53_lxbelshark_8f0f_h0_mini_gap_ref`). Largest per-task
  change vs GAP: READ 0.178 vs 0.202, but the READ fold sd is 0.080 (GAP 0.049); PRAD
  0.383 vs 0.371. G2_K8: average 0.364, READ 0.184.
- With the PLISM result (G0 above the eraser in all four rows), G0 is better on robustness
  and equal on utility: the linear bar for Exp 4 and later. Still one fit, no intervals.

Exp 4 partial reading (val sweep only, `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` `sweep_val.csv`; selection, probes and test not
written yet, so nothing is selected):
- G0 (γ = inf) val top-1 0.978735. Q1 stays at G0 (best 0.978857, m=16 γ=0.1).
- Q2 rises as γ falls: m=16 0.983122 at γ=0.001, m=8 0.981082 at γ=0.01. The Q2-rand
  control also rises, less: m=16 0.980653, m=8 0.979469 (γ=0.001). So part of the Q2 gain
  is generic extra capacity, part is the content × nuisance structure.
- Val consensus MSE: Q2 m=16 0.0143149 (γ=0.01) vs G0 0.0151742; Q2-rand m=16 best
  0.0149636 (γ=0.1).
- The best Q2 top-1 is at the smallest γ of the grid (0.001): the grid edge, less
  regularization might go further; the selection rule (largest γ at 95% of the best gain)
  will be read from `selected.json`.

Scanner-offset reading (2026-10-14; all values from `runs/bench/2026-10-14_11-45-49_claude-box_f4c9_scorpion_scanner_offsets_gap`,
offsets table, `likelihood.csv`, `scanner_stats.csv`). Question: is G0's pooled,
exchangeable nuisance N_0 an adequate model of the acquisition residual?
- Global per-scanner offsets are large in raw energy: 0.295 of the train residual energy
  (tile_all); subtracting the train offsets removes 0.259 of the val residual energy, so
  they carry over to unseen WSIs. Slide × scanner and region × scanner offsets are small
  (0.053, 0.069); the tile level is 0.583.
- They are low-rank: they span S - 1 = 4 directions, where they explain 0.849 of the
  residual energy, and those 4 directions hold only 0.042 of the content (consensus)
  variance. Hence their whitened share is only 0.005. A linear map can shrink that span at
  small content cost, which G0 presumably already does (not measured: G0's gain in that
  span).
- Scanner-aware residual models beat the exchangeable one on val in all 5 WSIs: per-scanner
  offset 0.118, per-scanner covariance 0.109, both 0.157 [0.140, 0.174] nats/dim (top-32
  residual subspace, the same subspace definition as Exp 1). The covariance shape adds to the
  offsets, so not all of the non-exchangeable part lies in the offset span.
- Without the label, the equal-weight scanner mixture still gets 0.110 [0.095, 0.125]; the
  component argmax recovers the scanner for 0.992 of val residuals. This is given the paired
  consensus; from a single raw view the scanner probe is 0.956 (scorpion table, test).
- Not driven by P1000: without it, offsets 0.330 of raw energy, both 0.154. The largest
  offsets are AT2 and Philips (offset energy share 0.113 and 0.091 vs 0.027–0.033,
  tile_all).
- So exchangeability is rejected as a likelihood model, by a margin of the same order as
  Exp 1's content dependence (tile_all K=8 val 0.137). Whether a scanner-aware map
  improves *retrieval* over G0 is untested. The cheapest test is closed form: the Exp 3
  mixture with components = scanners (per-scanner mean δ_k and N_k, single-view gating
  p(k|x)), before any VAE. Caveat for PLISM: unseen scanners match none of the modes, so
  the map must fall back to G0.

Experiment 3 reading (all values from `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap`, mixture table, `likelihood.csv`,
`gating.csv`, `inspection/`):
- D1 check 1, the high-energy tile cluster (K=4, 494 train tiles): mostly **blank glass
  and tissue edges** (background), not a tissue type; the other clusters are tissue
  (`inspection/*.png`, visual check). So part of the Exp 1 "content dependence" is
  background vs tissue. PLISM tiles are tissue, so gains driven by background tiles
  would not transfer. Not yet quantified (no tissue mask).
- D1 check 2, single-view gating: argmax p(k|x) matches the consensus cluster for 0.949 /
  0.915 / 0.879 of val views (G2, K = 2 / 4 / 8), and all 5 views of a tile gate to the
  same cluster for 0.937 / 0.892 / 0.822 of tiles. Usable, but at K=8 about one tile in
  five gets scanner-dependent gates.
- G0 (global Gaussian, tile-level consensus target; = Exp 2A) already beats the eraser on
  SCORPION: val top-1 0.979 vs 0.973, test 0.965 vs 0.959, region probe 0.986 on test for
  both. The bar on SCORPION is now G0, not the eraser (supervision differs: tile pairs vs
  slide-level deltas, Exp 2C).
- G1 (multimodal content, shared N_0) does not help: val top-1 0.978 / 0.973 / 0.973 vs
  G0 0.979. G2-scale (ML scale per cluster) is the same as G1 (0.978 / 0.974 / 0.973).
- G2 (cluster-specific N_k, γ = 0.10 / 0.10 / 0.25) is the best mixture: highest val
  residual likelihood (1.128 / 1.177 / 1.193 vs 1.087 for G1 and G0, nats/dim) and val
  top-1 0.980 / 0.980 / 0.981 (0.980 / 0.982 / 0.983 at α=1), test 0.967 / 0.965 /
  0.968, region probe not lower (val 0.987, test 0.987–0.989). The gain needs covariance
  shape: G2-scale does not get it, as Exp 1 predicted.
- But the margin over linear G0 is small: test top-1 +0.000 to +0.003, val +0.001 to
  +0.002 at the selected α. One fit seed, one split; SCORPION retrieval is near ceiling.
  G2 lowers the scanner probes more than G0 (test logreg 0.516–0.558 vs 0.617), which is a
  diagnostic, not the target.
- The reference eraser evaluated inside this run reproduces the eraser sweep's val top-1
  at λ=0.5 (0.9731 in `summary_val.csv` and in the sweep's `sweep_val.csv`); its test
  probes differ from the softshrink table by ≤ 0.008 (MLP/logreg refits).
- Superseded runs, deleted: two earlier starts of the same sweep, stopped before any
  evaluation (slow CPU likelihoods; then the G2-scale control was switched from the trace
  ratio to the ML scale, which the trace ratio fell below G1 even on train).

Experiment 1 reading (all values from `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap`, content table or
`cluster_stats.csv`):
- The content-conditioned residual model beats the global one on held-out WSIs in every
  setting and K: all 5 CV folds > 0, val WSI-bootstrap intervals above 0, and the
  shuffled-label control stays near 0 (e.g. tile_all K=4: CV 0.101 ± 0.009, val 0.113
  [0.078, 0.147], shuffled val -0.008 nats/dim). The gain grows with K (2 → 8).
- It is mostly **shape**, not magnitude: the scale-only control gets 0.020 of the 0.113 at
  tile_all K=4 (val), and about 0 at region level.
- Not registration-driven as far as we can test: dropping P1000 leaves the numbers
  unchanged (tile_noP1000 K=4: val 0.124), and spatial pooling to regions keeps the effect
  (region_all K=4: val 0.088 [0.046, 0.132]).
- Where it lives, tile_all K=4: a small cluster (494 train / 92 val units) with residual
  energy ratio 1.58 (val 1.67), only 0.074 (val 0.086) of its energy in the top-10
  nuisance directions of the eraser's B vs 0.32–0.46 elsewhere, and little overlap with
  the global top-5 residual subspace (0.059; val 0.134). Its P1000 energy share is the
  highest (0.282 vs 0.167–0.196). The other clusters differ mainly in energy (ratios
  0.76–1.25, replicated on val).
- Caveats: the clusters are assigned from the paired consensus, which an inference map
  does not have (Exp 3 needs single-view gating); the model is zero-mean, so
  content-dependent scanner offsets count as "shape"; val has 5 WSIs (50 regions), so
  region-level per-cluster stats on val rest on 4–20 regions.
- First run of the same code without the scale-only columns
  (`runs/bench/2026-10-13_19-59-10_claude-box_40f5_scorpion_content_dep_gap`, identical
  numbers otherwise) is superseded and excluded from the table.

### PairedVAE augmentation check (reading of the PairedVAE table, section 4)

Same in-loop protocol as the CGM table (z_gap = GAP(mu)). `ysltwjoj` =
`configs/paired_vae.yaml` (legacy augmentations); `lazztzda` =
`configs/paired_vae_geometric_aug.yaml` (flips, rot90, zoom only); `8271lf26` =
translation-only ablation (`align_weight: 0`, archived CATS phase).
Reading (one seed each): geometric augmentation gives val top-1 0.979 vs 0.977
(`ysltwjoj`), a small gain; z_gap scanner probe 0.362 vs 0.689 and align loss
0.079 vs 0.241: the scanner views end up much closer in z (why is not tested);
s probe 0.974 vs 0.933. PLISM done 2026-10-15
(`runs/bench/2026-10-15_02-56-10_lxbelshark_e50a_paired_vae_geometric_aug_lazztzda`, PLISM
table): inter-scanner 0.861 vs `ysltwjoj` 0.839, other rows about equal (0.287 / 0.217 / 0.269
vs 0.293 / 0.215 / 0.267); still below the eraser and G0 in every row. The inter-scanner gain is
mostly the HR pairs: HR inter-scanner top-1 0.626 vs 0.486 (PLISM per-pair table). One seed each.

### Exp 7 CGM: detailed history (2026-10-14 to 2026-10-16)

The ISBI-facing summary of these items is in `docs/progress.md` (section 2, claims).

- Does CGM's gain over G0 come from data augmentation? (plan 2026-10-15; code done, runs
  pending on the host, see "Commands" at the end of this item.)
  CGM differs from G0 in four ways at once: (a) its moments come from *augmented* train
  views (flips, rot90, zoom, through the backbone), while G0 uses the clean cache;
  (b) a different closed-form estimator at init (linear-Gaussian posterior with per-scanner
  offsets as a random effect, ridge floor; `ConsensusGenModel.init_from_moments`);
  (c) fewer views (40 batches = 25,600 views, `init/num_views` in W&B); (d) ELBO training.
  Evidence that (d) is small for the linear arm: in-loop `val/retrieval_top1_cgm` at the
  first validation (step 99) is 0.980 for V0-S (`runs/2026-10-15_08-50-48_lxbelshark_a602`)
  and 0.980 for V1-S (`runs/2026-10-14_22-30-38_lxbelshark_7b71`), against 0.981 / 0.985 at
  the end (CGM training table; read with `wandb_history`). So most of V0-S's PLISM gain over
  G0 is probably already in the init map, i.e. in (a)–(c). SCORPION val hardly separates
  these maps, so the readout is PLISM per pair vs G0 (explanatory ablation, nothing is
  selected on it). Plan, cheapest first:
  1. Augmented train caches: `sipe.features.extract --policy geo|dihedral|zoom|none
     --copies K` (train split; the scanner views of a tile copy share the transform; every
     view through the whole encoder). Smoke: `none` on one slide reproduces the GAP cache
     to fp16 level (`runs/bench/2026-10-15_15-51-30_claude-box_6e0c_smoke_augcache_none`,
     batch 64 vs 128).
  2. Closed-form attribution, no training (`sipe.eval.aug_attribution`, block `augattr`):
     {G0, CGM-init} × {clean, geo, dihedral, zoom} × {5120 tiles (= the CGM init), every
     tile once, every copy}, the same tiles in every arm; V0-S / V1-S maps in the same
     table; val, retrieval only. Then PLISM for CGM-init|geo|5120 (does training add
     anything?), G0|geo|all (augmentation alone?) and CGM-init|clean|all (estimator alone?).
  3. Only if the init arms do not reach V0-S: V0-S and V1-S fits without augmentation
     (`--data.init_args.augmentations=[] --data.init_args.horizontal_flip=false
     --data.init_args.vertical_flip=false`), user command.
  First hint (smoke run `runs/bench/2026-10-15_16-02-28_claude-box_e363_scorpion_smoke_aug_attribution`,
  `summary_val.csv`, retrieval only; its geo cache has 2 slides, so only the clean arms,
  which use the full train cache, are meaningful): CGM-init|clean|all val top-1 0.9808 vs
  V0-S_a602 0.9807 and G0|clean|all 0.9787 (= the mixture-sweep G0). On SCORPION val the
  V0-S estimator reaches V0-S on clean data without training, so the estimator, not the
  augmentation, may carry the gain; PLISM of `CGM-init|clean|all` decides (step 4 script).
  Also: `sipe.eval.cgm_select` (block `cgm`) puts V0-S / V1-S on the shared SCORPION
  protocol (val selection over G0, test read once), the missing ISBI piece.
  Reading: G0-geo ≈ V0-S → the gain is augmented moments, and G0 on augmented data is the
  simpler model to report; CGM-init-clean ≈ V0-S and G0-geo ≈ G0 → estimator, not
  augmentation; all init arms ≈ G0 → training (step 3 decides augmentation's role there).
  `dihedral` vs `zoom` says which part matters (GAP may be nearly dihedral-invariant).
  Commands (host, gitignored scripts, in order): `scripts/aug_study_1_extract.sh`;
  `scripts/aug_study_2_attribution.sh <geo> <dihedral> <zoom cache dirs>`;
  `scripts/aug_study_3_cgm_select.sh` (independent); `scripts/aug_study_4_plism.sh
  <aug_attribution run dir>`. After each: `results_tables --write`; after step 4, add the
  three `aug_*` tags to the `plism_pairs` blocks with ref=mixture_G0_K1 and ref=cgm_v0s_a602.
  Status 2026-10-15: step 1 done (`runs/bench/2026-10-15_18-20-26_lxbelshark_cfad_augcache_geo`,
  `runs/bench/2026-10-15_18-25-53_lxbelshark_490a_augcache_dihedral`,
  `runs/bench/2026-10-15_18-37-41_lxbelshark_54ba_augcache_zoom`; `meta_train.csv` row counts
  372,400 / 744,800 / 372,400 = 4 / 8 / 4 copies x 93,100 views, 18,620 tiles each). Step 3
  done (`runs/bench/2026-10-15_18-22-36_lxbelshark_aae8_scorpion_cgm_select`, CGM selection
  table): V1-S selected (`selected.json`, no G0 fallback, region guard not violated); val
  top-1 0.9852 (both seeds) vs V0-S 0.9807 and G0 0.9787; test top-1 0.9756 (both seeds) vs
  V0-S 0.9682 and G0 0.9645, test region probe 0.987 vs G0 0.986. So on the shared SCORPION
  protocol the nonlinear encoder carries most of the gain over G0 (+0.0065 val, V0-S
  +0.0020), unlike PLISM where V0-S has most of it; one V0-S seed.
  Step 2 done 2026-10-15 (`runs/bench/2026-10-15_19-36-05_lxbelshark_eda5_scorpion_aug_attribution`,
  augmentation-attribution table, val retrieval only): CGM-init|clean|all 0.9808 = V0-S_a602
  0.9807, i.e. the V0-S estimator on clean data, without training and without augmentation,
  already reaches V0-S on SCORPION val; CGM-init is above G0 on the same data in every arm
  (+0.0019 to +0.0022). Augmentation does not help on clean val: geo and zoom lower both
  estimators (G0|geo|all 0.9765, G0|zoom|all 0.9765 vs G0|clean|all 0.9787), dihedral is
  neutral (G0|dihedral|all 0.9787; GAP close to dihedral-invariant). The number of views
  hardly matters (5120 vs all within 0.0006 for clean and dihedral). Not a verdict on
  augmentation: PLISM (step 4) is the readout, not launched yet.
  V2-S fit + PLISM done 2026-10-15 (`runs/2026-10-15_19-48-26_lxbelshark_920f`, W&B d7ih8uaz,
  `scripts/cgm_v2s_fit.sh`; PLISM `runs/bench/2026-10-15_21-23-35_lxbelshark_60b4_cgm_v2s_920f`):
  in-loop val top-1 0.985 = V1-S, nonlinear_rel 0.192 vs 0.072 for 7b71 (CGM training
  table). PLISM 0.898 / 0.322 / 0.275 / 0.323 (PLISM table); per pair vs V1-S 7b71 (new block
  "CGM nonlinear generator vs V1-S"): all pairs +0.002 [+0.001, +0.002], inter-scanner −0.001
  [−0.002, −0.000], HR inter-scanner −0.006 [−0.014, −0.001]; seed 43 vs 7b71 is +0.001 on all
  pairs. So the nonlinear generator moves the map much more for a gain the size of a seed
  change → V1-S stays the simpler model; V2-S not yet in a `cgm_select` run (SCORPION val/test).
  No-aug fits: the V0-S fit finished (`runs/2026-10-15_19-42-06_lxbelshark_3fa4`, W&B ryqlwzeu,
  in-loop val top-1 0.983 vs 0.981 for the augmented a602; CGM training table), but the
  script then stopped before its PLISM and before the V1-S fit: it took the newest new
  `runs/` dir as the fit dir, and the concurrent V2-S fit had created one (fixed: the scripts
  now create the run dir and pass it via `SIPE_RUN_DIR`). Remaining (host): PLISM of 3fa4,
  V1-S no-aug fit + PLISM, step 4, a `cgm_select` run with 3fa4, 920f and the V1-S no-aug fit.
- Augmentation study, first PLISM readouts 2026-10-16 (per-pair block "CGM nonlinear vs
  linear encoder", ref = V0-S a602): V0-S without augmentation
  (`runs/bench/2026-10-16_11-56-22_lxbelshark_06ac_cgm_v0s_noaug_3fa4`) 0.896 / 0.318 / 0.269 /
  0.317 vs a602 0.897 / 0.319 / 0.270 / 0.318, all pairs −0.001 [−0.002, −0.000] → augmentation
  is not what drives CGM's PLISM gain. CGM-init|geo|5120, the untrained init of the V0-S fit
  (`runs/bench/2026-10-16_11-55-56_lxbelshark_d262_aug_cgminit_geo_5120`) 0.891 / 0.305 / 0.248 /
  0.298: all pairs −0.020 [−0.023, −0.017] below the trained V0-S (scanner + staining −0.022),
  while on SCORPION val the init equals V0-S (augmentation-attribution table) → on PLISM the
  gain comes from the ELBO training of the linear map, which SCORPION val does not see.
  In progress: `aug_g0_geo_all` PLISM (`runs/bench/2026-10-16_13-09-01_lxbelshark_9a38_aug_g0_geo_all`),
  then `aug_cgminit_clean_all`; V1-S no-aug fit `runs/2026-10-16_13-08-26_lxbelshark_7e3c`
  (W&B pva44z5o, then its PLISM `cgm_v1s_noaug_7e3c`); then `cgm_select_all` (command in the
  item above, user). To do when done: add `aug_g0_geo_all`, `aug_cgminit_clean_all` to the
  G0 per-pair block and `cgm_v1s_noaug_7e3c` to a block with ref=cgm_v1s_7b71.
- LOSO check for the selected CGM (V1-S), IMPLEMENTED 2026-10-16. It is mandatory per
  `docs/archive/experiment7-draft.md` (gates G-A/G-C): the nonlinear encoder could recognize scanner
  signatures (an implicit gate), which means extrapolation on an unseen scanner. Code:
  `sipe.eval.loso` (LOSO table), with `SCORPIONDataModule(hold_out_scanner=...)` dropping
  scanner j from train only (views=4), and `scripts/loso_fits.sh` for the trained arms.
  Arms (user decision 2026-10-16): G0^-j and CGM-init^-j (closed form, in the run), plus
  trained V1-S and V0-S, 5 fits each.
  First run, closed form only (`runs/bench/2026-10-16_11-34-36_claude-box_bf0f_scorpion_loso_closed_form`):
  - its full-data G0 (0.9787 val top-1) equals the cgm_select G0, which is the sanity check;
  - on the unseen pairs, CGM-init^-j is above G0^-j for every j, mean +0.0037
    [+0.0008, +0.0069];
  - holding a scanner out costs both estimators (G0 mean −0.0057, CGM-init −0.0041 vs the
    full fits). The largest costs are AT2 and Philips (G0 −0.0113 / −0.0090); P1000 does not
    stand out (−0.0029).
  Trained readout, 2026-10-16 (`runs/bench/2026-10-16_23-59-47_lxbelshark_b8e2_scorpion_loso`; the closed-form rows are identical to bf0f, so
  the table now shows b8e2 only): gate passed. V1-S^-j − G0^-j on the unseen pairs is
  positive with the CI above 0 for every j (AT2 +0.0155, Philips +0.0105, P1000 +0.0051,
  GT450 +0.0047, DP200 +0.0042), mean +0.0080 [+0.0019, +0.0148], larger than on the seen
  pairs (+0.0064). The linear control V0-S^-j is about the closed-form CGM-init (+0.0040 vs
  +0.0037; GT450 CI crosses 0). Holding j out costs V1-S −0.0042 vs the full fit, the same
  order as G0 (−0.0057), so the gap over G0 survives. Biggest gains where G0 loses most
  (AT2, Philips). Caveats: one fit per j, 5 val WSIs (wide intervals), val near ceiling,
  SCORPION scanners only (not unseen-scanner generalization in general).
  Done (was: host, ~10 h): `ARM=V1-S CUDA_VISIBLE_DEVICES=0 scripts/loso_fits.sh` and
  `ARM=V0-S CUDA_VISIBLE_DEVICES=1 scripts/loso_fits.sh`, then `sipe.eval.loso` with 7b71,
  a602 and the 10 fit dirs (the script prints the command), then `results_tables --write`.
  Gate: V1-S^-j − G0^-j on the unseen pairs not below 0, and compare it with V0-S^-j (the
  linear control). Roadmap section 10 frames LOSO as optional, the draft as mandatory for a
  nonlinear winner; PLISM scanner stratification is not a substitute.
- PLISM without HR (2026-10-15): HR = overnight Harris hematoxylin (checked on the images),
  so every PLISM table now has a `drop_stain=HR` twin (section 1). The ranking does not
  change: on all pairs without HR, CGM 7b71 0.371 and seed 43 0.372 > V0-S 0.367 > region
  eraser 0.359 > G0 0.339 > GAP 0.262 (PLISM-without-HR table); CGM vs the region eraser
  +0.012 [+0.010, +0.015] (per-pair block vs the region eraser, without HR). For the paper:
  full numbers as the main result, without-HR alongside with the reason.
- Exp 7 CGM: first fit done (`runs/2026-10-14_22-30-38_lxbelshark_7b71`); PLISM done
  2026-10-15, above G0 in every row (see roadmap status; per-pair table). HEST done
  2026-10-15 (`runs/bench/2026-10-15_07-47-25_lxbelshark_8823_cgm_v1s_7b71`): average 0.363 =
  G0, preservation holds on average. V0-S and seed-43 fits done 2026-10-15 (see roadmap
  status; on SCORPION val, seed 43 reproduces 7b71 and V0-S is lower). Next: their PLISM
  done 2026-10-15 (see roadmap status: V0-S gets most of the gain, nonlinear +0.003–0.004;
  vs the region eraser +0.011 on all pairs). Next: HEST for the region eraser and V0-S
  (`scripts/bench_hest_linear_anchor.sh`, gitignored; after D2 the region-eraser line is
  optional, for the follow-up paper), then and the `cgm_select` driver (SCORPION val/test with
  the shared protocol) before claiming the gain. Fit commands used (both done 2026-10-15):
  - linear-encoder arm V0-S (r = None: only M, L, b and the Cholesky factors train; smoke
    run OK 2026-10-15): `uv run sipe fit --config configs/consensus_gen.yaml --model.init_args.network.init_args.encoder_hidden_dim=0`.
    Separates the nonlinear encoder from the rest of the recipe (augmented images through
    the backbone, ELBO with per-scanner likelihood). V0-S ≈ 7b71 on PLISM → the gain is not
    the nonlinearity.
  - second seed of V1-S: `uv run sipe fit --config configs/consensus_gen.yaml --seed_everything=43`.
  Then PLISM for each with `--posthoc runs/<run>/map_cgm.pt` and add the tags to the
  `plism_pairs` block (`tags=`). SCORPION gain on the shared protocol: done 2026-10-15
  (`cgm_select`, CGM selection table). Fit command for more arms/seeds:
  `uv run sipe fit --config configs/consensus_gen.yaml`. Arm V1-S: frozen H0-mini GAP,
  per-scanner likelihood at fit time, nonlinear encoder, linear g; init = linear-Gaussian
  solution from 40 train batches; 128 locations x 5 views; geometric augmentations only.
  Check: `val/retrieval_top1_cgm` vs `val/retrieval_top1_backbone` over training
  (step 0 = linear init), `val/nonlinear_rel` (how far f moved from linear), `val/nll`.
  Then PLISM with `--posthoc runs/<run>/map_cgm.pt` (written at each checkpoint save) and
  compare with G0. Caveats: hyperparameters (ρ, β, lrs) not selected on val yet; the
  draft's `cgm_select` driver (clean-cache val/test with the shared protocol) and LOSO
  check are not implemented.

## 3. Open points outside the ISBI scope

- Content-dependent scanner offsets (2026-10-16, `sipe.eval.content_offsets`; motivation:
  the CGM's shift δ_k is constant per scanner). Design: per scanner pair (k, j), target
  x_k − x_j, content = mean of the other three views, so target and predictor share no
  acquisition noise if the noise is independent across scanners (regressing r_k on the
  full consensus is biased under scanner-specific noise covariances: on null synthetic
  data it found a spurious gain of up to 0.041, the pair design about 0; scratch check,
  not a run). Reading of the full run (all values from
  `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap`:
  content_offsets table, `per_scanner.csv`, `per_cluster.csv`, `ridge.csv`; the run used
  the code committed in 1630ac2, then still uncommitted on top of cf398dd):
  - The constant offset explains 0.259 of the val pair-difference energy (tile_all), the
    same value as `val_offset_explained_raw` in the offsets table (consistency check).
    The smoke run's p=8 row reproduces exactly (0.0915 val gain on both machines).
  - The scanner shift depends on content, and more so with more content directions.
    Val gain over the constant, raw / whitened (tile_all): linear p=8 0.091 / 0.017,
    p=32 0.128 / 0.038, p=128 0.158 / 0.057; Δll +0.069, +0.101, +0.123 [0.114, 0.133]
    nats/dim, positive in 5/5 val WSIs for every linear and cluster model. CV on train
    agrees (0.093, 0.133, 0.162), so no overfitting; the shuffled control is −0.000.
    Not saturated at p=128 (grid edge); λ = 0.1 for almost every pair (`ridge.csv`).
  - A linear function of content beats content clusters: cluster K=8 val gain 0.054,
    K=4 0.043. The dependence is smooth, not a few tissue modes.
  - Mostly a change of direction: the content-dependent part lies 0.283 (p=8) to 0.193
    (p=128) in the 4-d constant-offset span; the rest is outside it.
  - Small in the whitened metric: 0.057 of what the constant leaves (p=128). The effect
    sits in the high-variance residual directions, which a covariance-weighted map
    (G0, the CGM likelihood) already treats as noise.
  - Not P1000: without it the gain is slightly larger (p=128 raw 0.180, whitened 0.057,
    Δll +0.144).
  - Per scanner (p=128, pairs containing it): AT2 0.202 and Philips 0.186, the others
    0.126–0.136. These are also the scanners with the largest constant offsets and the
    largest LOSO costs (findings 6, 8).
  - Not only background: at p=128 every K=8 content cluster gains 0.127–0.225 (val). The
    highest-energy cluster (cluster 4: 0.047 of val tiles, energy ratio 2.21) gains the
    most (0.225) and is the only one where clusters alone do well (K=4 0.148). That
    cluster is probably the glass / edge cluster of Exp 1, but this was not checked
    visually (the clusters of this run are not the Exp 1 clusters).
  - Gate for a CGM arm with a per-scanner linear generator (L_k = L + D_k, Σ_k D_k = 0):
    passes on SCORPION (holds at p = 32 / 128, outside the high-energy cluster, positive
    whitened). Caveats: (i) relies on noise independence across scanners (future-work
    point below; AT2 + GT450 are both Leica); (ii) the small whitened gain means the
    retrieval benefit may be small; (iii) the content-aware maps G2 / Q2 did not transfer
    to PLISM (finding 7), so PLISM is the real test. Out of the ISBI scope; a candidate
    for the follow-up paper. No *scanner-blind linear* map can do better than G0 on
    the same moments (corrected 2026-10-16): such a map is fixed by the empirical second
    moments, and these already contain the content-dependent part. (The shift itself
    has a closed-form estimate, `content_offsets`; a closed-form retrieval ceiling is
    the oracle scanner-specific affine G0, proposed below, not run.) With
    x_k = (L + D_k) z + δ_k + ε_k and Σ_k D_k = 0, mean_k D_k D_kᵀ moves from the
    cross-view covariance into the within-tile one (`init_from_moments`: signal,
    noise), so the CGM-init with per-scanner loadings is about today's init
    (f = C (C + N + O)⁻¹), and G0 is already the best linear predictor of the consensus.
    D_k can only matter (a) through training, as a better-specified likelihood that
    changes the trained map (an arm "V0-S + D_k", compared with V0-S on PLISM), or
    (b) through a nonlinear scanner-blind map: E[consensus | x] is nonlinear when the
    shift depends on content.
  - Decision 2026-10-16 (user): no oracle run before the fit; test the model directly
    with one V1-S+D fit (`ContentShiftCGM`, `configs/consensus_gen_shift.yaml`,
    rank 128, `docs/experiment7.md`). V1-S+D clearly above V1-S on PLISM → worth
    pursuing (follow-up paper); otherwise finalize the ISBI paper with what exists and
    report the oracle as an analysis. Implemented and smoke-tested 2026-10-16.
  - Read 2026-10-16 (fit `runs/2026-10-16_19-27-29_lxbelshark_e3c7`, 6000 steps, seed 42
    as 7b71): the shift is used (W&B val/shift_rel 0.0705 at the last step) and the
    likelihood fits better (val/nll 0.0867 vs 0.0985 for 7b71), but the in-training val
    top-1 does not move (0.9849 vs 0.9852), and on PLISM V1-S+D is below V1-S in every group:
    all pairs −0.005 [−0.006, −0.004], inter-scanner −0.002 [−0.003, −0.001], 0% of all pairs
    better (`runs/bench/2026-10-16_20-17-12_lxbelshark_d531_cgm_v1s_shift_e3c7`, block vs
    7b71 in `docs/progress.md`). Against V0-S (block vs a602) it is only ahead on the
    inter-scanner pairs, +0.001 [+0.000, +0.002] (64% better), and below elsewhere: all pairs
    −0.002 [−0.003, −0.001] (27% better), inter-staining −0.003 [−0.004, −0.002]. One
    seed, but the gap is five times the seed spread (+0.001). Not verified: the reading
    that D_k z absorbs the content-dependent part of the nuisance, so the scanner-blind
    map no longer has to remove it. Verdict, by the user's rule: not pursued; finalize the
    ISBI paper with V1-S, the oracle stays a later analysis. The self-pair noise-fitting
    check was not run (moot once the arm is dropped).
  - Oracle retrieval ceiling (for the analysis, not run).
    Per scanner s, ridge-regress the consensus on x_s (train), apply with the true
    scanner on val, SCORPION retrieval vs G0. It captures δ_s + D_s exactly at the
    linear level: oracle ≈ G0 → no linear retrieval headroom from content-dependent
    shifts; oracle > G0 → headroom a scanner-blind model could partly reach (single-view
    scanner probe 0.956). CGM + D design if built: δ_s(z) = δ_s + A_s Pᵀ z, A_s
    centered over scanners (Σ D_s = 0), A_s = 0 and P = top-r eigenvectors of the init
    signal C at init, r ∈ {64, 128} (the diagnostic was not saturated at p = 128),
    full rank as ablation; check that the learned δ_k(m(u)) − δ_j(m(u)) does not
    explain more of the val pair differences than the leak-free diagnostic (self pair
    in the loss + scanner-blind encoder → possible noise fitting). V1-S + D is the
    informative arm; with a linear encoder D_s may leave the map unchanged.
- Future work (discussed 2026-10-16, not implemented): test the noise-independence
  assumption of the `content_offsets` pair design. The design is unbiased only if the
  acquisition noise of the target scanners (k, j) is independent of the noise of the
  predictor views. Noise shared by *all* scanners is not identifiable from one scan per
  scanner (it is indistinguishable from content) and cancels in x_k − x_j, so it is
  harmless. Noise shared by a *subset* of scanners would leak into the predictor and
  look like content dependence. Suspects: AT2 and GT450 (both Leica Aperio), and a
  common registration reference. From second moments alone, this cannot be separated
  from scanner-specific content gains D_k; a definitive test needs repeat scans on the
  same scanner, which SCORPION does not have. Proposed checks (as `--predictor
  all|drop1` + `noise_dependence.csv` in `sipe.eval.content_offsets`, with a synthetic
  test where two scanners share noise):
  1. Drop one view: for each pair, predict from 2 of the 3 other views (3 choices). Real
     content dependence holds for every choice (up to attenuation); noise shared between
     k and l makes the gain collapse when l is dropped, only for pairs containing k.
  2. Covariance between disjoint pairs, e.g. (AT2 − DP200) vs (GT450 − Philips), after
     removing the content prediction: top canonical correlation per disjoint pair with
     WSI bootstrap intervals. Null: residuals of different tiles of the same slide (keeps
     slide effects, breaks within-tile sharing). Concentrated on one scanner pair →
     correlated noise; diffuse → inadequate content model.
  3. If 1 or 2 flags AT2 + GT450: rerun the content diagnostic with the other Leica view
     excluded from the predictor for Leica pairs.
  Reading: a stable drop-one gain and disjoint-pair covariance at the null support
  independence as far as this design allows. Noise shared by all scanners remains
  untestable; state it as a caveat.
- Exp 6 design (discussed 2026-10-14, not started): consensus VAE whose linear limit is
  G0 (content latent shared by the views, acquisition latent marginalized analytically,
  decoder outputs the consensus). The offset diagnostic rejects exchangeable scanner
  nuisance as a likelihood model, so an acquisition mixture (components = scanners or
  latent modes, fallback to G0) is the candidate nonlinearity. Proposed first step:
  closed-form scanner-component mixture shrinkage on SCORPION val; not implemented.
- Exp 4 PLISM: DONE 2026-10-15, Q2 below G0 in every row → D4 taken: keep G0 (roadmap status). Commands that were used:
  `bench/run python -m sipe.bench.plism --posthoc runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap/map_Q2_m16.pt --tag quadratic_Q2_m16`
  and the control
  `bench/run python -m sipe.bench.plism --posthoc runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap/map_Q2-rand_m16.pt --tag quadratic_Q2-rand_m16`.
  D4: Q2 above G0 (`runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1`) in the
  PLISM rows *and* above Q2-rand → HEST for Q2; Q2 ≈ Q2-rand or below G0 → the gain is
  capacity / SCORPION-specific, keep G0.
- D3 taken 2026-10-14 (G2 not promoted; go to Exp 4). Remaining G0 follow-ups: HEST
  (running), per-pair / HR breakdown with intervals. History of the D3 check: before promoting G2, run PLISM (and then HEST) for the
  selected maps of G0 and G2, e.g. `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap/map_G0_K1.pt` and `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap/map_G2_K8.pt` (also
  `map_G2_K4.pt`), with `--posthoc` (module map files, loaded by `sipe.bench.posthoc`):
  `bench/run python -m sipe.bench.plism --posthoc <map> --tag mixture_<variant>_K<k>`.
  The question is whether G2's content-dependent N_k adds anything over G0 outside
  SCORPION, and whether G0 keeps its lead over the eraser on PLISM.
  Status 2026-10-14: G0 PLISM done (above the eraser in every row, see PLISM table);
  G2_K8 and G2_K4 PLISM done (both below G0 in every row); G0 and G2_K8 HEST done (see the G0 HEST reading); next per-pair / HR breakdown with
  group-aware intervals for G0 vs eraser.
- Augmentation check: geometric-aug fit done (`runs/2026-10-14_22-16-15_lxbelshark_44e5`, W&B
  lazztzda; PairedVAE augmentation table); PLISM in `scripts/bench_exp4.sh` (tag
  `paired_vae_geometric_aug_lazztzda`, `--representation z_gap` as `ysltwjoj`). Original plan: PairedVAE as `ysltwjoj`
  (`configs/paired_vae.yaml`) but train augmentations restricted to flips, exact 90°
  rotations and a mild zoom (`configs/paired_vae_geometric_aug.yaml`, `augmentations:
  [rot90, zoom]`; same transform for all scanner views of a location): does dropping
  blur/grayscale/invert/erasing/small-angle affine (blur and colour ops wash out scanner
  differences between the views) change PLISM/SCORPION? Compare with `ysltwjoj` (PLISM
  `runs/bench/2026-10-10_17-32-32_lxbelshark_95c5_paired_vae_ysltwjoj`). One seed each.
- Optional TODO (engineering, only if paired augmentation becomes a requirement): the
  scanner views of a location share their augmentation through seed replay in
  `PairedSCORPIONDataset.__getitem__` (same torch seed before each view). Correct and
  tested (`test_paired_views_share_geometric_augmentation`), but it relies on an implicit
  invariant: every random op draws from the torch RNG, the same number of times
  regardless of content. Albumentations / numpy / `random`-based ops would desync the
  views silently. Cleaner: torchvision v2, one call on all views (`t(*views)`) draws the
  parameters once; the paired dataset loads raw views and transforms them together,
  `RandomRot90` becomes a v2 transform; keep the v1 path for the legacy default so old
  configs reproduce bit-for-bit. ~1 h with tests.
- PLISM retrieval viz (tools added 2026-10-14, `docs/benchmarks.md`): pair table
  `runs/bench/2026-10-14_15-13-12_claude-box_1353_plism_pairs_gap_g0_eraser` (GAP ref, G0,
  eraser λ=0.5; merge of the official per-pair metrics, no recomputation). Smoke only so
  far: `runs/bench/2026-10-14_15-40-06_claude-box_cb44_plism_retrieval_viz_smoke_gmh`
  (one pair, GMH_S60–GMH_P, 460 tiles, not 8,139: not a result). Next: run the viz on the
  shortlist at 8,139 tiles (`--reason worst_ref gain_mixture_G0_K1`) and read
  `miss_frac_top1_same_slide` / `miss_frac_cross_rank1` in `summary.csv`. These measure
  whether misses lose to same-acquisition tiles rather than to other content.
- Background tiles: add a tissue fraction per tile (cheap image threshold) and report
  SCORPION retrieval on tissue tiles only, to see whether the G2 − G0 gain comes from the
  background cluster.
- Exp 2: the 2C pairing-scale control (eraser objective with tile-level deltas, or G0 with
  slide-level consensus) to explain why G0 beats the eraser on SCORPION. The eraser half is
  implemented (`--deltas region|tile`); sweeps to run on the GAP cache with the gain95 λ
  grid. The G0-on-slide-means arm is not implemented.
- D0 follow-up: per-slide-pair PLISM breakdown, HR stain, group-aware intervals for the
  eraser (and for every shortlisted method).
- In flight from the CATS phase: DistMatchVAE PLISM runs. Both are done and above the GAP
  reference `h0_mini_gap_ref` in every PLISM row (`sr40r1h1`
  `runs/bench/2026-10-13_20-17-04_lxbelshark_244b_dist_match_vae_sr40r1h1`: inter-scanner 0.855,
  all 0.251; `g7fkiwbp` `runs/bench/2026-10-13_21-11-53_lxbelshark_7015_dist_match_vae_g7fkiwbp`:
  0.837, 0.241; GAP 0.811, 0.227). Both are below the eraser (0.873, 0.281). Still to do:
  the one archive line, and revisit the archived "unpaired methods did not beat GAP".
- Seeds: every number has one seed so far; the roadmap asks for fit/train-seed variation for
  any shortlisted method (for the mixtures: k-means seed, which changes the clusters).

## 4. Generated tables outside the paper

The paper tables (PLISM, per-pair vs G0, CGM ablations, HEST, CGM selection, augmentation
attribution, LOSO) are in `docs/progress.md`.

### SCORPION proxy (held-out test split)

<!-- results:scorpion exclude=*smoke* -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| tag | repr | top-1 | top-5 | scanner probe logreg | scanner probe MLP | region probe | bench run dir | train W&B | SIPE commit |
|---|---|---|---|---|---|---|---|---|---|
| ce50 | backbone_gap | 0.802 | 0.919 | 0.956 | 0.939 | – | `runs/bench/2026-10-08_16-36-40_claude-box_44eb_scorpion_ce50` | – | 4fd9c05* |
| ce50 | z_gap | 0.668 | 0.826 | 0.908 | 0.903 | – | `runs/bench/2026-10-08_16-36-40_claude-box_44eb_scorpion_ce50` | s2bqyu2g | 4fd9c05* |
| ce50 | z_pre_gap | 0.451 | 0.653 | 0.954 | 0.947 | – | `runs/bench/2026-10-08_16-36-40_claude-box_44eb_scorpion_ce50` | s2bqyu2g | 4fd9c05* |
| ce50 | s | 0.001 | 0.005 | 0.951 | 0.957 | – | `runs/bench/2026-10-08_16-36-40_claude-box_44eb_scorpion_ce50` | s2bqyu2g | 4fd9c05* |
| cats_9lctrbqp | backbone_gap | 0.802 | 0.919 | 0.956 | 0.939 | 0.975 | `runs/bench/2026-10-08_21-04-03_claude-box_cd12_scorpion_cats_9lctrbqp` | – | d0c7a1a* |
| cats_9lctrbqp | z_gap | 0.844 | 0.941 | 0.917 | 0.891 | 0.976 | `runs/bench/2026-10-08_21-04-03_claude-box_cd12_scorpion_cats_9lctrbqp` | 9lctrbqp | d0c7a1a* |
| cats_9lctrbqp | z_pre_gap | 0.844 | 0.941 | 0.917 | 0.891 | 0.976 | `runs/bench/2026-10-08_21-04-03_claude-box_cd12_scorpion_cats_9lctrbqp` | 9lctrbqp | d0c7a1a* |
| cats_9lctrbqp | s | 0.000 | 0.002 | 0.967 | 0.961 | 0.075 | `runs/bench/2026-10-08_21-04-03_claude-box_cd12_scorpion_cats_9lctrbqp` | 9lctrbqp | d0c7a1a* |
| paired_vae_ysltwjoj | backbone_gap | 0.802 | 0.919 | 0.956 | 0.939 | 0.975 | `runs/bench/2026-10-10_14-52-05_claude-box_807c_scorpion_paired_vae_ysltwjoj` | – | 3d8ba47* |
| paired_vae_ysltwjoj | z_gap | 0.962 | 0.993 | 0.850 | 0.831 | 0.985 | `runs/bench/2026-10-10_14-52-05_claude-box_807c_scorpion_paired_vae_ysltwjoj` | ysltwjoj | 3d8ba47* |
| paired_vae_ysltwjoj | z_pre_gap | 0.962 | 0.993 | 0.850 | 0.831 | 0.985 | `runs/bench/2026-10-10_14-52-05_claude-box_807c_scorpion_paired_vae_ysltwjoj` | ysltwjoj | 3d8ba47* |
| paired_vae_ysltwjoj | s | 0.003 | 0.009 | 0.966 | 0.969 | 0.361 | `runs/bench/2026-10-10_14-52-05_claude-box_807c_scorpion_paired_vae_ysltwjoj` | ysltwjoj | 3d8ba47* |

Held-out test split; probes trained on val (`summary.csv`).

<!-- /results:scorpion -->

### Soft-shrinkage eraser sweep (SCORPION proxy, test)

<!-- results:softshrink -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| tag | pooling | deltas | λ | top-1 | top-5 | scanner probe logreg | scanner probe MLP | region probe | note | bench run dir | SIPE commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| softshrink_h0mini_gap | gap | slide | 0 | 0.802 | 0.919 | 0.956 | 0.939 | 0.974 | raw | `runs/bench/2026-10-13_15-44-35_lxbelshark_d6f7_scorpion_softshrink_h0mini_gap` | 5405d71 |
| softshrink_h0mini_gap | gap | slide | 100 | 0.966 | 0.993 | 0.265 | 0.363 | 0.984 | best val top-1 | `runs/bench/2026-10-13_15-44-35_lxbelshark_d6f7_scorpion_softshrink_h0mini_gap` | 5405d71 |
| softshrink_h0mini_gap_gain95 | gap | slide | 0 | 0.802 | 0.919 | 0.956 | 0.939 | 0.974 | raw | `runs/bench/2026-10-13_17-45-08_lxbelshark_db10_scorpion_softshrink_h0mini_gap_gain95` | 59e49ee* |
| softshrink_h0mini_gap_gain95 | gap | slide | 0.5 | 0.959 | 0.991 | 0.684 | 0.455 | 0.985 | smallest lambda with val top-1 gain >= 0.95 x best gain | `runs/bench/2026-10-13_17-45-08_lxbelshark_db10_scorpion_softshrink_h0mini_gap_gain95` | 59e49ee* |
| softshrink_gap_tile | gap | tile | 0 | 0.802 | 0.919 | 0.956 | 0.939 | 0.974 | raw | `runs/bench/2026-10-14_09-21-09_lxbelshark_c54b_scorpion_softshrink_gap_tile` | 4b6e266 |
| softshrink_gap_tile | gap | tile | 0.4 | 0.963 | 0.993 | 0.905 | 0.777 | 0.986 | smallest lambda with val top-1 gain >= 0.95 x best gain | `runs/bench/2026-10-14_09-21-09_lxbelshark_c54b_scorpion_softshrink_gap_tile` | 4b6e266 |
| softshrink_gap_region | gap | region | 0 | 0.802 | 0.919 | 0.956 | 0.939 | 0.974 | raw | `runs/bench/2026-10-14_19-02-29_lxbelshark_71bb_scorpion_softshrink_gap_region` | 67e83ee |
| softshrink_gap_region | gap | region | 0.7 | 0.971 | 0.994 | 0.720 | 0.393 | 0.987 | smallest lambda with val top-1 gain >= 0.95 x best gain | `runs/bench/2026-10-14_19-02-29_lxbelshark_71bb_scorpion_softshrink_gap_region` | 67e83ee |

Test split, proxy protocol (`summary_test.csv`); λ selected on val, rule in the note (`selected.json`).

<!-- /results:softshrink -->

### Content dependence of acquisition residuals (Experiment 1, SCORPION train CV + val)

Cached GAP features; test never read. Pairing unit: `tile_*` = tile-level matched pairs,
`region_*` = mean over a region's 49 tiles per scanner; `*_noP1000` drops P1000
(registration check). Likelihoods in the top-32 residual subspace of train.

<!-- results:content exclude=*_40f5_* -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| setting | K | α* | CV Δll | CV folds > 0 | CV shuffled | CV scale-only | val Δll [WSI 95% CI] | val WSIs > 0 | val shuffled | val scale-only | val energy MSE ratio (shuffled) | heterog. fit / val | bench run dir | SIPE commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tile_all | 2 | 0 | 0.060 ± 0.008 | 5 | -0.000 | 0.011 | 0.067 [0.045, 0.089] | 5/5 | 0.000 | 0.016 | 0.796 (0.998) | 0.202 / 0.153 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| tile_all | 4 | 0 | 0.101 ± 0.009 | 5 | -0.003 | 0.016 | 0.113 [0.078, 0.147] | 5/5 | -0.008 | 0.020 | 0.647 (0.993) | 0.242 / 0.243 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| tile_all | 8 | 0 | 0.129 ± 0.013 | 5 | -0.004 | 0.018 | 0.137 [0.106, 0.167] | 5/5 | -0.004 | 0.024 | 0.592 (0.994) | 0.271 / 0.283 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| tile_noP1000 | 2 | 0 | 0.066 ± 0.009 | 5 | 0.000 | 0.015 | 0.079 [0.052, 0.105] | 5/5 | 0.000 | 0.018 | 0.734 (0.998) | 0.222 / 0.172 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| tile_noP1000 | 4 | 0 | 0.105 ± 0.008 | 5 | -0.002 | 0.014 | 0.124 [0.081, 0.161] | 5/5 | -0.006 | 0.014 | 0.633 (0.993) | 0.243 / 0.244 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| tile_noP1000 | 8 | 0 | 0.134 ± 0.014 | 5 | -0.004 | 0.016 | 0.145 [0.105, 0.180] | 5/5 | -0.003 | 0.016 | 0.578 (0.990) | 0.270 / 0.288 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| region_all | 2 | 0 | 0.066 ± 0.018 | 5 | -0.011 | 0.003 | 0.078 [0.032, 0.117] | 4/5 | -0.009 | 0.007 | 0.626 (1.036) | 0.187 / 0.100 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| region_all | 4 | 0.25 | 0.108 ± 0.005 | 5 | -0.012 | 0.003 | 0.088 [0.046, 0.132] | 5/5 | -0.009 | -0.002 | 0.701 (1.043) | 0.208 / 0.131 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| region_all | 8 | 0.25 | 0.134 ± 0.026 | 5 | -0.030 | 0.000 | 0.103 [0.047, 0.150] | 4/5 | -0.022 | -0.003 | 0.667 (1.029) | 0.229 / 0.205 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| region_noP1000 | 2 | 0 | 0.065 ± 0.022 | 5 | -0.011 | 0.005 | 0.076 [0.021, 0.119] | 4/5 | -0.012 | 0.006 | 0.614 (1.034) | 0.192 / 0.102 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| region_noP1000 | 4 | 0.25 | 0.109 ± 0.007 | 5 | -0.014 | 0.005 | 0.086 [0.039, 0.137] | 5/5 | -0.011 | -0.010 | 0.755 (1.037) | 0.220 / 0.134 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |
| region_noP1000 | 8 | 0.25 | 0.124 ± 0.035 | 5 | -0.028 | -0.001 | 0.110 [0.058, 0.160] | 5/5 | -0.023 | -0.011 | 0.625 (1.025) | 0.234 / 0.207 | `runs/bench/2026-10-13_20-09-24_claude-box_c190_scorpion_content_dep_gap` | 9d6c388* |

Held-out mean log-likelihood difference, content-conditioned minus global residual model (nats/dim, > 0 favors content; `summary.csv`). CV = grouped folds over train WSIs (± sd over folds); val = fit on train. α* = shrinkage toward the global covariance, chosen by CV. Shuffled = cluster labels permuted in train; scale-only = cluster-specific scalar × global covariance. Energy MSE ratio < 1: cluster means predict held-out residual energy better than the global mean.

<!-- /results:content -->

### Scanner offsets: is the nuisance exchangeable across scanners? (SCORPION train + val)

Diagnostic for the exchangeable-nuisance assumption of G0 (pooled N_0) and of the
consensus-VAE sketch for Exp 6. Cached GAP features, tile-level pairs, test never read.
`sipe.eval.scanner_offsets`.

<!-- results:offsets -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| setting | scanners | energy share raw: global / slide / region / tile | whitened: global / slide / region / tile | val explained by train offsets raw / whitened | val Δll offset | val Δll cov | val Δll offset+cov [WSI 95% CI] | val Δll mixture, no label [WSI 95% CI] | val resid. scanner acc | offset span: rank / offset fraction / content share | bench run dir | SIPE commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tile_all | 5 | 0.295 / 0.053 / 0.069 / 0.583 | 0.005 / 0.027 / 0.056 / 0.913 | 0.259 / 0.004 | 0.118 | 0.109 | 0.157 [0.140, 0.174] | 0.110 [0.095, 0.125] | 0.992 | 4 / 0.849 / 0.042 | `runs/bench/2026-10-14_11-45-49_claude-box_f4c9_scorpion_scanner_offsets_gap` | e1692a2* |
| tile_noP1000 | 4 | 0.330 / 0.055 / 0.070 / 0.546 | 0.004 / 0.026 / 0.056 / 0.914 | 0.288 / 0.003 | 0.104 | 0.104 | 0.154 [0.141, 0.167] | 0.113 [0.100, 0.125] | 0.996 | 3 / 0.877 / 0.033 | `runs/bench/2026-10-14_11-45-49_claude-box_f4c9_scorpion_scanner_offsets_gap` | e1692a2* |

Tile-level residuals to the consensus over scanners (`summary.csv`). Energy shares (train) of E||r||^2 = N_0 split into nested scanner offsets: global per scanner, slide × scanner, region × scanner, tile; raw = identity metric, whitened = train N_0^-1. Val explained = residual energy removed by subtracting the train offsets per scanner. Δll = held-out val log-likelihood minus the exchangeable N(0, N_0) model (nats/dim, top residual subspace of train, > 0 favors scanner-aware): offset = per-scanner mean, cov = per-scanner covariance, offset+cov = both (these use the scanner label); mixture = equal-weight mixture of the offset+cov components, label marginalized. Resid. scanner acc = argmax component on val residuals (given the paired consensus). Offset span = span of the train global offsets: offset fraction = share of the residual energy there explained by the offsets; content share = share of the consensus variance there (what suppressing the span loses).

<!-- /results:offsets -->

### Content-dependent scanner offsets (SCORPION train CV + val)

Does a scanner's offset depend on the tissue? Per scanner pair, target = view difference,
content = the other views (leak-free design, module docstring). Cached GAP features,
tile-level pairs, test never read. `sipe.eval.content_offsets`.

<!-- results:content_offsets exclude=*smoke* -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| setting | scanners | model | CV gain raw | val explained raw / whitened | val gain over global raw / whitened | val Δll vs global [WSI 95% CI] | WSIs > 0 | content part in offset span | bench run dir | SIPE commit |
|---|---|---|---|---|---|---|---|---|---|---|
| tile_all | 5 | global | 0.000 | 0.259 / 0.004 | 0.000 / 0.000 | 0.000 [0.000, 0.000] | 0/5 | – | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_all | 5 | linear_p8 | 0.093 | 0.326 / 0.021 | 0.091 / 0.017 | 0.069 [0.066, 0.073] | 5/5 | 0.283 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_all | 5 | linear_p32 | 0.133 | 0.353 / 0.042 | 0.128 / 0.038 | 0.101 [0.093, 0.109] | 5/5 | 0.229 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_all | 5 | linear_p128 | 0.162 | 0.375 / 0.061 | 0.158 / 0.057 | 0.123 [0.114, 0.133] | 5/5 | 0.193 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_all | 5 | cluster_K4 | 0.041 | 0.290 / 0.010 | 0.043 / 0.006 | 0.023 [0.019, 0.029] | 5/5 | 0.353 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_all | 5 | cluster_K8 | 0.057 | 0.299 / 0.013 | 0.054 / 0.009 | 0.031 [0.026, 0.037] | 5/5 | 0.281 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_all | 5 | linear_p128_shuffled | -0.000 | 0.258 / 0.004 | -0.000 / -0.000 | -0.000 [-0.000, -0.000] | 1/5 | 0.067 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_noP1000 | 4 | global | 0.000 | 0.288 / 0.003 | 0.000 / 0.000 | 0.000 [0.000, 0.000] | 0/5 | – | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_noP1000 | 4 | linear_p8 | 0.106 | 0.363 / 0.019 | 0.106 / 0.015 | 0.081 [0.076, 0.086] | 5/5 | 0.290 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_noP1000 | 4 | linear_p32 | 0.152 | 0.392 / 0.039 | 0.147 / 0.036 | 0.117 [0.108, 0.126] | 5/5 | 0.226 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_noP1000 | 4 | linear_p128 | 0.184 | 0.416 / 0.060 | 0.180 / 0.057 | 0.144 [0.133, 0.156] | 5/5 | 0.192 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_noP1000 | 4 | cluster_K4 | 0.045 | 0.323 / 0.009 | 0.049 / 0.006 | 0.025 [0.021, 0.033] | 5/5 | 0.377 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_noP1000 | 4 | cluster_K8 | 0.066 | 0.331 / 0.011 | 0.060 / 0.008 | 0.035 [0.029, 0.042] | 5/5 | 0.283 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |
| tile_noP1000 | 4 | linear_p128_shuffled | -0.000 | 0.288 / 0.003 | -0.000 / -0.000 | -0.000 [-0.000, -0.000] | 1/5 | 0.063 | `runs/bench/2026-10-16_17-07-06_lxbelshark_90b7_scorpion_content_offsets_gap` | cf398dd* |

Does a scanner's offset depend on the content? Per scanner pair (k, j), target x_k − x_j, content = the mean of the other views (no shared noise with the target); `summary.csv`, train and val only. global = constant δ_k − δ_j; linear_p = constant + ridge on the top-p content PCs (λ per pair by grouped CV on train); cluster_K = one mean per content cluster; shuffled = linear fitted on permuted content (control). Explained = share of the val pair-difference energy removed; gain = share of what the constant offsets leave that the content predicts (CV = out-of-fold on train WSIs); whitened = train N_0^-1 metric. Δll = val Gaussian log-likelihood minus the global model, top residual subspace (nats/dim). Offset span = share of the content-dependent part lying in the span of the global offsets.

<!-- /results:content_offsets -->

### Gaussian-mixture shrinkage (Experiment 3, SCORPION val selection + test)

Cached GAP features; pairing unit = tile-level pairs (consensus over the 5 scanners), the
eraser (`reference`) uses slide-level deltas. Map: `sipe.estimators.mixture`, single-view
gating; G0 = global Gaussian (= Exp 2A consensus covariance shrinkage, α=1 is the affine
least-squares predictor of the tile consensus).

<!-- results:mixture -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| model | α | β | γ | val top-1 (α=1) | val region probe | val ll resid | val ll x | gate agree / views consistent | test top-1 | test top-5 | test scanner probe logreg / MLP | test region probe | note | bench run dir | SIPE commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | – | – | – | 0.887 | 0.978 | – | – | – | 0.802 | 0.919 | 0.956 / 0.939 | 0.975 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| reference (eraser) | – | – | – | 0.973 | 0.985 | – | – | – | 0.959 | 0.991 | 0.685 / 0.463 | 0.986 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G0 K=1 | 1.00 | – | – | 0.979 (0.979) | 0.986 | 1.087 | 0.131 | – | 0.965 | 0.993 | 0.617 / 0.426 | 0.986 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G1 K=2 | 1.00 | 0.25 | – | 0.978 (0.978) | 0.986 | 1.087 | 0.193 | 0.949 / 0.936 | 0.963 | 0.992 | 0.564 / 0.414 | 0.986 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G2-scale K=2 | 1.00 | 0.25 | – | 0.978 (0.978) | 0.987 | 1.085 | 0.191 | 0.949 / 0.936 | 0.963 | 0.992 | 0.566 / 0.422 | 0.987 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G2 K=2 | 1.00 | 0.25 | 0.10 | 0.980 (0.980) | 0.987 | 1.128 | 0.202 | 0.949 / 0.937 | 0.967 | 0.994 | 0.516 / 0.384 | 0.987 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G1 K=4 | 0.50 | 0.25 | – | 0.973 (0.975) | 0.987 | 1.087 | 0.242 | 0.915 / 0.897 | 0.958 | 0.990 | 0.621 / 0.487 | 0.986 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G2-scale K=4 | 0.50 | 0.25 | – | 0.974 (0.976) | 0.986 | 1.110 | 0.246 | 0.915 / 0.895 | 0.959 | 0.991 | 0.628 / 0.484 | 0.986 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G2 K=4 | 0.50 | 0.25 | 0.10 | 0.980 (0.982) | 0.987 | 1.177 | 0.258 | 0.915 / 0.892 | 0.965 | 0.993 | 0.532 / 0.426 | 0.987 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G1 K=8 | 0.50 | 0.50 | – | 0.973 (0.975) | 0.985 | 1.087 | 0.252 | 0.879 / 0.820 | 0.957 | 0.990 | 0.648 / 0.489 | 0.985 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G2-scale K=8 | 0.50 | 0.50 | – | 0.973 (0.975) | 0.985 | 1.111 | 0.260 | 0.879 / 0.819 | 0.958 | 0.991 | 0.649 / 0.496 | 0.986 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |
| G2 K=8 | 0.50 | 0.50 | 0.25 | 0.981 (0.983) | 0.987 | 1.193 | 0.280 | 0.879 / 0.822 | 0.968 | 0.994 | 0.558 / 0.418 | 0.989 | – | `runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap` | 9ee82ff* |

Val: selection split, scanner/region probes trained on train (`summary_val.csv`); α chosen by the 95%-of-best-val-top-1-gain rule, β by the val consensus likelihood, γ by the val residual likelihood (`selected.json`, `likelihood.csv`). ll = mean held-out log-likelihood, nats/dim: resid = residuals under N_k of their consensus cluster, x = single views under the mixture marginal. Gate = single-view argmax p(k|x) vs the consensus cluster, and share of tiles whose 5 views gate to one cluster (α=1 map, `gating.csv`). Test: proxy protocol, probes trained on val (`summary_test.csv`). Reference = soft-shrinkage eraser map evaluated in the same run.

<!-- /results:mixture -->

### Quadratic correction (Experiment 4, SCORPION val selection + test)

Cached GAP features; pairing unit = tile-level pairs; base = consensus ridge regression
(Exp 2B), so γ = inf is G0 (fitted in the same run). Map: `sipe.estimators.quadratic`;
Q1 = squares of nuisance projections, Q2 = content × nuisance products, Q2-rand = Q2 with
random nuisance directions (control).

<!-- results:quadratic exclude=*smoke* -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| model | features | γ | val top-1 | val Δtop-1 vs G0 | val consensus MSE / G0 (train) | val region probe | test top-1 | test top-5 | test scanner probe logreg / MLP | test region probe | note | bench run dir | SIPE commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | – | – | 0.887 | – | – | 0.978 | 0.802 | 0.919 | 0.956 / 0.939 | 0.975 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |
| reference (eraser) | – | – | 0.973 | – | – | 0.985 | 0.959 | 0.991 | 0.685 / 0.463 | 0.986 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |
| G0 | – | – | 0.979 | +0.0000 | – | 0.986 | 0.965 | 0.993 | 0.619 / 0.430 | 0.986 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |
| Q1 m=8 | 8 | 1.000 | 0.979 | +0.0001 | 0.999 (0.999) | 0.986 | 0.965 | 0.993 | 0.616 / 0.422 | 0.987 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |
| Q1 m=16 | 16 | 0.100 | 0.979 | +0.0002 | 0.996 (0.996) | 0.986 | 0.965 | 0.993 | 0.577 / 0.416 | 0.987 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |
| Q2 m=8 | 64 | 0.010 | 0.981 | +0.0024 | 0.960 (0.959) | 0.986 | 0.968 | 0.994 | 0.602 / 0.379 | 0.988 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |
| Q2 m=16 | 256 | 0.010 | 0.983 | +0.0043 | 0.943 (0.930) | 0.986 | 0.971 | 0.995 | 0.540 / 0.347 | 0.988 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |
| Q2-rand m=8 | 64 | 0.010 | 0.980 | +0.0008 | 0.995 (0.988) | 0.987 | 0.965 | 0.993 | 0.587 / 0.403 | 0.987 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |
| Q2-rand m=16 | 256 | 0.010 | 0.981 | +0.0019 | 0.991 (0.964) | 0.987 | 0.968 | 0.994 | 0.514 / 0.390 | 0.987 | – | `runs/bench/2026-10-14_11-53-58_lxbelshark_687d_scorpion_quadratic_gap` | e1692a2 |

Val: selection split, scanner/region probes trained on train (`summary_val.csv`); γ = ridge on the quadratic block, the largest γ whose val top-1 gain over G0 reaches 95% of the best gain (γ = inf is G0; `selected.json`). Features = number of quadratic terms (Q1: m squares, Q2: m × m products). Consensus MSE = mean squared error to the tile consensus per dim, ratio to G0 on val (train in brackets; `sweep_val.csv`). Test: proxy protocol, probes trained on val (`summary_test.csv`). G0 is fitted in the same run; reference = soft-shrinkage eraser map.

<!-- /results:quadratic -->

### PairedVAE: augmentation check (SCORPION val, in-loop)

<!-- results:training module=PairedVAEModule cols=retrieval_top1_zgap,retrieval_top1_backbone,probe_zgap_acc,probe_backbone_acc,probe_s_acc,align_loss,translation_loss -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| run dir | W&B | module | commit | step | retrieval_top1_zgap | retrieval_top1_backbone | probe_zgap_acc | probe_backbone_acc | probe_s_acc | align_loss | translation_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `runs/2026-10-10_15-18-38_lxbelshark_20c2` | ysltwjoj | PairedVAEModule | f48c72a | 6000/6000 | 0.977 | 0.887 | 0.689 | 0.802 | 0.933 | 0.241 | 0.237 |
| `runs/2026-10-13_19-12-17_lxbelshark_5465` | 8271lf26 | PairedVAEModule | ca9b819 | 6000/6000 | 0.952 | 0.887 | 0.797 | 0.802 | 0.933 | 0.326 | 0.234 |
| `runs/2026-10-14_22-16-15_lxbelshark_44e5` | lazztzda | PairedVAEModule | 9f003e7 | 6000/6000 | 0.979 | 0.887 | 0.362 | 0.802 | 0.974 | 0.079 | 0.219 |

Mean of the last 10 validations of `val/<col>`, read from the local `.wandb` file; step = last logged `trainer/global_step` / `max_steps`.

<!-- /results:training -->

### PLISM per slide pair vs the region-level eraser (D2 anchor candidate)

<!-- results:plism_pairs exclude=*smoke* ref=softshrink_gap_region_lam0.7 tags=mixture_G0_K1,softshrink_gap_tile_lam0.4,cgm_v0s_a602,cgm_v1s_7b71,cgm_v1s_seed43_8ed0 -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| method | pairs | n | top-1 | ref top-1 | Δ top-1 | 95% CI | pairs better | bench run dir |
|---|---|---|---|---|---|---|---|---|
| mixture_G0_K1 | inter-scanner | 273 | 0.887 | 0.896 | -0.009 | [-0.014, -0.005] | 9% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| mixture_G0_K1 | inter-staining | 546 | 0.303 | 0.321 | -0.018 | [-0.021, -0.015] | 2% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| mixture_G0_K1 | scanner + staining | 3276 | 0.243 | 0.261 | -0.018 | [-0.021, -0.015] | 2% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| mixture_G0_K1 | all | 4095 | 0.294 | 0.311 | -0.017 | [-0.020, -0.015] | 3% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| mixture_G0_K1 | inter-scanner, HR | 21 | 0.731 | 0.745 | -0.014 | [-0.031, -0.000] | 19% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| mixture_G0_K1 | all, HR | 609 | 0.032 | 0.038 | -0.006 | [-0.009, -0.004] | 1% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| softshrink_gap_tile_lam0.4 | inter-scanner | 273 | 0.889 | 0.896 | -0.007 | [-0.011, -0.004] | 24% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| softshrink_gap_tile_lam0.4 | inter-staining | 546 | 0.310 | 0.321 | -0.011 | [-0.013, -0.009] | 16% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| softshrink_gap_tile_lam0.4 | scanner + staining | 3276 | 0.254 | 0.261 | -0.006 | [-0.009, -0.004] | 33% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| softshrink_gap_tile_lam0.4 | all | 4095 | 0.304 | 0.311 | -0.007 | [-0.009, -0.005] | 30% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| softshrink_gap_tile_lam0.4 | inter-scanner, HR | 21 | 0.728 | 0.745 | -0.017 | [-0.038, -0.001] | 19% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| softshrink_gap_tile_lam0.4 | all, HR | 609 | 0.034 | 0.038 | -0.004 | [-0.006, -0.003] | 1% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| cgm_v0s_a602 | inter-scanner | 273 | 0.897 | 0.896 | +0.001 | [-0.002, +0.004] | 56% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v0s_a602 | inter-staining | 546 | 0.319 | 0.321 | -0.001 | [-0.004, +0.001] | 44% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v0s_a602 | scanner + staining | 3276 | 0.270 | 0.261 | +0.009 | [+0.007, +0.012] | 77% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v0s_a602 | all | 4095 | 0.318 | 0.311 | +0.007 | [+0.005, +0.010] | 71% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v0s_a602 | inter-scanner, HR | 21 | 0.731 | 0.745 | -0.014 | [-0.031, -0.002] | 10% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v0s_a602 | all, HR | 609 | 0.037 | 0.038 | -0.001 | [-0.001, -0.000] | 36% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v1s_7b71 | inter-scanner | 273 | 0.899 | 0.896 | +0.004 | [-0.000, +0.007] | 67% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_7b71 | inter-staining | 546 | 0.321 | 0.321 | +0.001 | [-0.002, +0.003] | 55% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_7b71 | scanner + staining | 3276 | 0.273 | 0.261 | +0.013 | [+0.010, +0.016] | 85% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_7b71 | all | 4095 | 0.321 | 0.311 | +0.011 | [+0.008, +0.013] | 80% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_7b71 | inter-scanner, HR | 21 | 0.732 | 0.745 | -0.013 | [-0.030, -0.000] | 14% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_7b71 | all, HR | 609 | 0.038 | 0.038 | +0.000 | [-0.001, +0.001] | 63% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_seed43_8ed0 | inter-scanner | 273 | 0.899 | 0.896 | +0.004 | [+0.000, +0.007] | 69% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |
| cgm_v1s_seed43_8ed0 | inter-staining | 546 | 0.322 | 0.321 | +0.001 | [-0.001, +0.004] | 60% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |
| cgm_v1s_seed43_8ed0 | scanner + staining | 3276 | 0.274 | 0.261 | +0.013 | [+0.010, +0.016] | 86% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |
| cgm_v1s_seed43_8ed0 | all | 4095 | 0.322 | 0.311 | +0.011 | [+0.008, +0.014] | 81% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |
| cgm_v1s_seed43_8ed0 | inter-scanner, HR | 21 | 0.733 | 0.745 | -0.012 | [-0.027, +0.001] | 14% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |
| cgm_v1s_seed43_8ed0 | all, HR | 609 | 0.038 | 0.038 | +0.000 | [-0.000, +0.001] | 63% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |

Reference: `softshrink_gap_region_lam0.7` (`runs/bench/2026-10-15_09-13-06_lxbelshark_0810_softshrink_gap_region_lam0.7`). Per slide pair top-1 (plismbench `metrics.csv`, mean of both directions); Δ = method minus reference, mean over the pairs of the group; 95% CI = percentile interval of a slide-level bootstrap (2000 draws, seed 0; pairs share slides, so slides are resampled and a pair counts once per drawn copy of each of its slides); pairs better = share of pairs with Δ > 0. HR = either slide has the HR stain. One fit per method, so the interval covers the slide sample only, not seeds.

<!-- /results:plism_pairs -->

### PLISM per slide pair vs the region-level eraser, without the HR stain

<!-- results:plism_pairs exclude=*smoke* drop_stain=HR ref=softshrink_gap_region_lam0.7 tags=mixture_G0_K1,softshrink_gap_tile_lam0.4,cgm_v0s_a602,cgm_v1s_7b71,cgm_v1s_seed43_8ed0 -->
<!-- generated by `uv run python -m sipe.eval.results_tables --write`; do not edit -->

| method | pairs | n | top-1 | ref top-1 | Δ top-1 | 95% CI | pairs better | bench run dir |
|---|---|---|---|---|---|---|---|---|
| mixture_G0_K1 | inter-scanner | 252 | 0.900 | 0.908 | -0.008 | [-0.014, -0.005] | 8% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| mixture_G0_K1 | inter-staining | 462 | 0.357 | 0.376 | -0.020 | [-0.023, -0.017] | 3% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| mixture_G0_K1 | scanner + staining | 2772 | 0.285 | 0.306 | -0.020 | [-0.023, -0.017] | 3% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| mixture_G0_K1 | all | 3486 | 0.339 | 0.359 | -0.019 | [-0.022, -0.017] | 3% | `runs/bench/2026-10-14_07-10-49_lxbelshark_0721_mixture_G0_K1` |
| softshrink_gap_tile_lam0.4 | inter-scanner | 252 | 0.902 | 0.908 | -0.006 | [-0.011, -0.003] | 24% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| softshrink_gap_tile_lam0.4 | inter-staining | 462 | 0.364 | 0.376 | -0.012 | [-0.015, -0.009] | 19% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| softshrink_gap_tile_lam0.4 | scanner + staining | 2772 | 0.299 | 0.306 | -0.007 | [-0.010, -0.004] | 38% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| softshrink_gap_tile_lam0.4 | all | 3486 | 0.351 | 0.359 | -0.007 | [-0.010, -0.005] | 35% | `runs/bench/2026-10-15_11-29-11_lxbelshark_06c6_softshrink_gap_tile_lam0.4` |
| cgm_v0s_a602 | inter-scanner | 252 | 0.910 | 0.908 | +0.002 | [+0.000, +0.004] | 60% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v0s_a602 | inter-staining | 462 | 0.375 | 0.376 | -0.002 | [-0.004, +0.001] | 50% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v0s_a602 | scanner + staining | 2772 | 0.317 | 0.306 | +0.011 | [+0.009, +0.014] | 84% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v0s_a602 | all | 3486 | 0.367 | 0.359 | +0.009 | [+0.007, +0.011] | 77% | `runs/bench/2026-10-15_10-03-00_lxbelshark_ae3c_cgm_v0s_a602` |
| cgm_v1s_7b71 | inter-scanner | 252 | 0.913 | 0.908 | +0.005 | [+0.002, +0.008] | 71% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_7b71 | inter-staining | 462 | 0.377 | 0.376 | +0.001 | [-0.002, +0.004] | 58% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_7b71 | scanner + staining | 2772 | 0.321 | 0.306 | +0.015 | [+0.012, +0.018] | 88% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_7b71 | all | 3486 | 0.371 | 0.359 | +0.012 | [+0.010, +0.015] | 83% | `runs/bench/2026-10-14_23-36-54_lxbelshark_22f4_cgm_v1s_7b71` |
| cgm_v1s_seed43_8ed0 | inter-scanner | 252 | 0.913 | 0.908 | +0.005 | [+0.002, +0.008] | 74% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |
| cgm_v1s_seed43_8ed0 | inter-staining | 462 | 0.378 | 0.376 | +0.002 | [-0.001, +0.005] | 63% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |
| cgm_v1s_seed43_8ed0 | scanner + staining | 2772 | 0.321 | 0.306 | +0.016 | [+0.013, +0.018] | 89% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |
| cgm_v1s_seed43_8ed0 | all | 3486 | 0.372 | 0.359 | +0.013 | [+0.011, +0.016] | 84% | `runs/bench/2026-10-15_11-30-45_lxbelshark_87d5_cgm_v1s_seed43_8ed0` |

Reference: `softshrink_gap_region_lam0.7` (`runs/bench/2026-10-15_09-13-06_lxbelshark_0810_softshrink_gap_region_lam0.7`). Per slide pair top-1 (plismbench `metrics.csv`, mean of both directions); Δ = method minus reference, mean over the pairs of the group; 95% CI = percentile interval of a slide-level bootstrap (2000 draws, seed 0; pairs share slides, so slides are resampled and a pair counts once per drawn copy of each of its slides); pairs better = share of pairs with Δ > 0. HR = either slide has the HR stain. One fit per method, so the interval covers the slide sample only, not seeds. Pairs with the stain(s) HR on either side are dropped before grouping and bootstrapping.

<!-- /results:plism_pairs -->
