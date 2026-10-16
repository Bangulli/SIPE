# Experiment 7: consensus generative model (CGM), as built (as of 2026-10-16)

The model behind the ISBI paper (scope and evidence: `docs/progress.md`; run history:
`docs/experiment-log.md`, "Exp 7 CGM: detailed history"). The original plan, written on
2026-10-14 before anything was built, is archived in `docs/archive/experiment7-draft.md`;
section 5 lists where the implementation departs from it.

Contract (roadmap section 1): train split only, selection on SCORPION val, test read once,
PLISM/HEST external. Pairing unit: tile-level pairs (the 5 scanner views of a tile).
Inference: one global map on one H0-mini GAP embedding: no scanner ID, no gate, no scanner
set.

## 1. Model

Code: `src/sipe/model/consensus_gen.py` (network, `CGMMap`), `src/sipe/training/
consensus_gen_module.py` (loss, init, validation), `configs/consensus_gen.yaml`.

- **Generative model (fit time)**, in standardized coordinates u = (x − center) / scale, for
  tile i seen by scanner k:
  - z_i ~ N(0, I_768), shared by the views of the tile;
  - u_{i,k} = g(z_i) + δ_k + B_k ε, with g(z) = b + L z + h(z);
  - the acquisition noise is integrated out analytically (full covariance B_k B_kᵀ, one
    Cholesky factor per scanner);
  - the scanner label k is used only in this likelihood. Exchangeable arms: δ = 0 and one
    shared factor.
- **V1-S+D** (`ContentShiftCGM` / `ContentShiftHead`, 2026-10-16): the shift becomes
  δ_k + D_k z, D_k = A_k Pᵀ of rank `shift_rank`, A_k centered over the scanners
  (Σ_k w_k D_k = 0, w = scanner frequencies of the init batches). Init A_k = 0 (the init
  is exactly V1-S's) and P = top-r eigenvectors of the init L; ρ also covers Σ‖D_k‖²_F;
  A, P in the linear optimizer group. Fit-time only: the map and `map_cgm.pt` are the
  V1-S ones. Logged: `shift_rel` = mean ‖D_k z‖ / mean ‖u − g − δ_k‖. Motivation and
  caveats: `docs/experiment-log.md`, section 3 (content-dependent scanner offsets).
- **Encoder (single view):** q(z|u) = N(m(u), A Aᵀ), m(u) = M u + r(u).
- **Inference map:** f(x) = center + scale · g(m(u)) (`CGMMap`, saved as `map_cgm.pt`, a
  `--posthoc` module map file).
- **Init** (`init_from_moments`, before training, from `init_batches` = 40 train batches =
  25,600 views): the linear-Gaussian solution. δ_k = per-scanner means; the cross-view
  covariance gives the signal (L = its square root); the within-tile covariance gives the
  noise; for the scanner-blind encoder the offsets enter as a random effect (marginal =
  signal + noise + offset covariance), M = Lᵀ marginal⁻¹. Eigenvalue floor `init_ridge` =
  1e-3 × tr(Cov)/D. The last layers of r and h start at zero, so at init f is the
  linear-Gaussian posterior-mean predictor of the noise-free signal (close to G0, which
  predicts the 5-scanner mean).
- **Loss**, per tile and per source view s, z_s ~ q(z|u_s): the negative ELBO of *all* views
  of the tile (each under its scanner's likelihood) + β·KL, divided by D = 768, + ρ‖output
  layers of r, h‖². A valid lower bound on the tile likelihood; it replaces PairedVAE's
  swap, translation and alignment losses.

## 2. Training recipe (`configs/consensus_gen.yaml`, the same for every arm)

Backbone in the loop: frozen H0-mini (fp16 autocast), GAP over patch tokens = `backbone_gap`.
`PairedSCORPIONDataModule`, 128 tile locations × 5 views per step, augmentations flips +
rot90 + zoom shared by the views of a tile (seed replay). AdamW: lr 3e-4 for the r/h MLPs
(weight decay 0.01), 1e-4 for M, L, b, δ and the Cholesky factors (no decay); cosine decay
over 6000 steps; β = 1, no KL warmup; ρ = 1e-3; seed 42. In-loop validation every 100 steps
(PLISM-style cross-scanner top-1 on all val tiles, scanner probes on WSI halves of val,
val NLL / KL, `nonlinear_rel`). Fit: `uv run sipe fit --config configs/consensus_gen.yaml`
plus the arm overrides below. Hyperparameters are fixed, not selected on val.

## 3. Arms and fits

| arm | overrides | fits |
|---|---|---|
| V0 linear, exchangeable (sanity: should reproduce G0) | `per_scanner=false encoder_hidden_dim=0` | queued (`scripts/cgm_arms_fits.sh`) |
| V0-S linear, per-scanner | `encoder_hidden_dim=0` | `runs/2026-10-15_08-50-48_lxbelshark_a602`; no aug `runs/2026-10-15_19-42-06_lxbelshark_3fa4` |
| V1 nonlinear encoder, exchangeable (isolates the per-scanner likelihood) | `per_scanner=false` | queued (`scripts/cgm_arms_fits.sh`) |
| **V1-S** nonlinear encoder, per-scanner (selected) | – | seed 42 `runs/2026-10-14_22-30-38_lxbelshark_7b71`, seed 43 `runs/2026-10-15_08-51-41_lxbelshark_8ed0`, seed 44 queued; no aug `runs/2026-10-16_13-08-26_lxbelshark_7e3c` |
| V2-S + nonlinear generator h | `generator_hidden_dim=1024` | `runs/2026-10-15_19-48-26_lxbelshark_920f` |
| V1-S+D content-dependent scanner shift (follow-up, not ISBI) | `--config configs/consensus_gen_shift.yaml` (`ContentShiftCGM`, `shift_rank: 128`) | pending (`scripts/cgm_shift_fit.sh`) |
| LOSO, per held-out scanner j | `--data.init_args.hold_out_scanner=<j> --data.init_args.views=4` | V1-S and V0-S × 5 scanners pending (`scripts/loso_fits.sh`) |

Overrides are under `--model.init_args.network.init_args.`; no-aug = `--data.init_args.
augmentations=[] --data.init_args.horizontal_flip=false --data.init_args.vertical_flip=false`.
Host scripts (gitignored): `scripts/cgm_arms_fits.sh`, `cgm_v2s_fit.sh`,
`aug_study_5_noaug_fits.sh`, `loso_fits.sh`; each creates its run dir and passes it via
`SIPE_RUN_DIR`, then runs PLISM on `map_cgm.pt`.

## 4. Evaluation pipeline and gates

- **SCORPION, shared protocol:** `sipe.eval.cgm_select` scores each `map_cgm.pt` on the clean
  GAP cache next to raw, the eraser and an in-run G0; fits differing only by seed form one
  candidate; simplest arm (V0 < V0-S < V1 < V1-S < V2 < V2-S, then larger ρ, smaller β) at
  95% of the best val gain over G0; test read once. Table block `cgm`.
- **Augmentation attribution (no training):** `sipe.eval.aug_attribution` + augmented caches
  (`sipe.features.extract`); block `augattr`.
- **LOSO:** `sipe.eval.loso`, closed-form G0^-j / CGM-init^-j in the run plus the trained
  LOSO fits; block `loso`.
- **External:** PLISM / HEST with `--posthoc runs/<fit>/map_cgm.pt`; per-pair blocks
  (`plism_pairs`) against G0, V0-S and V1-S.

Gates (from the draft; status in `docs/progress.md`, section 2):
- **G-C (= roadmap D6):** CGM > G0 on PLISM with similar HEST, and LOSO not below G0 →
  keep it and report the ablations; CGM ≈ the best closed-form model → report the simpler
  one. PLISM and HEST pass; LOSO pending for the trained arms.
- **LOSO gate:** V1-S^-j − G0^-j on the unseen pairs not below 0, read next to V0-S^-j.

## 5. Departures from the draft (`docs/archive/experiment7-draft.md`)

- **Backbone in the loop instead of staged feature caches** (the draft's
  `PairedFeatureDataModule` was not built). Augmented feature caches were built later, only
  for the closed-form attribution study.
- **Step 1, the offset-span quadratic control (Q1-off / Q2-off), was not built.** After D4
  (Q2 below G0 on PLISM, 2026-10-15) the quadratic family is out of ISBI scope; G-A and the
  Q-off part of G-C are dropped.
- **Step 2's augmentation audit and the `content` / `acquisition` / PairedVAE-augmentation
  policies were not built.** The question it served (does augmentation drive the gain?) was
  answered directly: PLISM of V0-S without augmentation ≈ V0-S, and the untrained init is
  well below the trained map (`docs/findings.md`, finding 9). Gate G-B is moot.
- **Selection is over arms, not ρ:** hyperparameters are fixed (section 2). The draft's ρ
  selection by the 95% rule was not run.
- **Seeds:** the draft asks for 3 seeds of the selected configuration; V1-S has 2, the third
  is queued.
