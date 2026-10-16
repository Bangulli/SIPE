# Draft plan: Experiment 7: generative consensus estimator, acquisition-aware, with an augmentation study

> **Status: archived 2026-10-16.** The plan as written on 2026-10-14, before anything was
> built. The model as built, its fits, gates and the departures from this plan:
> `docs/experiment7.md`. Paths below (e.g. `docs/progress.md` sections) are as of 2026-10-14.

## Context

The latest results (`docs/progress.md`, 2026-10-14):
- **G0 is the linear bar.** G0 is the linear least-squares predictor of the tile consensus. It beats the eraser on SCORPION and on all four PLISM rows, and keeps the HEST average (0.363 vs 0.364).
- **Content-dependent nuisance does not transfer.** G2 is below G0 on PLISM, so Exp 5 is dropped.
- **Exp 4 is still running.** In the val sweep, Q2 rises above G0 and above the Q2-rand control.
- **Exchangeable nuisance is rejected.** The scanner-offset diagnostic (`runs/bench/2026-10-14_11-45-49_claude-box_f4c9_scorpion_scanner_offsets_gap`) shows large, low-rank per-scanner offsets that carry over to unseen WSIs (they span 4 directions), plus per-scanner covariance shape.

The user wants a new experiment: model the embeddings generatively and estimate the consensus. It builds on G0 (linear limit, initialization) and on PairedVAE (single-view encoder trained with paired views). Its acquisition model is tailored to the offsets.

Data augmentation is studied explicitly, because the PairedVAE pipeline (inherited from CATS; `src/sipe/data/scorpion.py:213`) blurs every image, inverts half of them, and applies grayscale and erasing. All of these are shared across views, and blur and grayscale directly attenuate scanner differences.

User decisions (2026-10-14):
- staged feature caches, not backbone-in-loop;
- all three augmentation arms: audit, content-only, per-view acquisition;
- per-scanner likelihood, **at fit time only**;
- **no gating to known scanners at inference**: one global map like G0, since PLISM has 7 scanners;
- a cheap closed-form control first;
- Exp 5 dropped; Exp 6 replaced by this experiment.

Contract: train split only, selection on SCORPION val, test read once, PLISM is external; single-view inference with no scanner ID, no gate and no scanner set. Pairing unit: tile-level pairs.

## Step 0: Bookkeeping
- `docs/roadmap.md`: add a dated Update. D5 not pursued (G2 below G0 on PLISM). Exp 6 replaced by "Experiment 7" as defined here: model, arms, gates, and the no-gating rule.
- `docs/progress.md`: status rows for Exp 5 (dropped) and Exp 7; open points.

## Step 1: Gate-free closed-form control (offset-span quadratic, ~½ day)
- **Why no gated scanner mixture.** A gate-free *linear* map can't beat G0 on the consensus objective, because G0 is the least-squares linear predictor and its pooled moments already contain the offsets. So the only gate-free way to exploit the offsets is nonlinear.
- **The control.** Exp 4's quadratic map, with nuisance basis = the span of the train per-scanner offsets δ_k (rank 4), at unit variance:
  - Q1-off: squares and pairwise products of the 4 offset coordinates;
  - Q2-off: content directions w_j × offset coordinates.
- **Why it matters.** It is closed-form, has no gate, and γ → ∞ gives G0. It tests whether a smooth nonlinearity in the scanner-signature directions helps, which is the mechanism the CGM would exploit.
- **Code.**
  - `src/sipe/estimators/quadratic.py`: an `offset_directions(views, scanners)` helper beside `nuisance_directions` / `random_directions`, using the δ_k as in `sipe.eval.scanner_offsets`.
  - `src/sipe/eval/quadratic_sweep.py`: the variants `Q1-off` / `Q2-off`, with Q1 extended to products for this basis.
  - Same selection rule, in-run G0 and output layout; new rows in the `quadratic` table block.
- **Leave-one-scanner-out (LOSO) check on val.** For each scanner j: fit on train without j (consensus over the 4 remaining scanners). Report val retrieval for pairs that include j, compared with G0 fitted the same way. This tests how a nonlinear map behaves on an unseen scanner, within SCORPION, and is reused for the CGM. Code: a `--hold-out-scanner` option shared by `quadratic_sweep` and `cgm_select`, so the leave-out logic lives in one helper.

## Step 2: Augmentation study on feature caches
- **Dedicated extraction module, not part of a sweep.** `extract()` currently lives in `src/sipe/eval/soft_shrinkage_sweep.py`. Move it, with the cache writing (features npz + meta csv + provenance), into a module of its own, e.g. `src/sipe/features/extract.py`: a library function plus a CLI `uv run python -m sipe.features.extract`. `soft_shrinkage_sweep` then imports it, with unchanged behavior: check that it gives identical features on a few slides. The new tool is that module, not a new eval script. It reuses the datamodule split. It writes a cache dir in the format of the current cache, adding an `aug` column to meta (copy index), so existing loaders keep working.
  - **Every augmented view goes through the whole encoder:** image → augmentation → full H0-mini forward pass → GAP. No shortcuts such as rotating, flipping or masking the patch-token grid of a clean pass; the dihedral policy included.
  - Augmentations are seeded deterministically. A shared augmentation uses seed = hash(pair_id, copy), so all scanner views get the same transform. A per-view augmentation uses hash(pair_id, copy, scanner).
  - Policies:
    - `none`;
    - `dihedral`: exactly the 8 flips/90° rotations, an exact cache;
    - `content`: K draws of small crops/translations, shared;
    - each current PairedVAE augmentation alone (blur, affine, grayscale, invert, erasing) and the full pipeline, shared;
    - `acquisition`: HED/color jitter + mild blur, per view.
- **2a. Audit (val only, fast).** All audited features come from the full-encoder extraction above (augmented image through the whole H0-mini), never from transforming clean token grids. New `src/sipe/eval/augmentation_audit.py --clean <cache> --augmented <cache>...`. For each policy:
  - the displacement energy of the augmentation relative to the scanner residual energy;
  - the share of that displacement lying in the scanner-offset span (does the augmentation mimic a scanner?);
  - the scanner offsets under augmentation vs clean (reusing `scanner_offsets.level_moments` and `offset_subspace`);
  - raw val top-1;
  - dihedral copies: the flip/rotation invariance of H0-mini GAP.

  Writes `summary.csv`, plus a parser and an `augaudit` block in `src/sipe/eval/results_tables.py`.
- **2b. Train caches** (user command, GPU): `dihedral`, `content` with K=4 and 8, and `acquisition`. Only the policies the audit doesn't rule out: `dihedral` is skipped if GAP is near-invariant.

## Step 3: Consensus generative model (CGM)
- **Generative model (fit time)**, for tile i with views from scanners k:
  - z_i ~ N(0, I_768), shared by the views;
  - x_{i,k} = g(z_i) + δ_k + N_k^{1/2} ε, with g(z) = μ + Lz + h(z).
  - The acquisition latent is integrated out analytically: a Gaussian with full N_k, using a Cholesky factor per scanner.
  - The scanner label k of each target view is used only inside this training likelihood.
  - The exchangeable arm uses δ = 0 and N₀.
- **Encoder (single view):** q(z|x) = N(m(x), Σ), with m(x) = M(x − μ) + r(x).
  - L, M and Σ start at the linear-Gaussian posterior from the G0 moments.
  - The last layers of r and h start at zero, so at init f(x) is the linear-Gaussian posterior mean (≈ G0).
- **Loss**, per tile and per source view s:
  - draw z ~ q(z|x_s) and score *all* views of the tile under their scanner's likelihood;
  - add β·KL and an L2 penalty ρ on the r and h output layers.
  - This is a valid lower bound on the tile's likelihood; it replaces PairedVAE's translation and alignment losses.
- **Inference: f(x) = g(m(x)).** One global map: no gate, no scanner ID, no scanner set, like G0. Caveat: any gain from the offsets means r(x) recognizes scanner signatures (an implicit gate). That is extrapolation on unseen scanners, so the LOSO check is mandatory for the selected configuration.
- **Arms (minimal):**
  - V0: linear, exchangeable; must reproduce G0's val top-1 within noise;
  - V0-S: linear, per-scanner likelihood;
  - V1: nonlinear encoder, exchangeable;
  - **V1-S**: nonlinear encoder, per-scanner (the main arm);
  - V2-S: V1-S plus nonlinear h.
  - No content-dependent nuisance, since G2 didn't transfer.
- **Data arms:** cache A (clean), then B/C/acquisition from Step 2, for V1-S (and V0-S as the linear control on the same data).
- **Code**, following the CLAUDE.md training rules:
  - `src/sipe/model/consensus_gen.py`: network plus a `CGMMap(nn.Module)` [B, 768] → [B, 768], saved in the run dir as a `class_path` map file for `--posthoc`.
  - `src/sipe/training/consensus_gen_module.py`: a `SIPEModule` subclass using `cosine_adamw` and an in-loop val top-1 from `paired_vae_module.cross_scanner_top1`.
  - `src/sipe/data/feature_cache.py`: `PairedFeatureDataModule`. Items are tile locations with all views; it samples one `aug` copy per item (shared) or one per view (the per-view arm); optional hold-out scanner. Splits come from the cache meta.
  - `configs/consensus_gen.yaml`.
  - Training: `uv run sipe fit --config configs/consensus_gen.yaml`.
- **Selection driver `src/sipe/eval/cgm_select.py --runs <dirs> --features-from <clean cache>`:**
  - evaluates every run's map on clean val with `soft_shrinkage_sweep.evaluate` (probes trained on train);
  - selects ρ by the 95% rule over G0;
  - reads test once for the selected run plus G0;
  - writes `summary_val.csv`, `summary_test.csv` and `selected.json`, plus a `cgm` table block.
  - Evaluation always uses *clean* features; augmentation only affects fitting.
- **Seeds:** 3 for the selected configuration.

## Gates
- **G-A (Step 1):** Q1-off / Q2-off above G0 on val beyond the Q2-rand-sized margin, and LOSO not below G0? Then PLISM `--posthoc`. Reported as a control either way.
- **G-B (Step 2):** an augmentation arm is kept only if it raises V1-S val top-1 or shrinks its train/val gap, without the audit showing it erasing the offsets. Backbone-in-loop only if `content` K=8 > K=4 > B.
- **G-C (Step 3, = roadmap D6):**
  - CGM > G0, Q2 and Q-off on PLISM, with similar HEST, and LOSO not below G0 → keep it, run the ablations, add intervals.
  - CGM ≈ the best closed-form model → report the simpler one.

## Verification
- Unit tests:
  - `tests/test_quadratic.py` (offset directions span the δ_k and have unit variance; γ = inf equals G0);
  - `tests/test_extract_features.py` (shared seeds give identical transforms across scanners; dihedral copies are exact);
  - `tests/test_consensus_gen.py`:
    - CGMMap at init equals the closed-form linear posterior map;
    - the likelihood matches a dense computation;
    - the per-scanner and exchangeable likelihoods agree when δ = 0 and N_k = N₀;
    - map-file round-trip through `sipe.bench.posthoc`;
    - the datamodule's shared vs per-view `aug` sampling and hold-out scanner.
- Smoke runs:
  - `sipe fit ... --trainer.fast_dev_run=true --trainer.logger=false --trainer.enable_checkpointing=false`;
  - `quadratic_sweep --variants Q1-off Q2-off` and `extract_features` on a few slides;
  - `augmentation_audit` on the val cache.
- `uv run pytest`, `ruff check` on the new files, and `results_tables --check` after each new block.
- Full sweeps, extractions, PLISM and HEST are handed over as exact commands (CLAUDE.md, long commands). Each check of a finished run updates `docs/progress.md`.

## Order and timing (deadline 2026-10-26)
| when | what |
|---|---|
| day 1 | Step 0, then Step 1 implementation and sweep (user) |
| days 1–2 | `extract_features` + audit, then audit run; CGM implementation |
| days 3–5 | CGM sweeps on cache A, then the augmented caches; LOSO for the selected configuration |
| days 6–8 | PLISM/HEST for the shortlisted models |
| after day 8 | writing |

Q2's test/selection (running) and PLISM feed the same comparison table.
