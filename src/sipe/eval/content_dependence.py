"""Experiment 1 (roadmap section 3): does acquisition variability depend on content?

Diagnostic on the cached SCORPION features (no map is fitted for inference). For a
unit i seen on S scanners, consensus c_i = mean_s x_{i,s} and residuals
r_{i,s} = x_{i,s} - c_i. The consensus is an empirical target, not scanner-free truth.

Pairing unit (`--settings`):
- `tile_*`: tile-level matched pairs (pair_id, one 224 px tile on every scanner);
- `region_*`: spatial pooling, the mean over a region's 49 tiles per scanner;
- `*_noP1000`: P1000 dropped (it sits ~1 token off the others: registration check).

Per setting and K (`--ks`), fit on train units only:
1. content clusters: PCA (`--content-dim`) of c, then k-means with K clusters;
2. residual subspace: top `--residual-dim` eigenvectors of the global residual second
   moment (energy captured is reported); likelihoods live in this subspace;
3. global model: N(0, S_g), S_g the Ledoit-Wolf second moment of projected residuals;
   content model: cluster k -> N(0, (1 - a) S_k + a S_g), a in `--alphas` (a = 1 is
   the global model); scale-only control: cluster k -> N(0, s_k S_g),
   s_k = tr S_k / tr S_g (content changes the magnitude, not the shape);
4. shuffled control: the content model fit with cluster labels permuted over train
   units (all views of a unit keep one label), evaluated with the true assignments.

Held-out evidence, never test:
- `cv`: GroupKFold over train WSIs (`--folds`); everything is refit per fold. a* is
  the alpha with the best mean CV likelihood (chosen on train only);
- `val`: fit on all of train, held-out val WSIs; per-WSI differences and a WSI
  bootstrap interval.

Metrics (held out): mean log-likelihood difference content - global per residual
dimension (nats/dim; > 0 favors content), energy-prediction MSE ratio (per-unit mean
||r||^2 predicted by the cluster mean vs the global mean; < 1 favors content), and
heterogeneity = weighted RMS of log(cluster energy / overall energy). Cluster stats
(fit and val): energy, top eigenvalues, share of energy in the top nuisance directions
(eigenvectors of the soft-shrinkage B, slide-level deltas), overlap of the leading
residual subspace with the global one, and per-scanner energy shares.

    uv run python -m sipe.eval.content_dependence --tag content_dep \\
        --features-from runs/bench/<feature cache run>   # CLAUDE.md, "Cached features"
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.eval.soft_shrinkage_sweep import slide_to_mean_deltas

log = logging.getLogger(__name__)

SETTINGS = ("tile_all", "tile_noP1000", "region_all", "region_noP1000")
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
N_EIG = 5  # leading eigenvalues / subspace overlap
N_NUISANCE = 10  # top eigenvectors of B for the nuisance-energy share


# --- data ---------------------------------------------------------------------


@dataclass
class Paired:
    """Views [n, S, D] of n units on S scanners (same scanner order for every unit)."""

    views: np.ndarray
    slide: np.ndarray  # [n] WSI of each unit
    scanners: list[str]

    @property
    def consensus(self) -> np.ndarray:
        return self.views.mean(axis=1)

    @property
    def residuals(self) -> np.ndarray:
        return self.views - self.views.mean(axis=1, keepdims=True)

    def subset(self, idx: np.ndarray) -> Paired:
        return Paired(self.views[idx], self.slide[idx], self.scanners)


def paired_views(
    feats: np.ndarray, meta: pd.DataFrame, setting: str
) -> tuple[Paired, pd.DataFrame]:
    """Stack the scanner views of each unit; also return the rows used (for B)."""
    unit, _, scanners = setting.partition("_")
    meta = meta.copy()
    if scanners == "noP1000":
        meta = meta.loc[meta["scanner"] != "P1000"]
    elif scanners != "all":
        raise ValueError(f"Unknown setting {setting!r}.")
    key = {"tile": "pair_id", "region": "region_id"}[unit]
    order = sorted(meta["scanner"].unique())
    df = pd.DataFrame(feats[meta["row"].to_numpy()])
    df[key] = meta[key].to_numpy()
    df["scanner"] = meta["scanner"].to_numpy()
    # Region units: mean over the region's tiles per scanner (spatial pooling).
    means = df.groupby([key, "scanner"], sort=True).mean()
    counts = means.groupby(level=0).size()
    if (counts != len(order)).any():
        raise ValueError(f"{setting}: some units miss a scanner view.")
    units = means.index.get_level_values(0).unique()
    views = means.to_numpy(np.float64).reshape(len(units), len(order), feats.shape[1])
    slide = np.array([u.split("-")[0] for u in units])
    return Paired(views, slide, order), meta


# --- models -------------------------------------------------------------------


def second_moment(r: np.ndarray) -> np.ndarray:
    flat = r.reshape(-1, r.shape[-1])
    return flat.T @ flat / len(flat)


def gaussian_ll(z: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Zero-mean Gaussian log-density of the rows of z [..., d]."""
    d = cov.shape[0]
    chol = np.linalg.cholesky(cov)
    sol = np.linalg.solve(chol, z.reshape(-1, d).T)
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    ll = -0.5 * (d * np.log(2 * np.pi) + logdet + (sol**2).sum(axis=0))
    return ll.reshape(z.shape[:-1])


def lw_moment(z: np.ndarray) -> np.ndarray:
    return ledoit_wolf(z.reshape(-1, z.shape[-1]), assume_centered=True)[0]


@dataclass
class Fit:
    pca: PCA
    kmeans: KMeans
    basis: np.ndarray  # [D, d] residual subspace
    energy_captured: float
    global_cov: np.ndarray
    cluster_covs: list[np.ndarray]  # LW second moment per cluster (true labels)
    shuffled_covs: list[list[np.ndarray]]  # per permutation
    labels: np.ndarray
    unit_energy: np.ndarray  # [n] mean_s ||r||^2 on the fit units
    shuffled_labels: list[np.ndarray]

    def assign(self, data: Paired) -> np.ndarray:
        return self.kmeans.predict(self.pca.transform(data.consensus))


def fit(
    data: Paired,
    k: int,
    content_dim: int,
    residual_dim: int,
    n_perm: int,
    seed: int,
) -> Fit:
    pca = PCA(n_components=content_dim, random_state=seed).fit(data.consensus)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(pca.transform(data.consensus))
    r = data.residuals
    moment = second_moment(r)
    eigval, eigvec = np.linalg.eigh(moment)
    basis = eigvec[:, ::-1][:, :residual_dim]
    z = r @ basis
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(labels) for _ in range(n_perm)]

    def per_cluster(lab: np.ndarray) -> list[np.ndarray]:
        return [lw_moment(z[lab == c]) for c in range(k)]

    return Fit(
        pca=pca,
        kmeans=kmeans,
        basis=basis,
        energy_captured=float(eigval[::-1][:residual_dim].sum() / eigval.sum()),
        global_cov=lw_moment(z),
        cluster_covs=per_cluster(labels),
        shuffled_covs=[per_cluster(lab) for lab in shuffled],
        labels=labels,
        unit_energy=(r**2).sum(axis=-1).mean(axis=1),
        shuffled_labels=shuffled,
    )


def unit_ll(z: np.ndarray, assign: np.ndarray, covs: list[np.ndarray]) -> np.ndarray:
    """Per-unit mean (over views) log-likelihood, each unit under its cluster's cov."""
    out = np.empty(len(z))
    for c, cov in enumerate(covs):
        mask = assign == c
        if mask.any():
            out[mask] = gaussian_ll(z[mask], cov).mean(axis=1)
    return out


def heterogeneity(energy: np.ndarray, assign: np.ndarray, k: int) -> float:
    """Weighted RMS of log(cluster mean energy / overall mean energy)."""
    total, acc, weight = energy.mean(), 0.0, 0.0
    for c in range(k):
        mask = assign == c
        if mask.any():
            acc += mask.sum() * np.log(energy[mask].mean() / total) ** 2
            weight += mask.sum()
    return float(np.sqrt(acc / weight))


def evaluate(
    model: Fit, held: Paired, k: int, alphas: list[float]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-unit log-likelihood differences (content - global, nats/dim) per alpha."""
    assign = model.assign(held)
    r = held.residuals
    z = r @ model.basis
    d = z.shape[-1]
    ll_global = gaussian_ll(z, model.global_cov).mean(axis=1)
    g_trace = np.trace(model.global_cov)
    scale_covs = [np.trace(c) / g_trace * model.global_cov for c in model.cluster_covs]
    diff_scale = (unit_ll(z, assign, scale_covs) - ll_global) / d
    energy = (r**2).sum(axis=-1).mean(axis=1)

    def energy_mse_ratio(labels: np.ndarray) -> float:
        cluster_mean = np.array(
            [
                model.unit_energy[labels == c].mean()
                if (labels == c).any()
                else model.unit_energy.mean()
                for c in range(k)
            ]
        )
        mse_c = np.mean((energy - cluster_mean[assign]) ** 2)
        return float(mse_c / np.mean((energy - model.unit_energy.mean()) ** 2))

    rows, per_unit = [], {}
    for alpha in alphas:

        def mix(covs: list[np.ndarray], a: float = alpha) -> list[np.ndarray]:
            return [(1 - a) * c + a * model.global_cov for c in covs]

        diff = (unit_ll(z, assign, mix(model.cluster_covs)) - ll_global) / d
        shuffled = np.mean(
            [
                (unit_ll(z, assign, mix(covs)) - ll_global) / d
                for covs in model.shuffled_covs
            ],
            axis=0,
        )
        per_unit[f"content_{alpha:g}"] = diff
        per_unit[f"shuffled_{alpha:g}"] = shuffled
        rows.append(
            {
                "alpha": alpha,
                "ll_global": float(ll_global.mean() / d),
                "diff": float(diff.mean()),
                "diff_shuffled": float(shuffled.mean()),
                "diff_scale": float(diff_scale.mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["energy_mse_ratio"] = energy_mse_ratio(model.labels)
    out["energy_mse_ratio_shuffled"] = float(
        np.mean([energy_mse_ratio(lab) for lab in model.shuffled_labels])
    )
    out["heterogeneity"] = heterogeneity(energy, assign, k)
    out["heterogeneity_fit"] = heterogeneity(model.unit_energy, model.labels, k)
    return out, per_unit


def wsi_bootstrap(
    values: np.ndarray, slide: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """95% interval of the unit mean, resampling WSIs."""
    slides = np.unique(slide)
    groups = [values[slide == s] for s in slides]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(len(groups), size=len(groups))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_stats(
    model: Fit, data: Paired, assign: np.ndarray, nuisance: np.ndarray, k: int
) -> list[dict]:
    """Residual statistics per cluster (full feature space)."""
    r = data.residuals
    total = (r**2).sum(axis=-1).mean()
    _, g_vec = np.linalg.eigh(second_moment(r))
    g_top = g_vec[:, ::-1][:, :N_EIG]
    rows = []
    for c in range(k):
        rc = r[assign == c]
        row: dict = {"cluster": c, "n_units": int(len(rc))}
        if len(rc) == 0:
            rows.append(row)
            continue
        moment = second_moment(rc)
        eigval, eigvec = np.linalg.eigh(moment)
        eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
        energy = float(np.trace(moment))
        row.update(
            {
                # Energy per view: E ||r||^2.
                "energy": energy,
                "energy_ratio": float((rc**2).sum(axis=-1).mean() / total),
                **{f"eig{j + 1}": float(eigval[j]) for j in range(N_EIG)},
                "top5_share": float(eigval[:N_EIG].sum() / energy),
                "nuisance_share": float(
                    np.trace(nuisance.T @ moment @ nuisance) / energy
                ),
                "overlap_global_top5": float(
                    np.linalg.norm(eigvec[:, :N_EIG].T @ g_top) ** 2 / N_EIG
                ),
            }
        )
        per_scanner = (rc**2).sum(axis=-1).mean(axis=0)
        for s, name in enumerate(data.scanners):
            row[f"share_{name}"] = float(per_scanner[s] / per_scanner.sum())
        rows.append(row)
    return rows


def nuisance_basis(feats: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Top eigenvectors of B (slide-level scanner deltas, as the eraser)."""
    deltas = slide_to_mean_deltas(feats, meta.reset_index(drop=True)).astype(np.float64)
    _, vec = np.linalg.eigh(deltas.T @ deltas / len(deltas))
    return vec[:, ::-1][:, :N_NUISANCE]


# --- driver -------------------------------------------------------------------


def run_setting(
    setting: str,
    feats: dict[str, np.ndarray],
    metas: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict]]:
    train, train_meta = paired_views(feats["train"], metas["train"], setting)
    val, _ = paired_views(feats["val"], metas["val"], setting)
    nuisance = nuisance_basis(feats["train"], train_meta)
    lik_rows, summary_rows, stat_rows = [], [], []
    for k in args.ks:
        common = dict(
            k=k,
            content_dim=args.content_dim,
            residual_dim=args.residual_dim,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        # Grouped CV over train WSIs: every fold refits clusters, basis and covs.
        cv = []
        folds = GroupKFold(n_splits=args.folds).split(
            np.arange(len(train.views)), groups=train.slide
        )
        for fold, (fit_idx, held_idx) in enumerate(folds):
            model = fit(train.subset(fit_idx), **common)
            res, _ = evaluate(model, train.subset(held_idx), k, args.alphas)
            res["fold"] = fold
            cv.append(res)
        cv = pd.concat(cv, ignore_index=True)
        for rec in cv.to_dict("records"):
            lik_rows.append({"setting": setting, "K": k, "protocol": "cv", **rec})
        cv_mean = cv.groupby("alpha")["diff"].mean()
        alpha_star = float(cv_mean.idxmax())
        cv_star = cv.loc[cv["alpha"] == alpha_star]

        model = fit(train, **common)
        res, per_unit = evaluate(model, val, k, args.alphas)
        for rec in res.to_dict("records"):
            lik_rows.append(
                {"setting": setting, "K": k, "protocol": "val", "fold": -1, **rec}
            )
        val_star = res.loc[res["alpha"] == alpha_star].iloc[0]
        diff = per_unit[f"content_{alpha_star:g}"]
        per_wsi = pd.Series(diff).groupby(val.slide).mean()
        lo, hi = wsi_bootstrap(diff, val.slide, args.n_boot, args.seed)
        summary_rows.append(
            {
                "setting": setting,
                "K": k,
                "n_train_units": len(train.views),
                "n_val_units": len(val.views),
                "energy_captured": model.energy_captured,
                "alpha_star": alpha_star,
                "cv_diff": float(cv_star["diff"].mean()),
                "cv_diff_sd": float(cv_star["diff"].std()),
                "cv_folds_pos": int((cv_star["diff"] > 0).sum()),
                "cv_diff_shuffled": float(cv_star["diff_shuffled"].mean()),
                "cv_diff_scale": float(cv_star["diff_scale"].mean()),
                "val_diff": float(val_star["diff"]),
                "val_ci_lo": lo,
                "val_ci_hi": hi,
                "val_wsi_pos": int((per_wsi > 0).sum()),
                "val_n_wsi": int(len(per_wsi)),
                "val_diff_shuffled": float(val_star["diff_shuffled"]),
                "val_diff_scale": float(val_star["diff_scale"]),
                "val_energy_mse_ratio": float(val_star["energy_mse_ratio"]),
                "val_energy_mse_ratio_shuffled": float(
                    val_star["energy_mse_ratio_shuffled"]
                ),
                "heterogeneity_fit": float(val_star["heterogeneity_fit"]),
                "heterogeneity_val": float(val_star["heterogeneity"]),
            }
        )
        for split, data, assign in (
            ("train", train, model.labels),
            ("val", val, model.assign(val)),
        ):
            for row in cluster_stats(model, data, assign, nuisance, k):
                stat_rows.append({"setting": setting, "K": k, "split": split, **row})
        log.info("%s K=%d: %s", setting, k, summary_rows[-1])
    return lik_rows, summary_rows, stat_rows


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--pooling", default="gap", choices=["gap", "cls"])
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS))
    parser.add_argument("--ks", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--content-dim", type=int, default=32)
    parser.add_argument("--residual-dim", type=int, default=32)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if 1.0 not in args.alphas:
        args.alphas = [*args.alphas, 1.0]  # alpha = 1 is the global model

    feats, metas = {}, {}
    for split in ("train", "val"):  # test is never read here
        feats[split] = np.load(args.features_from / f"features_{split}.npz")[
            args.pooling
        ]
        metas[split] = pd.read_csv(args.features_from / f"meta_{split}.csv")

    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_content_dependence",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "pairing_unit": {
                "tile": "tile-level matched pairs (pair_id)",
                "region": "region mean over 49 tiles per scanner",
            },
        },
        checkpoint={"features_from": str(args.features_from.resolve())},
        features={"pooling": args.pooling, "source": "cached SCORPION features"},
    )
    lik, summary, stats = [], [], []
    for setting in args.settings:
        a, b, c = run_setting(setting, feats, metas, args)
        lik += a
        summary += b
        stats += c
        pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(lik).to_csv(run_dir / "likelihood.csv", index=False)
    pd.DataFrame(stats).to_csv(run_dir / "cluster_stats.csv", index=False)
    (run_dir / "done.json").write_text(json.dumps({"settings": args.settings}) + "\n")
    print(pd.DataFrame(summary).to_string(float_format="%.4f"))
    log.info("Content-dependence results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
