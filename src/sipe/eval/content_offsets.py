"""Content-dependent scanner shifts: does a scanner's offset depend on the tissue?

Follow-up of the scanner-offset diagnostic (`sipe.eval.scanner_offsets`). The CGM models
the shift of scanner k as a constant, u_{i,k} = g(z_i) + delta_k + noise. This
diagnostic asks whether E[x_{i,k} - x_{i,j} | content] varies with the content of tile
i beyond the constant delta_k - delta_j, on the cached SCORPION features (no map is
fitted for inference; test is never read).

Pairing unit: tile-level matched pairs (pair_id; `*_noP1000`: P1000 dropped).

Design (leak-free pairs). For every scanner pair (k, j), the target is the view
difference d_i = x_{i,k} - x_{i,j} and the content predictor is the mean of the *other*
views, c_i^{-kj} = mean_{s != k, j} x_{i,s}. Regressing the residual to the full
consensus on the consensus would be biased: with scanner-specific noise covariances
Sigma_k, Cov(x_k - c, c) = (Sigma_k - mean Sigma) / S != 0 even when no offset depends
on content. With independent acquisition noise across scanners, the predictor here
shares no noise with the target (its own noise only attenuates the fit: conservative).
Since r_{i,k} = sum_j d_i^{(k,j)} / S, the pairs carry the same information as
per-scanner offsets.

Models of E[d | c^{-kj}] per pair, fitted on train units:
- `global`: the constant delta_k - delta_j (what the CGM and the offset table use);
- `linear_p<p>`: constant + D phi_p(c^{-kj}), phi_p = the top-p PCA scores of the
  train consensus, standardized; ridge (lambda * n) chosen per pair by GroupKFold CV
  over train WSIs (`--folds`, `--ridges`). In the CGM this is a per-scanner linear
  generator L_k = L + D_k;
- `cluster_K<K>`: one mean difference per content cluster (k-means on the top
  `--content-dim` PCA scores of the train consensus, as Experiment 1), assigned from
  c^{-kj};
- `linear_p<max>_shuffled`: control, the linear model fitted with phi permuted over
  train units, evaluated with the true phi (expected <= 0).

Readouts (train CV = out-of-fold predictions; val = fit on all of train):
- explained: 1 - E||d - m(c)||^2 / E||d||^2, `raw` (identity) and `whitened` (the
  ridged inverse of N_0, the train residual second moment of `scanner_offsets`);
- gain: (E_global - E_model) / E_global, the share of what the constant offsets leave
  that the content predicts;
- dll: held-out Gaussian log-likelihood in the top `--residual-dim` residual
  directions of train, N(m(c), Sigma) minus N(delta_kj, Sigma_global) (nats/dim,
  Sigma = Ledoit-Wolf of the out-of-fold train residuals), WSI bootstrap on val;
- offset_span: share of the content-dependent part m(c) - delta_kj that lies in the
  span of the train global offsets (low: the content changes the direction of the
  shift, not only its size along the constant offsets).
Per scanner (`per_scanner.csv`, pairs that contain it) and per content cluster
(`per_cluster.csv`, K = `--report-k`, clusters of the full consensus, descriptive:
Experiment 1 found a high-energy cluster of glass and tissue edges).

    uv run python -m sipe.eval.content_offsets --tag content_offsets_gap \\
        --features-from runs/bench/<feature cache run>   # CLAUDE.md, "Cached features"
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.eval.content_dependence import (
    Paired,
    gaussian_ll,
    lw_moment,
    second_moment,
    wsi_bootstrap,
)
from sipe.eval.scanner_offsets import ridged_inverse, tile_views

log = logging.getLogger(__name__)

SETTINGS = ("tile_all", "tile_noP1000")
DEFAULT_DIMS = (8, 32, 128)
DEFAULT_KS = (4, 8)
DEFAULT_RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


# --- models -------------------------------------------------------------------


def ridge_predict(
    phi_tr: np.ndarray, y_tr: np.ndarray, phi_te: np.ndarray, lams: list[float]
) -> list[np.ndarray]:
    """Ridge predictions with intercept, penalty lam * n, one per lam (one SVD)."""
    xm, ym = phi_tr.mean(axis=0), y_tr.mean(axis=0)
    u, s, vt = np.linalg.svd(phi_tr - xm, full_matrices=False)
    uty = u.T @ (y_tr - ym)
    proj = (phi_te - xm) @ vt.T
    n = len(phi_tr)
    return [ym + proj @ ((s / (s**2 + lam * n))[:, None] * uty) for lam in lams]


def cluster_predict(
    lab_tr: np.ndarray, y_tr: np.ndarray, lab_te: np.ndarray, k: int
) -> np.ndarray:
    """Per-cluster means (the overall mean for clusters empty in the fit units)."""
    means = np.tile(y_tr.mean(axis=0), (k, 1))
    for c in range(k):
        mask = lab_tr == c
        if mask.any():
            means[c] = y_tr[mask].mean(axis=0)
    return means[lab_te]


@dataclass
class Spec:
    name: str
    kind: str  # global | linear | cluster
    size: int = 0  # p or K
    shuffled: bool = False


def model_specs(dims: list[int], ks: list[int]) -> list[Spec]:
    specs = [Spec("global", "global")]
    specs += [Spec(f"linear_p{p}", "linear", p) for p in dims]
    specs += [Spec(f"cluster_K{k}", "cluster", k) for k in ks]
    specs.append(Spec(f"linear_p{max(dims)}_shuffled", "linear", max(dims), True))
    return specs


@dataclass
class Content:
    """Content coordinates fitted on the train consensus."""

    pca: PCA  # top max(dims) components
    scale: np.ndarray  # std of the scores on train
    cluster_pca: PCA
    kmeans: dict[int, KMeans]

    def phi(self, c: np.ndarray) -> np.ndarray:
        return self.pca.transform(c) / self.scale

    def labels(self, c: np.ndarray, k: int) -> np.ndarray:
        return self.kmeans[k].predict(self.cluster_pca.transform(c))


def fit_content(consensus: np.ndarray, args: argparse.Namespace) -> Content:
    pca = PCA(n_components=max(args.dims), random_state=args.seed).fit(consensus)
    scale = pca.transform(consensus).std(axis=0)
    cpca = PCA(n_components=args.content_dim, random_state=args.seed).fit(consensus)
    scores = cpca.transform(consensus)
    kmeans = {
        k: KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(scores)
        for k in sorted(set(args.ks) | {args.report_k})
    }
    return Content(pca, scale, cpca, kmeans)


# --- per pair -------------------------------------------------------------------


@dataclass
class PairResult:
    """Train-CV and val readouts of every model for one scanner pair."""

    cv_energy: dict[str, float]  # sum over train units, raw
    val_raw: dict[str, np.ndarray]  # [n_val] squared residual norm
    val_white: dict[str, np.ndarray]
    val_ll: dict[str, np.ndarray]  # [n_val] log-density in the residual subspace
    span_share: dict[str, float]  # content part in the offset span (val)
    ridge_rows: list[dict]


def fit_pair(
    k: int,
    j: int,
    train: Paired,
    val: Paired,
    content: Content,
    specs: list[Spec],
    metric: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> PairResult:
    white_chol, basis, span = metric
    others = [s for s in range(train.views.shape[1]) if s not in (k, j)]
    y_tr = train.views[:, k] - train.views[:, j]
    y_va = val.views[:, k] - val.views[:, j]
    c_tr = train.views[:, others].mean(axis=1)
    c_va = val.views[:, others].mean(axis=1)
    phi_tr, phi_va = content.phi(c_tr), content.phi(c_va)
    perm = np.random.default_rng(args.seed + 1).permutation(len(phi_tr))
    labels_tr = {kk: content.labels(c_tr, kk) for kk in args.ks}
    labels_va = {kk: content.labels(c_va, kk) for kk in args.ks}
    folds = list(GroupKFold(n_splits=args.folds).split(y_tr, groups=train.slide))
    lams = list(args.ridges)

    def predict(spec: Spec, tr, te_phi, te_lab, lam=None) -> np.ndarray:
        """Fit spec on train units `tr`, predict for (te_phi, te_lab)."""
        if spec.kind == "global":
            return np.broadcast_to(y_tr[tr].mean(axis=0), (len(te_phi), y_tr.shape[1]))
        if spec.kind == "cluster":
            lab = labels_tr[spec.size]
            return cluster_predict(lab[tr], y_tr[tr], te_lab[spec.size], spec.size)
        x = phi_tr[perm] if spec.shuffled else phi_tr
        return ridge_predict(
            x[tr][:, : spec.size], y_tr[tr], te_phi[:, : spec.size], [lam]
        )[0]

    cv_energy, val_raw, val_white, val_ll, span_share = {}, {}, {}, {}, {}
    ridge_rows = []
    for spec in specs:
        lam = None
        if spec.kind == "linear":
            # lambda per pair by grouped CV on train (raw energy).
            x = phi_tr[perm] if spec.shuffled else phi_tr
            errs = np.zeros(len(lams))
            for tr, te in folds:
                # The shuffled control picks lambda on permuted phi on both sides.
                preds = ridge_predict(
                    x[tr][:, : spec.size], y_tr[tr], x[te][:, : spec.size], lams
                )
                errs += [((y_tr[te] - p) ** 2).sum() for p in preds]
            lam = lams[int(errs.argmin())]
            ridge_rows += [
                {"model": spec.name, "lam": lv, "cv_energy": float(e),
                 "selected": lv == lam}
                for lv, e in zip(lams, errs, strict=True)
            ]  # fmt: skip
        # Out-of-fold train residuals (CV energy, covariance for the likelihood).
        oof = np.empty_like(y_tr)
        for tr, te in folds:
            te_lab = {kk: v[te] for kk, v in labels_tr.items()}
            oof[te] = predict(spec, tr, phi_tr[te], te_lab, lam)
        res_tr = y_tr - oof
        cv_energy[spec.name] = float((res_tr**2).sum())
        all_tr = np.arange(len(y_tr))
        pred_va = predict(spec, all_tr, phi_va, labels_va, lam)
        res_va = y_va - pred_va
        val_raw[spec.name] = (res_va**2).sum(axis=1)
        val_white[spec.name] = ((res_va @ white_chol) ** 2).sum(axis=1)
        cov = lw_moment((res_tr - res_tr.mean(axis=0)) @ basis)
        val_ll[spec.name] = gaussian_ll(res_va @ basis, cov)
        if spec.kind != "global":
            part = pred_va - y_tr.mean(axis=0)
            total = float((part**2).sum())
            span_share[spec.name] = (
                float(((part @ span) ** 2).sum()) / total if total > 0 else float("nan")
            )
    return PairResult(cv_energy, val_raw, val_white, val_ll, span_share, ridge_rows)


# --- driver -------------------------------------------------------------------


def run_setting(
    setting: str,
    feats: dict[str, np.ndarray],
    metas: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    train, _ = tile_views(feats["train"], metas["train"], setting)
    val, _ = tile_views(feats["val"], metas["val"], setting)
    r_train = train.residuals
    n0 = second_moment(r_train)
    white_chol = np.linalg.cholesky(ridged_inverse(n0, args.ridge))
    _, vec = np.linalg.eigh(n0)
    basis = vec[:, ::-1][:, : args.residual_dim]
    delta = r_train.mean(axis=0)
    u, sing, _ = np.linalg.svd(delta.T, full_matrices=False)
    span = u[:, sing > sing[0] * 1e-8]
    content = fit_content(train.consensus, args)
    specs = model_specs(args.dims, args.ks)

    scanners = train.scanners
    pairs = list(itertools.combinations(range(len(scanners)), 2))
    results = {}
    ridge_rows = []
    for k, j in pairs:
        log.info("%s: pair %s - %s", setting, scanners[k], scanners[j])
        res = fit_pair(k, j, train, val, content, specs,
                       (white_chol, basis, span), args)  # fmt: skip
        results[(k, j)] = res
        ridge_rows += [
            {"setting": setting, "pair": f"{scanners[k]}-{scanners[j]}", **row}
            for row in res.ridge_rows
        ]

    def stack(field: str, name: str, keep=pairs) -> np.ndarray:
        """[n_val, pairs] values of one model."""
        return np.stack([getattr(results[p], field)[name] for p in keep], axis=1)

    # Val energy of the pair differences themselves (denominator of "explained").
    y_raw = np.stack(
        [((val.views[:, k] - val.views[:, j]) ** 2).sum(axis=1) for k, j in pairs], 1
    )
    y_white = np.stack(
        [(((val.views[:, k] - val.views[:, j]) @ white_chol) ** 2).sum(axis=1)
         for k, j in pairs], 1
    )  # fmt: skip
    cv_total = {
        spec.name: sum(results[p].cv_energy[spec.name] for p in pairs) for spec in specs
    }
    d = args.residual_dim
    summary = []
    g_raw, g_white = stack("val_raw", "global"), stack("val_white", "global")
    g_ll = stack("val_ll", "global")
    for spec in specs:
        raw, white = stack("val_raw", spec.name), stack("val_white", spec.name)
        dll = ((stack("val_ll", spec.name) - g_ll) / d).mean(axis=1)
        per_wsi = pd.Series(dll).groupby(val.slide).mean()
        lo, hi = wsi_bootstrap(dll, val.slide, args.n_boot, args.seed)
        span_vals = [results[p].span_share.get(spec.name, np.nan) for p in pairs]
        summary.append(
            {
                "setting": setting,
                "n_scanners": len(scanners),
                "n_train_units": len(train.views),
                "n_val_units": len(val.views),
                "model": spec.name,
                "kind": spec.kind,
                "size": spec.size,
                "cv_gain_raw": 1.0 - cv_total[spec.name] / cv_total["global"],
                "val_explained_raw": 1.0 - raw.sum() / y_raw.sum(),
                "val_explained_whitened": 1.0 - white.sum() / y_white.sum(),
                "val_gain_raw": 1.0 - raw.sum() / g_raw.sum(),
                "val_gain_whitened": 1.0 - white.sum() / g_white.sum(),
                "val_dll": float(dll.mean()),
                "val_dll_ci_lo": lo,
                "val_dll_ci_hi": hi,
                "val_dll_wsi_pos": int((per_wsi > 0).sum()),
                "n_wsi": int(len(per_wsi)),
                "offset_span_share": float(np.nanmean(span_vals))
                if spec.kind != "global"
                else float("nan"),
            }
        )

    per_scanner = []
    for s, name in enumerate(scanners):
        keep = [p for p in pairs if s in p]
        g = stack("val_raw", "global", keep).sum()
        for spec in specs[1:]:
            per_scanner.append(
                {"setting": setting, "scanner": name, "model": spec.name,
                 "val_gain_raw": 1.0 - stack("val_raw", spec.name, keep).sum() / g}
            )  # fmt: skip

    per_cluster = []
    lab = content.labels(val.consensus, args.report_k)
    for c in range(args.report_k):
        mask = lab == c
        if not mask.any():
            continue
        g = g_raw[mask].sum()
        for spec in specs[1:]:
            per_cluster.append(
                {
                    "setting": setting,
                    "K": args.report_k,
                    "cluster": c,
                    "val_unit_share": float(mask.mean()),
                    "global_energy_share": float(g / g_raw.sum()),
                    "global_energy_ratio": float(g_raw[mask].mean() / g_raw.mean()),
                    "model": spec.name,
                    "val_gain_raw": 1.0 - stack("val_raw", spec.name)[mask].sum() / g,
                }
            )
    log.info("%s:\n%s", setting, pd.DataFrame(summary).to_string())
    return summary, per_scanner, per_cluster, ridge_rows


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--pooling", default="gap", choices=["gap", "cls"])
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS))
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--report-k", type=int, default=8)
    parser.add_argument("--content-dim", type=int, default=32)
    parser.add_argument("--ridges", nargs="+", type=float, default=list(DEFAULT_RIDGES))
    parser.add_argument("--residual-dim", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

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
            "benchmark": "scorpion_content_offsets",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "pairing_unit": "tile-level matched pairs (pair_id), scanner-pair"
            " differences, content from the other views",
        },
        checkpoint={"features_from": str(args.features_from.resolve())},
        features={"pooling": args.pooling, "source": "cached SCORPION features"},
    )
    outputs = {"summary": [], "per_scanner": [], "per_cluster": [], "ridge": []}
    for setting in args.settings:
        for name, rows in zip(outputs, run_setting(setting, feats, metas, args),
                              strict=True):  # fmt: skip
            outputs[name] += rows
        for name, rows in outputs.items():
            pd.DataFrame(rows).to_csv(run_dir / f"{name}.csv", index=False)
    (run_dir / "done.json").write_text(json.dumps({"settings": args.settings}) + "\n")
    print(pd.DataFrame(outputs["summary"]).to_string())
    log.info("Content-offset results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
