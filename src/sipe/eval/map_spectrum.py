"""Spectrum of fitted maps in G0's eigenbasis: does training only shrink more?

Basis: the generalized eigenvectors of C v = lambda N v (C = Cov(consensus), N =
E[r r^T], plug-in moments of the train split, tile-level matched pairs over all five
scanners, ridged exactly as G0 in `sipe.estimators.mixture`), normalized V^T N V = I.
lambda_k is the content-to-scanner variance ratio of direction k. With U = N V, G0's
operator P_alpha = C (C + alpha N)^-1 is diagonal: P_alpha u_k = lambda_k / (lambda_k +
alpha) u_k.

For each map f, J = the Jacobian of f (exact for affine maps and the linear-encoder
CGM; for nonlinear maps the mean Jacobian over `--n-points` val embeddings), and
A = V^T J U (= U^-1 J U). Reported per map:
- `gain_k` = A_kk, the share of direction k kept (G0: lambda_k / (lambda_k + alpha));
- `offdiag_frac` = ||A - diag A||_F^2 / ||A||_F^2: mixing between directions (G0: 0);
- `alpha_fit`: the single alpha whose G0 gains best fit gain_k (least squares over k,
  log grid), and `fit_rmse`: how far the map is from the G0 family;
- `kept_content` = tr(J C J^T) / tr(C), `kept_scanner` = tr(J N J^T) / tr(N) (raw
  moments);
- `jac_spread` (nonlinear maps): mean over points of ||J(x) - J_mean||_F / ||J_mean||_F.

Outputs (run dir): `summary.csv`, `spectrum.csv` (gain per map and direction),
`spectrum_bins.csv` (mean gain and G0-equivalent alpha per lambda decile),
`spectrum.png`. Nothing is selected here; no PLISM data is read.

    uv run python -m sipe.eval.map_spectrum --tag map_spectrum \\
        --features-from runs/bench/<feature cache run> \\
        --maps G0_a8=<map file> V0-S=<map file> ...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch

from sipe.bench.posthoc import load_posthoc_map
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.estimators.mixture import Moments, _ridge
from sipe.eval.content_dependence import paired_views

SETTING = "tile_all"
ALPHA_GRID = np.logspace(-2, 3, 2001)


def g0_basis(views: np.ndarray, ridge: float) -> tuple[np.ndarray, ...]:
    """lambda (descending), V (V^T N V = I), raw C, raw N, mean."""
    m = Moments.fit(views, np.zeros(len(views), int), 1)
    lam, v = scipy.linalg.eigh(_ridge(m.C_all, ridge), _ridge(m.N_all, ridge))
    order = np.argsort(lam)[::-1]
    return lam[order], v[:, order], m.C_all, m.N_all, m.mu_all


def mean_jacobian(
    fmap: torch.nn.Module, points: np.ndarray, batch: int
) -> tuple[np.ndarray, float]:
    """Mean Jacobian over the points and the mean relative spread around it."""

    def f(x: torch.Tensor) -> torch.Tensor:
        return fmap(x[None])[0]

    jac = torch.func.vmap(torch.func.jacrev(f))
    x = torch.as_tensor(points, dtype=torch.float32)
    with torch.no_grad():
        js = [jac(x[i : i + batch]).double() for i in range(0, len(x), batch)]
    js = torch.cat(js)
    j_mean = js.mean(0)
    spread = (js - j_mean).flatten(1).norm(dim=1) / j_mean.norm()
    return j_mean.numpy(), float(spread.mean())


def fit_alpha(lam: np.ndarray, gain: np.ndarray) -> tuple[float, float]:
    pred = lam[None] / (lam[None] + ALPHA_GRID[:, None])
    err = ((pred - gain[None]) ** 2).mean(axis=1)
    i = int(np.argmin(err))
    return float(ALPHA_GRID[i]), float(np.sqrt(err[i]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--maps", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--pooling", default="gap")
    parser.add_argument("--ridge", type=float, default=1e-4, help="As G0.")
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    maps = dict(m.split("=", 1) for m in args.maps)

    feats, metas = {}, {}
    for split in ("train", "val"):
        feats[split] = np.load(args.features_from / f"features_{split}.npz")[
            args.pooling
        ].astype(np.float32)
        metas[split] = pd.read_csv(args.features_from / f"meta_{split}.csv")

    run_dir = new_bench_run_dir(args.tag)
    write_run_info(
        run_dir,
        config={
            "benchmark": "map_spectrum",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "pairing_unit": "tile-level matched pairs (pair_id), all scanners",
            "basis": "generalized eigenvectors of (C, N), train split, G0 ridge",
        },
        checkpoint={"features_from": str(args.features_from.resolve()), "maps": maps},
        features={"pooling": args.pooling, "source": "cached SCORPION features"},
    )

    train, _ = paired_views(feats["train"], metas["train"], SETTING)
    lam, v, c, n, _ = g0_basis(train.views, args.ridge)
    u = _ridge(n, args.ridge) @ v  # columns u_k; V^T U = I
    rng = np.random.default_rng(args.seed)
    points = feats["val"][rng.choice(len(feats["val"]), args.n_points, replace=False)]
    bins = np.array_split(np.arange(len(lam)), 10)  # deciles, high lambda first

    rows, spec_rows, bin_rows = [], [], []
    for name, path in maps.items():
        loaded = load_posthoc_map(path)
        j, spread = mean_jacobian(loaded.map, points, args.batch)
        a = v.T @ j @ u
        gain = np.diag(a).copy()
        alpha, rmse = fit_alpha(lam, gain)
        rows.append({
            "map": name,
            "kind": loaded.kind,
            "alpha_fit": alpha,
            "fit_rmse": rmse,
            "offdiag_frac": float(((a - np.diag(gain)) ** 2).sum() / (a**2).sum()),
            "kept_content": float(np.trace(j @ c @ j.T) / np.trace(c)),
            "kept_scanner": float(np.trace(j @ n @ j.T) / np.trace(n)),
            "jac_spread": spread,
            "path": path,
        })  # fmt: skip
        spec_rows += [
            {"map": name, "k": k, "lambda": lam[k], "gain": gain[k]}
            for k in range(len(lam))
        ]
        for b, idx in enumerate(bins):
            g = gain[idx]
            eff = lam[idx] * (1 - g) / np.clip(g, 1e-9, None)
            bin_rows.append({
                "map": name,
                "decile": b + 1,
                "lambda_min": lam[idx].min(),
                "lambda_max": lam[idx].max(),
                "gain_mean": g.mean(),
                "alpha_eff_median": float(np.median(eff)),
            })  # fmt: skip
        logging.info("%s: %s", name, rows[-1])

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False, float_format="%.6g")
    spec = pd.DataFrame(spec_rows)
    spec.to_csv(run_dir / "spectrum.csv", index=False, float_format="%.6g")
    pd.DataFrame(bin_rows).to_csv(
        run_dir / "spectrum_bins.csv", index=False, float_format="%.6g"
    )
    plot(spec, run_dir / "spectrum.png")
    print(summary.drop(columns="path").to_string(index=False))
    print(f"run dir: {run_dir}")


def plot(spec: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, df in spec.groupby("map", sort=False):
        ax.plot(df["lambda"], df["gain"], lw=1, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("content / scanner variance ratio lambda_k")
    ax.set_ylabel("gain along direction k")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
