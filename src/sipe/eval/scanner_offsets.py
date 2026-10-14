"""Scanner-offset diagnostic: is the acquisition nuisance exchangeable across scanners?

G0 (and the consensus VAE sketched for roadmap Experiment 6) pools the residual second
moment over scanners, N_0 = E[r r^T], i.e. treats the scanner effect as exchangeable
noise. This diagnostic measures what that pooling hides, on the cached SCORPION
features (no map is fitted for inference; test is never read).

Pairing unit: tile-level matched pairs (pair_id). Consensus c_i = mean_s x_{i,s},
residuals r_{i,s} = x_{i,s} - c_i (`*_noP1000`: P1000 dropped, registration check).

1. Nested energy decomposition of N_0 (full feature space, count-weighted means, exact):
   r = delta_s                                  global scanner offset
     + (mean over the slide - delta_s)          slide x scanner offset
     + (mean over the region - slide mean)      region x scanner offset
     + (r - region mean)                        tile level
   Shares of E||r||^2 in two metrics: `raw` (identity) and `whitened` (N_0 of train,
   ridged, inverted: every residual direction weighs the same). On val, also the share
   of residual energy removed by subtracting the *train* offsets delta_s (held out).
   Slide x scanner means are what the eraser's slide-level deltas measure.
   Offset subspace Q = span of the global offsets (rank <= S - 1): share of the
   residual energy in Q that the offsets explain, and share of the content
   (consensus) variance in Q, which a map that suppresses Q loses.
2. Held-out Gaussian likelihood of val residuals in the top `--residual-dim` residual
   directions of train, each model minus the exchangeable one N(0, S) (nats/dim, > 0
   favors the scanner-aware model; Ledoit-Wolf covariances as Experiment 1):
   - `offset`: N(delta_s, S_w), per-scanner mean, pooled within-scanner covariance;
   - `cov`: N(0, S_s), per-scanner second moment (shape, offset absorbed);
   - `offset_cov`: N(delta_s, S_s^w);
   - `mixture`: (1/S) sum_s N(delta_s, S_s^w), scanner label marginalized (what a
     single-view acquisition mixture could use, here given the paired consensus).
     Not knowing the label costs at most log(S) / d nats/dim vs `offset_cov`.
   The first three use the scanner label (oracle); `resid_scanner_acc` is the argmax
   of the mixture components on val residuals. WSI bootstrap intervals on val.
3. Per scanner (`scanner_stats.csv`): offset energy share, within-scanner energy,
   offset Mahalanobis length sqrt(delta^T S_w^-1 delta) in the subspace, val Delta ll.

    uv run python -m sipe.eval.scanner_offsets --tag scanner_offsets_gap \\
        --features-from runs/bench/<feature cache run>   # CLAUDE.md, "Cached features"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.eval.content_dependence import (
    Paired,
    gaussian_ll,
    lw_moment,
    paired_views,
    second_moment,
    wsi_bootstrap,
)

log = logging.getLogger(__name__)

SETTINGS = ("tile_all", "tile_noP1000")
LEVELS = ("global", "slide", "region", "tile")
MODELS = ("offset", "cov", "offset_cov", "mixture")


# --- data ---------------------------------------------------------------------


def tile_views(
    feats: np.ndarray, meta: pd.DataFrame, setting: str
) -> tuple[Paired, np.ndarray]:
    """Tile-level views and the region of each unit (unit order of `paired_views`)."""
    if not setting.startswith("tile_"):
        raise ValueError(f"Only tile-level settings: {setting!r}.")
    data, used = paired_views(feats, meta, setting)
    first = used.drop_duplicates("pair_id").set_index("pair_id")["region_id"]
    region = first.loc[sorted(first.index)].to_numpy()
    if len(region) != len(data.views):
        raise ValueError("Unit order mismatch between views and regions.")
    return data, region


# --- energy decomposition -----------------------------------------------------


def group_moment(r: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """sum_g n_g sum_s m_{g,s} m_{g,s}^T / (n S), m = mean of r over the units of g."""
    n, s, d = r.shape
    codes, _ = pd.factorize(groups)
    ind = sp.csr_matrix((np.ones(n), (codes, np.arange(n))))
    counts = np.asarray(ind.sum(axis=1)).ravel()
    means = (ind @ r.reshape(n, s * d)) / counts[:, None]
    weighted = (np.sqrt(counts)[:, None] * means).reshape(-1, d)
    return weighted.T @ weighted / (n * s)


def level_moments(
    r: np.ndarray, slide: np.ndarray, region: np.ndarray
) -> dict[str, np.ndarray]:
    """Second moments of the nested components (they sum to E[r r^T])."""
    n = len(r)
    total = second_moment(r)
    glob = group_moment(r, np.zeros(n, dtype=int))
    slide_m = group_moment(r, slide)
    region_m = group_moment(r, region)
    return {
        "global": glob,
        "slide": slide_m - glob,
        "region": region_m - slide_m,
        "tile": total - region_m,
        "total": total,
    }


def ridged_inverse(m: np.ndarray, ridge: float) -> np.ndarray:
    d = m.shape[0]
    return np.linalg.inv(m + ridge * np.trace(m) / d * np.eye(d))


def energy_rows(
    moments: dict[str, np.ndarray], metrics: dict[str, np.ndarray]
) -> list[dict]:
    rows = []
    for name, weight in metrics.items():
        total = float(np.sum(weight * moments["total"]))  # tr(W M), W symmetric
        row = {"metric": name, "total": total}
        for level in LEVELS:
            row[f"share_{level}"] = float(np.sum(weight * moments[level])) / total
        rows.append(row)
    return rows


def offset_subspace(
    delta: np.ndarray, n0: np.ndarray, consensus: np.ndarray
) -> dict[str, float]:
    """The span Q of the global offsets delta_s (rank <= S - 1), train only.

    offset_fraction: share of the residual energy in Q that the offsets explain,
    tr(Q^T G Q) / tr(Q^T N_0 Q), G = mean_s delta_s delta_s^T. content_share: share of
    the consensus (content) variance that lies in Q, tr(Q^T C Q) / tr C, i.e. what a
    map that suppresses Q loses.
    """
    u, sing, _ = np.linalg.svd(delta.T, full_matrices=False)
    q = u[:, sing > sing[0] * 1e-8]
    glob = delta.T @ delta / len(delta)
    centered = consensus - consensus.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    return {
        "offset_rank": int(q.shape[1]),
        "offset_fraction": float(np.trace(q.T @ glob @ q) / np.trace(q.T @ n0 @ q)),
        "offset_content_share": float(np.trace(q.T @ cov @ q) / np.trace(cov)),
    }


# --- likelihood models ----------------------------------------------------------


class ResidualModels:
    """Exchangeable vs scanner-aware Gaussians in a fixed residual subspace."""

    def __init__(self, z: np.ndarray) -> None:
        # z: [n, S, d] projected train residuals.
        self.exch = lw_moment(z)
        self.delta = z.mean(axis=0)  # [S, d]
        within = z - self.delta
        self.within = lw_moment(within)
        self.cov = [lw_moment(z[:, s]) for s in range(z.shape[1])]
        self.cov_w = [lw_moment(within[:, s]) for s in range(z.shape[1])]

    def component_ll(self, z: np.ndarray) -> np.ndarray:
        """[n, S(view), S(component)]: view under each scanner's offset_cov Gaussian."""
        return np.stack(
            [gaussian_ll(z - self.delta[k], self.cov_w[k]) for k in range(z.shape[1])],
            axis=-1,
        )

    def per_view(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Per-view log-likelihood difference to the exchangeable model, nats/dim."""
        d = z.shape[-1]
        exch = gaussian_ll(z, self.exch)
        comp = self.component_ll(z)
        n_s = z.shape[1]
        own = comp[:, np.arange(n_s), np.arange(n_s)]
        ll = {
            "offset": gaussian_ll(z - self.delta, self.within),
            "cov": np.stack(
                [gaussian_ll(z[:, s], self.cov[s]) for s in range(n_s)], axis=1
            ),
            "offset_cov": own,
            "mixture": logsumexp(comp, axis=-1) - np.log(n_s),
        }
        return {k: (v - exch) / d for k, v in ll.items()}

    def scanner_accuracy(self, z: np.ndarray) -> float:
        pred = self.component_ll(z).argmax(axis=-1)
        return float((pred == np.arange(z.shape[1])).mean())


# --- driver -------------------------------------------------------------------


def run_setting(
    setting: str,
    feats: dict[str, np.ndarray],
    metas: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    train, train_region = tile_views(feats["train"], metas["train"], setting)
    val, val_region = tile_views(feats["val"], metas["val"], setting)
    r_train, r_val = train.residuals, val.residuals

    # 1. energy decomposition
    n0 = second_moment(r_train)
    metrics = {"raw": np.eye(n0.shape[0]), "whitened": ridged_inverse(n0, args.ridge)}
    energy, explained = [], {}
    for split, data, region, r in (
        ("train", train, train_region, r_train),
        ("val", val, val_region, r_val),
    ):
        for row in energy_rows(level_moments(r, data.slide, region), metrics):
            energy.append({"setting": setting, "split": split, **row})
    delta = r_train.mean(axis=0)  # [S, D] train offsets
    left = second_moment(r_val - delta)
    total_val = second_moment(r_val)
    for name, weight in metrics.items():
        explained[name] = 1.0 - float(np.sum(weight * left)) / float(
            np.sum(weight * total_val)
        )
    offset_space = offset_subspace(delta, n0, train.consensus)

    # 2. likelihoods in the top residual subspace of train
    _, vec = np.linalg.eigh(n0)
    basis = vec[:, ::-1][:, : args.residual_dim]
    z_train, z_val = r_train @ basis, r_val @ basis
    models = ResidualModels(z_train)
    lik = []
    summary = {
        "setting": setting,
        "n_train_units": len(train.views),
        "n_val_units": len(val.views),
        "n_scanners": len(train.scanners),
        "residual_dim": args.residual_dim,
    }
    for row in energy:
        if row["split"] == "train":
            for level in LEVELS:
                summary[f"{row['metric']}_{level}"] = row[f"share_{level}"]
    for name, value in explained.items():
        summary[f"val_offset_explained_{name}"] = value
    summary.update(offset_space)
    for split, data, z in (("train", train, z_train), ("val", val, z_val)):
        for model, diff in models.per_view(z).items():
            unit = diff.mean(axis=1)
            per_wsi = pd.Series(unit).groupby(data.slide).mean()
            lo, hi = wsi_bootstrap(unit, data.slide, args.n_boot, args.seed)
            rec = {
                "setting": setting,
                "split": split,
                "model": model,
                "diff": float(unit.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "wsi_pos": int((per_wsi > 0).sum()),
                "n_wsi": int(len(per_wsi)),
            }
            lik.append(rec)
            if split == "val":
                summary[f"val_diff_{model}"] = rec["diff"]
                summary[f"val_ci_lo_{model}"] = lo
                summary[f"val_ci_hi_{model}"] = hi
    summary["val_resid_scanner_acc"] = models.scanner_accuracy(z_val)

    # 3. per scanner
    within_inv = np.linalg.inv(models.within)
    val_diff = models.per_view(z_val)["offset_cov"].mean(axis=0)
    total_energy = (r_train**2).sum(axis=-1).mean()
    stats = []
    for s, name in enumerate(train.scanners):
        offset_energy = float(delta[s] @ delta[s])
        stats.append(
            {
                "setting": setting,
                "scanner": name,
                # Shares of the pooled residual energy E||r||^2 (sum over scanners
                # of offset + within shares = 1).
                "offset_energy_share": offset_energy
                / len(train.scanners)
                / total_energy,
                "within_energy_share": float(
                    ((r_train[:, s] - delta[s]) ** 2).sum(axis=-1).mean()
                )
                / len(train.scanners)
                / total_energy,
                "offset_mahalanobis": float(
                    np.sqrt(models.delta[s] @ within_inv @ models.delta[s])
                ),
                "val_diff_offset_cov": float(val_diff[s]),
            }
        )
    log.info("%s: %s", setting, summary)
    return summary, energy, lik, stats


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--pooling", default="gap", choices=["gap", "cls"])
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS))
    parser.add_argument("--residual-dim", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-4)
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
            "benchmark": "scorpion_scanner_offsets",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "pairing_unit": "tile-level matched pairs (pair_id)",
        },
        checkpoint={"features_from": str(args.features_from.resolve())},
        features={"pooling": args.pooling, "source": "cached SCORPION features"},
    )
    summary, energy, lik, stats = [], [], [], []
    for setting in args.settings:
        a, b, c, d = run_setting(setting, feats, metas, args)
        summary.append(a)
        energy += b
        lik += c
        stats += d
        pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(energy).to_csv(run_dir / "energy.csv", index=False)
    pd.DataFrame(lik).to_csv(run_dir / "likelihood.csv", index=False)
    pd.DataFrame(stats).to_csv(run_dir / "scanner_stats.csv", index=False)
    (run_dir / "done.json").write_text(json.dumps({"settings": args.settings}) + "\n")
    print(pd.DataFrame(summary).set_index("setting").T.to_string())
    log.info("Scanner-offset results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
