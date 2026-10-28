"""Does CGM's gain over G0 come from augmentation? Closed-form 2x2 on feature caches.

CGM (Experiment 7) differs from G0 in (a) its data: moments of augmented train views
through the backbone (flips, rot90, zoom) vs the clean cache, (b) its estimator at
init: the linear-Gaussian posterior with per-scanner offsets as a random effect
(`ConsensusGenerativeHead.init_from_moments`, the V0-S init), (c) the number of views
(40 batches = 5,120 tiles x 5 views), (d) ELBO training. This driver isolates (a)-(c)
without training (docs/experiment-log.md, Exp 7 history, 2026-10-15):

- estimators: `G0` (consensus shrinkage of `sipe.estimators.mixture`, K = 1;
  alpha by the 95% rule on val, as `mixture_sweep`) and `CGM-init` (the V0-S init map,
  r = h = 0, `init_ridge` as configs/consensus_gen.yaml; no hyperparameter);
  `--init-noise-weights b ...` adds `CGM-init(b=<b>)` arms: the same init with the
  within-tile noise scaled by b (`init_from_moments(noise_weight=b)`; stronger
  shrinkage). b = 1 is `CGM-init`. An analysis knob, chosen outside this driver
  (e.g. matched to a trained map), never selected here;
  `--fixed-point shared per_scanner` adds `CGM-fp(<noise>,k=<k>)` arms: k block
  coordinate descent steps on the CGM training objective with a linear encoder
  (`sipe.estimators.cgm_fixed_point`), from the CGM-init, for every k of
  `--fp-iters` (`--fp-iters 0` = run to convergence: the optimum is posterior
  collapse, a constant map; the useful maps are early iterates); with
  `--fp-init-noise-weights b ...` the descent starts from CGM-init(b=<b>) (arms
  `CGM-fp(<noise>,k=<k>,b=<b>)`, b = 1 unlabeled). k and b are analysis axes here,
  never selected;
- data: `clean` (train split of `--features-from`) and every `--aug-caches
  label=dir` (`sipe.features.extract`); a unit = one tile (clean) or one tile copy
  (augmented, all scanner views share the transform);
- size (`--n-units`): an integer n = n distinct tiles, one random copy each (5120 =
  the CGM init); `tiles` = every train tile once, one random copy each (= clean count);
  `all` = every unit (K copies per tile). The tile draw is the same for every data
  arm (seeded), so arms at the same size see the same tissue.

Pairing unit: tile-level pairs (all 5 scanners), as G0 and CGM. Evaluation is on the
clean val split with the shared proxy protocol (`sipe.eval.proxy`); nothing is
selected across arms. This is an explanatory ablation: PLISM per pair (`--posthoc`
on the saved maps) is the readout, never a selection criterion. Test is not read
unless `--test`.

Outputs (run dir): `sweep_val.csv` (G0 alpha grid, retrieval only), `arms.json`,
`summary_val.csv` (raw, reference, `--maps`, every arm; `--retrieval-only` skips the
CPU-heavy probes), `map_<est>_<data>_<n>.pt` (posthoc module files),
`summary_test.csv` with `--test`; `fp_history_<noise>_<data>_<n>.csv` (objective per
iteration) and `linear_dist.csv` (relative distance of the linear parts L M of every
CGM arm to every linear CGM map of `--maps`, e.g. trained V0-S).

    uv run python -m sipe.eval.aug_attribution --tag aug_attribution \\
        --features-from <GAP cache> --aug-caches geo=<cache> dihedral=<cache> \\
        --maps V0-S=runs/<a602>/map_cgm.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sipe.bench.posthoc import load_posthoc_map
from sipe.bench.provenance import new_bench_run_dir, write_run_info
from sipe.estimators import cgm_fixed_point
from sipe.estimators.mixture import Gate, Mixture, MixtureShrinkageMap, Moments
from sipe.eval.content_dependence import paired_views
from sipe.eval.mixture_sweep import DEFAULT_ALPHAS, DEFAULT_REFERENCE, SETTING
from sipe.eval.proxy import (
    apply_map,
    cache_encoder,
    load_cache,
    retrieval_metrics,
    select_lambda,
    summarize,
)
from sipe.model.consensus_gen import CGMMap, ConsensusGenerativeHead

log = logging.getLogger(__name__)

ESTIMATORS = ("G0", "CGM-init")
CGM_INIT_RIDGE = 1e-3  # configs/consensus_gen.yaml, model.init_args.init_ridge


def parse_labeled(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        label, sep, path = item.partition("=")
        if not sep or not label:
            raise ValueError(f"expected label=path, got {item!r}.")
        out[label] = Path(path)
    return out


def select_units(
    meta: pd.DataFrame, size: str, tiles: np.ndarray, seed: int
) -> pd.DataFrame:
    """Rows of the chosen units. `tiles` = the seeded tile order shared by all arms."""
    if "aug" not in meta:
        meta = meta.assign(aug=0)
    if size == "all":
        return meta
    n = len(tiles) if size == "tiles" else int(size)
    if n > len(tiles):
        raise ValueError(f"--n-units {n} > {len(tiles)} train tiles.")
    chosen = pd.Series(tiles[:n], name="pair_id")
    copies = np.sort(meta["aug"].unique())
    pick = np.random.default_rng(seed + 1).choice(copies, size=n)
    keys = pd.DataFrame({"pair_id": chosen, "aug": pick})
    return meta.merge(keys, on=["pair_id", "aug"])


def unit_views(feats: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """[units, scanners, D] float64; a unit = (pair_id, aug)."""
    keyed = meta.assign(pair_id=meta["pair_id"] + "#a" + meta["aug"].astype(str))
    return paired_views(feats, keyed, SETTING)[0].views


def g0_model(views: np.ndarray, ridge: float) -> Mixture:
    return Mixture.build(
        Moments.fit(views, np.zeros(len(views), int), 1), "G0", ridge=ridge
    )


def g0_map(model: Mixture, alpha: float) -> MixtureShrinkageMap:
    # K = 1: the gate is irrelevant (responsibility 1).
    dim = model.mu.shape[1]
    return MixtureShrinkageMap.from_fit(
        model, Gate(np.zeros(dim), np.eye(1, dim)), alpha
    )


def cgm_init_map(
    views: np.ndarray,
    device: torch.device,
    ridge: float = CGM_INIT_RIDGE,
    noise_weight: float = 1.0,
) -> CGMMap:
    """The V0-S map at init (linear encoder, per-scanner likelihood), closed form;
    `noise_weight` scales the within-tile noise (1 = the training init)."""
    n, s, d = views.shape
    head = ConsensusGenerativeHead(d, num_scanners=s, per_scanner=True,
                                   encoder_hidden_dim=0).to(device)  # fmt: skip
    head.init_from_moments(
        torch.tensor(views.reshape(-1, d), device=device),
        torch.arange(s, device=device).repeat(n),
        torch.arange(n, device=device).repeat_interleave(s),
        ridge=ridge,
        noise_weight=noise_weight,
    )
    return head.map.cpu().eval()


def retrieval_summary(
    maps: dict[str, torch.nn.Module | None],
    x: np.ndarray,
    meta: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    """`summarize` without the probes: retrieval metrics and relative change."""
    rows = []
    for name, fmap in maps.items():
        out = x if fmap is None else apply_map(fmap, x, device)
        change = np.linalg.norm(out.astype(np.float64) - x, axis=1)
        rows.append(
            {
                "name": name,
                **retrieval_metrics(out, meta, device),
                "relative_change": float(np.mean(change / np.linalg.norm(x, axis=1))),
            }
        )
    return pd.DataFrame(rows)


def save_map(path: Path, fmap: torch.nn.Module, init_args: dict, **info) -> None:
    torch.save(
        {
            "class_path": f"{type(fmap).__module__}.{type(fmap).__name__}",
            "init_args": init_args,
            "state_dict": fmap.state_dict(),
            "pooling": "gap",
            "encoder": "h0-mini",  # overridden by info["encoder"] (the cache's)
            "pairing_unit": "tile",
            **info,
        },
        path,
    )


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-from", type=Path, required=True)
    parser.add_argument("--aug-caches", nargs="*", default=[],
                        help="label=<sipe.features.extract run dir>")  # fmt: skip
    parser.add_argument("--n-units", nargs="+", default=["5120", "tiles", "all"])
    parser.add_argument("--estimators", nargs="+", default=list(ESTIMATORS),
                        choices=ESTIMATORS)  # fmt: skip
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--ridge", type=float, default=1e-4, help="G0 ridge.")
    parser.add_argument("--init-ridge", type=float, default=CGM_INIT_RIDGE)
    parser.add_argument("--init-noise-weights", nargs="+", type=float, default=[1.0],
                        help="CGM-init noise weights b (1 = CGM-init).")  # fmt: skip
    parser.add_argument("--fixed-point", nargs="*", default=[],
                        choices=cgm_fixed_point.NOISE,
                        help="Closed-form V0-S arms (noise model).")  # fmt: skip
    parser.add_argument("--fp-iters", nargs="+", type=int, default=[1, 2, 3],
                        help="Iterates kept as arms (0 = converged).")  # fmt: skip
    parser.add_argument("--fp-init-noise-weights", nargs="+", type=float,
                        default=[1.0],
                        help="Start the descent from CGM-init(b).")  # fmt: skip
    parser.add_argument("--fp-max-iter", type=int, default=2000,
                        help="Cap for --fp-iters 0.")  # fmt: skip
    parser.add_argument("--fp-tol", type=float, default=1e-7,
                        help="Stop at this relative change of L M.")  # fmt: skip
    parser.add_argument("--select-gain-frac", type=float, default=0.95)
    parser.add_argument("--maps", nargs="*", default=[],
                        help="label=<map file> evaluated alongside.")  # fmt: skip
    parser.add_argument("--reference-map", type=Path, default=Path(DEFAULT_REFERENCE),
                        help="'none' to skip.")  # fmt: skip
    parser.add_argument("--probe-train-max", type=int, default=20000)
    parser.add_argument("--test", action="store_true", help="Also read test once.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip the probes (CPU-heavy): retrieval and relative change only.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Tile/copy draw.")
    parser.add_argument("--device", type=int, default=0, help="CUDA index, -1 = CPU.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    if args.alphas[0] != 0.0:
        args.alphas = [0.0, *args.alphas]  # a = 0 is the identity (raw)
    if 1.0 not in args.alphas:
        args.alphas = sorted([*args.alphas, 1.0])
    reference = None if str(args.reference_map) == "none" else args.reference_map
    aug_caches = parse_labeled(args.aug_caches)
    extra_maps = parse_labeled(args.maps)
    if "clean" in aug_caches:
        raise ValueError("'clean' is the --features-from train split.")

    splits = ("train", "val", "test") if args.test else ("train", "val")
    feats, metas = load_cache(args.features_from, splits)
    encoder = cache_encoder(args.features_from)
    data = {"clean": (feats["train"], metas["train"])}
    for label, cache in aug_caches.items():
        f, m = load_cache(cache, ("train",))
        data[label] = (f["train"], m["train"])

    run_dir = new_bench_run_dir(f"scorpion_{args.tag}")
    write_run_info(
        run_dir,
        config={
            "benchmark": "scorpion_aug_attribution",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "pairing_unit": "tile-level matched pairs (pair_id), all scanners;"
            " augmented: one unit per tile copy",
        },
        checkpoint={
            "features_from": str(args.features_from.resolve()),
            "aug_caches": {k: str(v.resolve()) for k, v in aug_caches.items()},
            "maps": {k: str(v.resolve()) for k, v in extra_maps.items()},
            "reference_map": str(reference.resolve()) if reference else None,
        },
        features={"pooling": "gap", "source": "cached SCORPION features"},
    )

    # Tiles present in every data arm (all train tiles unless a cache is partial).
    common = set.intersection(*(set(m["pair_id"]) for _, m in data.values()))
    tiles = np.sort(list(common))
    tiles = tiles[np.random.default_rng(args.seed).permutation(len(tiles))]
    maps: dict[str, torch.nn.Module | None] = {"raw": None}
    if reference is not None:
        maps["reference"] = load_posthoc_map(reference).map
    for label, path in extra_maps.items():
        maps[label] = load_posthoc_map(path).map

    arms, sweep = {}, []
    for data_label, (x, meta) in data.items():
        sizes = args.n_units
        if "aug" not in meta:  # clean: one copy, so 'tiles' = 'all'
            sizes = [s for s in args.n_units if s not in ("tiles", "all")] + ["all"]
        for size in sizes:
            rows = select_units(meta, size, tiles, args.seed)
            views = unit_views(x, rows)
            variants = [
                (est, None) for est in args.estimators if est == "G0"
            ] + [
                (est, b)
                for est in args.estimators
                if est == "CGM-init"
                for b in args.init_noise_weights
            ] + [
                ("CGM-fp", (noise, b))
                for noise in args.fixed_point
                for b in args.fp_init_noise_weights
            ]
            for est, beta in variants:
                if est == "CGM-fp":
                    est = f"CGM-fp({beta[0]})"
                elif beta is not None and beta != 1.0:
                    est = f"CGM-init(b={beta:g})"
                name = f"{est}|{data_label}|{size}"
                info = {
                    "estimator": est,
                    "data": data_label,
                    "n_units": size,
                    "units": len(views),
                    "views": len(views) * views.shape[1],
                }
                if est.startswith("CGM-fp"):
                    file_est = None  # one map per iterate, below
                else:
                    file_est = est if beta in (None, 1.0) else f"CGM-init-b{beta:g}"
                stem = f"map_{file_est}_{data_label}_{size}.pt"
                if est == "G0":
                    model = g0_model(views, args.ridge)
                    grid = []
                    for alpha in args.alphas:
                        fmap = g0_map(model, alpha) if alpha else None
                        val_x = (
                            feats["val"]
                            if fmap is None
                            else apply_map(fmap, feats["val"], device)
                        )
                        metrics = retrieval_metrics(val_x, metas["val"], device)
                        grid.append({"name": name, "lambda": alpha, **metrics})
                        log.info("val %s a=%g: %s", name, alpha, metrics)
                    grid = pd.DataFrame(grid)
                    best = select_lambda(grid, args.select_gain_frac)
                    alpha = float(best["lambda"])
                    fmap = g0_map(model, alpha)
                    at_1 = grid.loc[grid["lambda"] == 1.0, "top_1_accuracy"]
                    info |= {"alpha": alpha, "val_top_1_alpha1": float(at_1.iloc[0])}
                    sweep.append(grid)
                    save_map(run_dir / stem, fmap,
                             {"k": 1, "dim": model.mu.shape[1], "gate_dim": 1},
                             variant="G0", K=1, alpha=alpha, data=data_label,
                             n_units=size, encoder=encoder)  # fmt: skip
                elif est.startswith("CGM-fp"):
                    noise, start = beta
                    tag = "" if start == 1.0 else f",b={start:g}"
                    ftag = "" if start == 1.0 else f"-b{start:g}"
                    converge = 0 in args.fp_iters
                    keep = tuple(k for k in args.fp_iters if k > 0)
                    result = cgm_fixed_point.fit(
                        views, device, noise=noise, ridge=args.init_ridge,
                        noise_weight=start,
                        max_iter=args.fp_max_iter if converge else max(keep),
                        tol=args.fp_tol if converge else 0.0, log_every=1,
                        snapshots=keep,
                    )  # fmt: skip
                    history = pd.DataFrame(result.history)
                    history.to_csv(
                        run_dir / f"fp_history_{noise}{ftag}_{data_label}_{size}.csv",
                        index=False, float_format="%.8g",
                    )  # fmt: skip
                    iterates = dict(result.snapshots)
                    if converge:
                        iterates[0] = result.map
                    for k, fmap in sorted(iterates.items()):
                        at = history.iloc[-1] if k == 0 else (
                            history.loc[history["iter"] == k].iloc[0])  # fmt: skip
                        label = "conv" if k == 0 else f"{k}"
                        est_k = f"CGM-fp({noise},k={label}{tag})"
                        name_k = f"{est_k}|{data_label}|{size}"
                        stem_k = (f"map_CGM-fp-{noise}-k{label}{ftag}"
                                  f"_{data_label}_{size}.pt")  # fmt: skip
                        arm = {
                            **info, "estimator": est_k,
                            "noise": noise, "init_noise_weight": start,
                            "iterations": int(at["iter"]),
                            "converged": bool(result.converged) if k == 0 else False,
                            "J_init": float(history["J"].iloc[0]),
                            "J_final": float(at["J"]),
                            "tr_noise_ratio": float(at["tr_noise_ratio"]),
                            "tr_posterior": float(at["tr_posterior"]),
                        }  # fmt: skip
                        save_map(run_dir / stem_k, fmap,
                                 {"dim": fmap.dim, "encoder_hidden_dim": 0,
                                  "generator_hidden_dim": 0},
                                 arm="V0-S objective, block coordinate descent",
                                 noise=noise, init_noise_weight=start,
                                 iterations=arm["iterations"],
                                 init_ridge=args.init_ridge, data=data_label,
                                 n_units=size, encoder=encoder)  # fmt: skip
                        maps[name_k] = fmap
                        arms[name_k] = {**arm, "map": stem_k}
                        log.info("arm %s: %s", name_k, arms[name_k])
                    continue
                else:
                    fmap = cgm_init_map(views, device, args.init_ridge, beta)
                    info["noise_weight"] = beta
                    save_map(run_dir / stem, fmap,
                             {"dim": fmap.dim, "encoder_hidden_dim": 0,
                              "generator_hidden_dim": 0},
                             arm="V0-S init", init_ridge=args.init_ridge,
                             noise_weight=beta, data=data_label, n_units=size,
                             encoder=encoder)  # fmt: skip
                maps[name] = fmap
                arms[name] = {**info, "map": stem}
                log.info("arm %s: %s", name, arms[name])
            (run_dir / "arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    if sweep:
        pd.concat(sweep).rename(columns={"lambda": "alpha"}).to_csv(
            run_dir / "sweep_val.csv", index=False, float_format="%.6g"
        )

    linear = {
        name: m for name, m in maps.items()
        if isinstance(m, CGMMap) and m.r is None and m.h is None
    }  # fmt: skip
    dist = [
        {"name": a, "to": b, "rel_dist_LM": cgm_fixed_point.linear_distance(
            linear[a], linear[b])}
        for a in linear if a in arms for b in extra_maps if b in linear
    ]  # fmt: skip
    if dist:
        pd.DataFrame(dist).to_csv(
            run_dir / "linear_dist.csv", index=False, float_format="%.6f"
        )

    def logger(name: str, metrics: dict) -> None:
        log.info("summary %s: %s", name, metrics)

    for split in ("val", "test") if args.test else ("val",):
        if args.retrieval_only:
            summary = retrieval_summary(maps, feats[split], metas[split], device)
        else:
            summary = summarize(maps, feats, metas, split, device,
                                args.probe_train_max, logger)  # fmt: skip
        summary.to_csv(
            run_dir / f"summary_{split}.csv", index=False, float_format="%.4f"
        )
        print(summary.to_string(float_format="%.4f"))
    log.info("Augmentation attribution results: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
