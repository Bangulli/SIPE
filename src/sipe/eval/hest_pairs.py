"""HEST paired comparison: per-gene, per-fold Pearson of two HEST runs, side by side.

Every `sipe.bench.hest` run keeps HEST's per-task k-fold output,
`results/<run>/<task>/<encoder>/results_kfold.json`: for each of the 50 genes, the
Pearson correlation on each patient-level fold (`pearson_corrs`), and the per-fold mean
(`mean_per_split`). Nothing is recomputed: this module reads those values, aligns two
runs on (task, fold, gene) and summarizes method minus reference.

HEST has 2 to 6 folds per task, so differences between representations cannot be tested
over patients per task. What is reported (`summarize`):
- Δ per task = difference of the task means (mean over folds and genes; equals the
  difference of the official `pearson_mean`);
- folds better = folds whose mean Pearson is higher than the reference's (paired: same
  test patients, same genes);
- gene CI = percentile interval of a bootstrap over the 50 genes, folds fixed: the
  uncertainty from the gene panel *given these test patients*, not over patients;
- average row: Δ of the 10-task average, with a bootstrap over tasks (each draw
  resamples the tasks, then the genes within each drawn task).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAPS = 2000
SEED = 0


def load_kfold(run_dir: Path) -> pd.DataFrame:
    """Long table (task, fold, gene, pearson) of one HEST run dir."""
    found = sorted(run_dir.glob("results/*/*/*/results_kfold.json"))
    if not found:
        raise FileNotFoundError(f"no results_kfold.json under {run_dir}/results")
    records = []
    for path in found:
        task = path.parents[1].name
        data = json.loads(path.read_text())
        for gene in data["pearson_corrs"]:
            for fold, value in enumerate(gene["pearson_corrs"]):
                records.append((task, fold, gene["name"], float(value)))
    df = pd.DataFrame(records, columns=["task", "fold", "gene", "pearson"])
    if df.duplicated(["task", "fold", "gene"]).any():
        raise ValueError(f"duplicate (task, fold, gene) in {run_dir}")
    return df


def pair_runs(ref: pd.DataFrame, method: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (task, fold, gene); refuses runs whose folds or genes differ."""
    keys = ["task", "fold", "gene"]
    merged = ref.merge(method, on=keys, how="outer", suffixes=("_ref", "_method"))
    missing = merged["pearson_ref"].isna() | merged["pearson_method"].isna()
    if missing.any():
        tasks = sorted(merged.loc[missing, "task"].unique())
        raise ValueError(f"folds/genes differ between the runs in tasks {tasks}")
    merged["delta"] = merged["pearson_method"] - merged["pearson_ref"]
    return merged


def _gene_matrix(task_df: pd.DataFrame) -> np.ndarray:
    """[genes, folds] of Δ for one task."""
    return task_df.pivot(index="gene", columns="fold", values="delta").to_numpy()


def summarize(
    paired: pd.DataFrame, bootstraps: int = BOOTSTRAPS, seed: int = SEED
) -> pd.DataFrame:
    """One row per task plus `average`: ref, method, Δ, gene CI, folds better."""
    rng = np.random.default_rng(seed)
    rows = []
    task_draws = {}
    for task, task_df in paired.groupby("task", sort=True):
        delta = _gene_matrix(task_df)
        n_genes = delta.shape[0]
        idx = rng.integers(0, n_genes, size=(bootstraps, n_genes))
        draws = delta[idx].mean(axis=(1, 2))
        task_draws[task] = draws
        fold_delta = delta.mean(axis=0)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "task": task,
                "ref": task_df["pearson_ref"].mean(),
                "method": task_df["pearson_method"].mean(),
                "delta": delta.mean(),
                "ci_lo": lo,
                "ci_hi": hi,
                "folds_better": int((fold_delta > 0).sum()),
                "folds": len(fold_delta),
            }
        )
    tasks = pd.DataFrame(rows)
    # Average over tasks: resample tasks, and inside each the genes (draws above).
    draws = np.stack([task_draws[t] for t in tasks["task"]])  # [tasks, bootstraps]
    pick = rng.integers(0, len(tasks), size=(bootstraps, len(tasks)))
    average_draws = draws[pick, np.arange(bootstraps)[:, None]].mean(axis=1)
    lo, hi = np.percentile(average_draws, [2.5, 97.5])
    average = {
        "task": "average",
        "ref": tasks["ref"].mean(),
        "method": tasks["method"].mean(),
        "delta": tasks["delta"].mean(),
        "ci_lo": lo,
        "ci_hi": hi,
        "folds_better": int(tasks["folds_better"].sum()),
        "folds": int(tasks["folds"].sum()),
    }
    return pd.concat([tasks, pd.DataFrame([average])], ignore_index=True)
