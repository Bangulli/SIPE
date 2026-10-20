import json

import numpy as np
import pytest
import yaml

from sipe.eval import results_tables as rt
from sipe.eval.hest_pairs import load_kfold, pair_runs, summarize


def write_kfold(run_dir, task, encoder, genes):
    """genes: {name: [pearson per fold]}, in the layout of HEST results_kfold.json."""
    path = run_dir / "results" / f"{encoder}::stamp" / task / encoder
    path.mkdir(parents=True)
    corrs = [{"name": n, "pearson_corrs": v} for n, v in genes.items()]
    folds = len(next(iter(genes.values())))
    data = {
        "mean_per_split": [
            float(np.mean([v[f] for v in genes.values()])) for f in range(folds)
        ],
        "pearson_corrs": corrs,
    }
    (path / "results_kfold.json").write_text(json.dumps(data))


def test_pairs_align_genes_by_name_and_reproduce_task_means(tmp_path):
    ref, method = tmp_path / "ref", tmp_path / "method"
    write_kfold(ref, "COAD", "a", {"G1": [0.1, 0.3], "G2": [0.5, 0.7]})
    # Same genes, other order (HEST sorts by score); method better on fold 0 only.
    write_kfold(method, "COAD", "b", {"G2": [0.9, 0.6], "G1": [0.2, 0.2]})
    write_kfold(ref, "READ", "a", {"G1": [0.4, 0.4], "G2": [0.2, 0.2]})
    write_kfold(method, "READ", "b", {"G1": [0.3, 0.3], "G2": [0.1, 0.1]})

    summary = summarize(pair_runs(load_kfold(ref), load_kfold(method)), bootstraps=200)
    coad = summary.set_index("task").loc["COAD"]
    assert coad["ref"] == pytest.approx(0.4)
    assert coad["method"] == pytest.approx(0.475)
    assert coad["delta"] == pytest.approx(0.075)
    assert (coad["folds_better"], coad["folds"]) == (1, 2)  # +0.25, -0.1
    read = summary.set_index("task").loc["READ"]
    assert read["delta"] == pytest.approx(-0.1)
    assert read["ci_lo"] == pytest.approx(-0.1) and read["ci_hi"] == pytest.approx(-0.1)
    average = summary.set_index("task").loc["average"]
    assert average["delta"] == pytest.approx((0.075 - 0.1) / 2)
    assert (average["folds_better"], average["folds"]) == (1, 4)
    assert average["ci_lo"] <= average["delta"] <= average["ci_hi"]


def test_pairs_refuse_mismatched_genes(tmp_path):
    ref, method = tmp_path / "ref", tmp_path / "method"
    write_kfold(ref, "COAD", "a", {"G1": [0.1, 0.3]})
    write_kfold(method, "COAD", "b", {"G9": [0.1, 0.3]})
    with pytest.raises(ValueError, match="COAD"):
        pair_runs(load_kfold(ref), load_kfold(method))


def bench_dir(root, name):
    run = root / "bench" / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"benchmark": "hest", "tag": name}))
    return run


def test_hest_run_by_tag_needs_all_tasks(tmp_path):
    full = bench_dir(tmp_path, "s_full")
    for task in rt.HEST_TASKS:
        write_kfold(full, task, "a", {"G1": [0.1, 0.2]})
    partial = bench_dir(tmp_path, "s_partial")
    write_kfold(partial, "COAD", "b", {"G1": [0.3, 0.4]})
    with pytest.raises(FileNotFoundError):
        rt._hest_run_by_tag(tmp_path, "s_partial", [])
    assert rt._hest_run_by_tag(tmp_path, "s_full", []).dir == full
