from __future__ import annotations

import json
from pathlib import Path

import yaml

from sipe.eval import results_tables as rt

PLISM_CSV = """\
,cosine_similarity,top_1_accuracy
inter-scanner,0.966 (0.048) ; 0.979 (0.013),0.839 (0.187) ; 0.891 (0.154)
inter-staining,0.845 (0.251) ; 0.946 (0.028),0.293 (0.207) ; 0.296 (0.301)
"inter-scanner, inter-staining",0.839 (0.252) ; 0.939,0.215 (0.167) ; 0.191 (0.235)
all,0.848 (0.245) ; 0.941 (0.029),0.267 (0.233) ; 0.218 (0.272)
"""


def bench_dir(
    root: Path, name: str, benchmark: str, representation: str = "z_gap"
) -> Path:
    run = root / "bench" / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump({"benchmark": benchmark, "tag": name})
    )
    train = root / "train_run"
    train.mkdir(exist_ok=True)
    (train / "wandb.txt").write_text("id: abc123\n")
    provenance = {
        "checkpoint": {"train_run_dir": str(train)},
        "features": {"representation": representation},
        "repos": {"sipe": {"commit": "0123456789", "dirty": True}},
    }
    (run / "provenance.json").write_text(json.dumps(provenance))
    return run


def test_plism_reads_top1_mean(tmp_path):
    run = bench_dir(tmp_path, "s_x_plism", "plism")
    metrics = run / "metrics" / "8139_tiles" / "tag"
    metrics.mkdir(parents=True)
    (metrics / "results.csv").write_text(PLISM_CSV)
    bench_dir(tmp_path, "s_smoke", "plism")

    table = rt.plism_table(tmp_path, {"exclude": ["*smoke*"]})
    row = next(line for line in table.splitlines() if "s_x_plism" in line)
    assert "| 0.839 | 0.293 | 0.215 | 0.267 |" in row
    assert "abc123" in row and "0123456*" in row
    assert "s_smoke" not in table



def test_include_keeps_only_matching_runs(tmp_path):
    for name in ("s_a_uni2h_x", "s_b_other"):
        metrics = bench_dir(tmp_path, name, "plism") / "metrics" / "8139_tiles" / "t"
        metrics.mkdir(parents=True)
        (metrics / "results.csv").write_text(PLISM_CSV)

    table = rt.plism_table(tmp_path, {"include": ["*uni2h*"]})
    assert "s_a_uni2h_x" in table and "s_b_other" not in table
    table = rt.plism_table(tmp_path, {"exclude": ["*uni2h*"]})
    assert "s_a_uni2h_x" not in table and "s_b_other" in table

def test_hest_average_only_for_full_task_set(tmp_path):
    run = bench_dir(tmp_path, "s_hest", "hest")
    results = run / "results" / "tag"
    results.mkdir(parents=True)
    data = {
        "average": {"tag": 0.5},
        "results": [
            {
                "dataset_name": "IDC",
                "results": [{"pearson_mean": 0.5, "pearson_std": 0.1}],
            }
        ],
    }
    (results / "dataset_results.json").write_text(json.dumps(data))
    row = next(
        line for line in rt.hest_table(tmp_path, {}).splitlines() if "s_hest" in line
    )
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2 + rt.HEST_TASKS.index("IDC")] == "0.500 (0.100)"
    assert cells[2 + len(rt.HEST_TASKS)] == "–"  # IDC only: no average


def test_render_replaces_block_and_check_detects_staleness(tmp_path):
    run = bench_dir(tmp_path, "s_proxy", "scorpion_retrieval")
    (run / "summary.csv").write_text(
        "representation,top_1_accuracy,probe_logreg_acc\n"
        "backbone_gap,0.8,0.95\nz_gap,0.9,0.85\n"
    )
    doc = tmp_path / "progress.md"
    doc.write_text(
        "intro\n<!-- results:scorpion -->\nstale 0.123\n"
        "<!-- /results:scorpion -->\nend\n"
    )

    assert rt.main(["--check", "--doc", str(doc), "--runs", str(tmp_path)]) == 1
    assert rt.main(["--write", "--doc", str(doc), "--runs", str(tmp_path)]) == 0
    text = doc.read_text()
    assert "stale" not in text and text.startswith("intro\n") and text.endswith("end\n")
    assert "| s_proxy | z_gap | 0.900 |" in text
    assert rt.main(["--check", "--doc", str(doc), "--runs", str(tmp_path)]) == 0


def test_write_and_check_cover_every_doc(tmp_path):
    run = bench_dir(tmp_path, "s_proxy", "scorpion_retrieval")
    (run / "summary.csv").write_text(
        "representation,top_1_accuracy,probe_logreg_acc\nz_gap,0.9,0.85\n"
    )
    block = "<!-- results:scorpion -->\nstale\n<!-- /results:scorpion -->\n"
    docs = [tmp_path / "a.md", tmp_path / "b.md"]
    for doc in docs:
        doc.write_text(block)
    args = ["--doc", *map(str, docs), "--runs", str(tmp_path)]

    assert rt.main(["--write", *args]) == 0
    assert all("| s_proxy | z_gap | 0.900 |" in d.read_text() for d in docs)
    docs[1].write_text(block)  # one stale doc is enough to fail the check
    assert rt.main(["--check", *args]) == 1


def test_last_val_mean_uses_last_n():
    history = [{"val/a": float(i)} for i in range(20)] + [{"train/b": 1.0}]
    assert rt.last_val_mean(history, "val/a", n=10) == sum(range(10, 20)) / 10
    assert rt.last_val_mean(history, "val/missing") is None


def test_content_table_reads_summary(tmp_path):
    run = bench_dir(tmp_path, "s_content", "scorpion_content_dependence")
    row = {
        "setting": "tile_all", "K": 4, "alpha_star": 0.25, "cv_diff": 0.1011,
        "cv_diff_sd": 0.0092, "cv_folds_pos": 5, "cv_diff_shuffled": -0.0032,
        "cv_diff_scale": 0.05, "val_diff": 0.1133, "val_ci_lo": 0.0785,
        "val_ci_hi": 0.1469, "val_wsi_pos": 5, "val_n_wsi": 5,
        "val_diff_shuffled": -0.0082, "val_diff_scale": 0.06,
        "val_energy_mse_ratio": 0.6469, "val_energy_mse_ratio_shuffled": 0.9929,
        "heterogeneity_fit": 0.2417, "heterogeneity_val": 0.2433,
    }  # fmt: skip
    import pandas as pd

    pd.DataFrame([row]).to_csv(run / "summary.csv", index=False)
    table = rt.content_table(tmp_path, {})
    line = next(x for x in table.splitlines() if "tile_all" in x)
    assert "| 0.101 ± 0.009 |" in line and "0.113 [0.079, 0.147]" in line
    assert "| 5/5 |" in line and "0.647 (0.993)" in line


def test_plism_pairs_table_delta_vs_reference(tmp_path):
    from test_plism_pairs import fake_metrics

    for name, seed in (("s_ref", 0), ("s_m", 1)):
        run = bench_dir(tmp_path, name, "plism")
        out = run / "metrics" / "8139_tiles" / name
        out.mkdir(parents=True)
        fake_metrics(seed).to_csv(out / "metrics.csv", index=False)
    table = rt.plism_pairs_table(tmp_path, {"ref": ["s_ref"], "tags": ["s_m"]})
    delta = fake_metrics(1)["top_1_accuracy"] - fake_metrics(0)["top_1_accuracy"]
    row = next(line for line in table.splitlines() if "| all |" in line)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[:3] == ["s_m", "all", "6"]
    assert cells[5] == f"{delta.mean():+.3f}"
    assert "bench/s_m" in cells[-1]
    assert rt.plism_pairs_table(tmp_path, {}).startswith("(needs")


def test_plism_drop_stain_recomputes_from_metrics(tmp_path):
    from test_plism_pairs import fake_metrics

    run = bench_dir(tmp_path, "s_x_plism", "plism")
    out = run / "metrics" / "8139_tiles" / "s_x_plism"
    out.mkdir(parents=True)
    (out / "results.csv").write_text(PLISM_CSV)
    df = fake_metrics(0)
    df.to_csv(out / "metrics.csv", index=False)

    full = rt.plism_top1_from_metrics(out / "metrics.csv", [])
    assert full["all"] == df["top_1_accuracy"].mean()
    keep = (df["staining_a"] != "HR") & (df["staining_b"] != "HR")
    inter_scanner = keep & (df["staining_a"] == df["staining_b"])
    table = rt.plism_table(tmp_path, {"drop_stain": ["HR"]})
    row = next(line for line in table.splitlines() if "s_x_plism" in line)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == f"{df.loc[inter_scanner, 'top_1_accuracy'].mean():.3f}"
    assert cells[5] == f"{df.loc[keep, 'top_1_accuracy'].mean():.3f}"
    assert "without the stain(s) HR" in table


def test_plism_pairs_table_drop_stain(tmp_path):
    from test_plism_pairs import fake_metrics

    for name, seed in (("s_ref", 0), ("s_m", 1)):
        run = bench_dir(tmp_path, name, "plism")
        out = run / "metrics" / "8139_tiles" / name
        out.mkdir(parents=True)
        fake_metrics(seed).to_csv(out / "metrics.csv", index=False)
    opts = {"ref": ["s_ref"], "tags": ["s_m"], "drop_stain": ["HR"]}
    table = rt.plism_pairs_table(tmp_path, opts)
    ref, m = fake_metrics(0), fake_metrics(1)
    keep = (ref["staining_a"] != "HR") & (ref["staining_b"] != "HR")
    delta = (m["top_1_accuracy"] - ref["top_1_accuracy"])[keep]
    row = next(line for line in table.splitlines() if "| all |" in line)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == str(int(keep.sum()))
    assert cells[5] == f"{delta.mean():+.3f}"
    assert ", HR |" not in table
