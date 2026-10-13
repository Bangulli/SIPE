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


def test_last_val_mean_uses_last_n():
    history = [{"val/a": float(i)} for i in range(20)] + [{"train/b": 1.0}]
    assert rt.last_val_mean(history, "val/a", n=10) == sum(range(10, 20)) / 10
    assert rt.last_val_mean(history, "val/missing") is None
