import importlib.util

import yaml


def _load_compare_module():
    path = (
        __import__("pathlib").Path(__file__).parents[1]
        / "examples"
        / "08_training_set_strategies_batio3"
        / "compare_strategy_reports.py"
    )
    spec = importlib.util.spec_from_file_location("compare_strategy_reports", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_strategy_metrics_and_write_outputs(tmp_path):
    compare = _load_compare_module()
    strategy_dir = tmp_path / "batio3_rattle"
    strategy_dir.mkdir()
    (strategy_dir / "report.yaml").write_text(
        yaml.safe_dump(
            {
                "validation_metrics": {
                    "passed": True,
                    "metrics": {
                        "energy_ev": {"rmse": 1.0},
                        "forces_ev_ang": {"rmse": 0.2},
                        "stress_ev_ang3": {"rmse": 0.03},
                    },
                    "sampling": {"n_train": 2, "n_test": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    rows, warnings = compare.collect_strategy_metrics(
        tmp_path, {"rattle": "batio3_rattle", "missing": "missing_dir"}
    )
    outputs = compare.write_comparison_outputs(rows, warnings, tmp_path / "comparison")

    assert rows[0]["strategy"] == "rattle"
    assert rows[0]["output_dir"] == "batio3_rattle"
    assert rows[0]["report"] == "batio3_rattle/report.yaml"
    assert rows[0]["force_rmse_ev_ang"] == 0.2
    assert rows[1]["status"] == "missing"
    assert rows[1]["output_dir"] == "missing_dir"
    assert warnings == ["missing: missing report.yaml and validation/metrics.json"]
    assert outputs["summary_md"].exists()
    assert "rattle" in outputs["summary_md"].read_text(encoding="utf-8")
    assert outputs["metrics_csv"].exists()


def test_collect_strategy_metrics_reads_real_report_metrics_path(tmp_path):
    compare = _load_compare_module()
    strategy_dir = tmp_path / "batio3_rattle"
    metrics_dir = strategy_dir / "validation"
    metrics_dir.mkdir(parents=True)
    (strategy_dir / "report.yaml").write_text(
        yaml.safe_dump(
            {
                "stages": {
                    "validate": {
                        "outputs": {
                            "metrics": "validation/metrics.json",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "metrics.json").write_text(
        yaml.safe_dump(
            {
                "passed": False,
                "metrics": {
                    "energy_ev": {"rmse": 2.0},
                    "forces_ev_ang": {"rmse": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )

    rows, warnings = compare.collect_strategy_metrics(
        tmp_path, {"rattle": "batio3_rattle"}
    )

    assert warnings == []
    assert rows[0]["status"] == "completed"
    assert rows[0]["passed"] is False
    assert rows[0]["force_rmse_ev_ang"] == 0.5
    assert rows[0]["report"] == "batio3_rattle/report.yaml"


def test_collect_strategy_metrics_reads_sample_counts_from_report(tmp_path):
    compare = _load_compare_module()
    strategy_dir = tmp_path / "batio3_rattle"
    metrics_dir = strategy_dir / "validation"
    metrics_dir.mkdir(parents=True)
    (strategy_dir / "report.yaml").write_text(
        yaml.safe_dump(
            {
                "stages": {
                    "sample": {"outputs": {"n_train": 1000, "n_test": 40}},
                    "validate": {"outputs": {"metrics": "validation/metrics.json"}},
                }
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "metrics.json").write_text(
        yaml.safe_dump(
            {
                "passed": False,
                "metrics": {
                    "energy_ev": {"rmse": 2.0},
                    "forces_ev_ang": {"rmse": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )

    rows, _warnings = compare.collect_strategy_metrics(
        tmp_path, {"rattle": "batio3_rattle"}
    )

    assert rows[0]["n_train"] == 1000
    assert rows[0]["n_test"] == 40
