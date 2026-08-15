"""Tests for the mlbvmodel CLI (story-009)."""

import pytest
import tomllib
import yaml
from ase import Atoms
from ase.io import write

from atomchain import bondvalence_workflow as bvw
from tests.test_bondvalence_fit import SEED


@pytest.fixture
def inputs(tmp_path):
    structure = tmp_path / "POSCAR"
    parent = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [2.30, 0, 0]],
        cell=__import__("numpy").eye(3) * 12.0,
    )
    write(structure, parent, format="vasp", vasp5=True, direct=True)
    seed = tmp_path / "seed.yaml"
    seed.write_text(SEED.lstrip(), encoding="utf-8")
    return structure, seed


def test_cli_dry_run_prints_stage_plan(tmp_path, inputs, capsys, monkeypatch):
    structure, seed = inputs
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlbvmodel",
            str(structure),
            "--seed-file",
            str(seed),
            "--output-dir",
            str(tmp_path / "bv"),
            "--dry-run",
        ],
    )
    assert bvw.mlbvmodel_cli() == 0
    out = capsys.readouterr().out
    for stage in bvw.STAGE_ORDER:
        assert stage in out


def test_cli_config_yaml_and_flag_precedence(tmp_path, inputs, monkeypatch):
    structure, seed = inputs
    config_file = tmp_path / "workflow.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "seed_file": str(seed),
                "output_dir": str(tmp_path / "bv"),
                "validation": {"gii_threshold": 0.3},
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run(config=None, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        result = bvw.BVWorkflowResult(
            output_dir=tmp_path,
            manifest={"stages": {}},
            report_md=tmp_path / "report.md",
            report_yaml=tmp_path / "report.yaml",
            fitted_parameters=None,
            validation_metrics={},
            passed=None,
        )
        result.manifest = {"stage_plan": [{"name": n} for n in bvw.STAGE_ORDER]}
        return result

    monkeypatch.setattr(bvw, "run_bondvalence_workflow", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlbvmodel",
            str(structure),
            "--config",
            str(config_file),
            "--gii-threshold",
            "0.15",
            "--dry-run",
        ],
    )
    assert bvw.mlbvmodel_cli() == 0
    config = captured["config"]
    assert isinstance(config, bvw.BVWorkflowConfig)
    assert config.validation.gii_threshold == 0.15  # flag beats YAML
    assert config.output_dir == str(tmp_path / "bv")  # YAML retained


def test_cli_failure_returns_nonzero(tmp_path, inputs, monkeypatch):
    structure, seed = inputs

    def broken_run(config=None, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bvw, "run_bondvalence_workflow", broken_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlbvmodel",
            str(structure),
            "--seed-file",
            str(seed),
            "--output-dir",
            str(tmp_path / "bv"),
        ],
    )
    assert bvw.mlbvmodel_cli() == 1


def test_cli_requires_seed_file(tmp_path, inputs, monkeypatch, capsys):
    structure, _ = inputs
    monkeypatch.setattr(
        "sys.argv",
        ["mlbvmodel", str(structure), "--output-dir", str(tmp_path / "bv")],
    )
    assert bvw.mlbvmodel_cli() == 1
    assert "seed-file" in capsys.readouterr().err


def test_pyproject_registers_mlbvmodel_entry_point():
    pyproject = tomllib.loads(
        (bvw.Path(bvw.__file__).parents[1] / "pyproject.toml").read_text("utf-8")
    )
    assert (
        pyproject["project"]["scripts"]["mlbvmodel"]
        == "atomchain.bondvalence_workflow:mlbvmodel_cli"
    )
    extras = pyproject["project"]["optional-dependencies"]
    assert "easybondvalence" in extras["bondvalence"]
