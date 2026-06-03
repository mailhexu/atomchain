import sys

import tomllib
from ase import Atoms

from atomchain import materials_project as mp
from atomchain.materials_project import MaterialsProjectStructure


def atoms():
    return Atoms("Si", cell=[3, 3, 3], pbc=True)


def test_mlmp_fetch_writes_one_structure(monkeypatch, tmp_path):
    calls = []

    def fake_get(material_id, api_key=None, structure_type="final"):
        calls.append((material_id, api_key, structure_type))
        return atoms()

    monkeypatch.setattr(mp, "get_structure_by_id", fake_get)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlmp",
            "fetch",
            "mp-149",
            "--api-key",
            "key",
            "--output",
            str(tmp_path / "Si.vasp"),
        ],
    )

    assert mp.mlmp_cli() == 0
    assert calls == [("mp-149", "key", "final")]
    assert (tmp_path / "Si.vasp").exists()


def test_mlmp_spacegroup_writes_batch_and_manifest(monkeypatch, tmp_path):
    calls = []

    def fake_query(spacegroup, **kwargs):
        calls.append((spacegroup, kwargs))
        return [MaterialsProjectStructure("mp-1", "Si", atoms())]

    monkeypatch.setattr(mp, "get_structures_by_spacegroup", fake_query)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mlmp", "spacegroup", "141", "--output-dir", str(tmp_path), "--limit", "2"],
    )

    assert mp.mlmp_cli() == 0
    assert calls[0][0] == 141
    assert calls[0][1]["limit"] == 2
    assert (tmp_path / "manifest.yaml").exists()


def test_mlmp_query_parses_first_class_and_generic_filters(monkeypatch, tmp_path):
    calls = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return [MaterialsProjectStructure("mp-1", "ZrSi", atoms())]

    monkeypatch.setattr(mp, "query_structures", fake_query)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlmp",
            "query",
            "--elements",
            "Zr",
            "Si",
            "--energy-above-hull",
            "0",
            "0.05",
            "--filter",
            "is_stable=true",
            "--filter",
            "nelements=2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert mp.mlmp_cli() == 0
    call = calls[0]
    assert call["elements"] == ["Zr", "Si"]
    assert call["energy_above_hull"] == (0.0, 0.05)
    assert call["is_stable"] is True
    assert call["nelements"] == 2


def test_mlmp_query_rejects_output_without_structure_field(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlmp",
            "query",
            "--fields",
            "material_id",
            "formula_pretty",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert mp.mlmp_cli() == 1
    assert "structure" in capsys.readouterr().err


def test_mlmp_reports_runtime_error(monkeypatch, tmp_path, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("missing dependency")

    monkeypatch.setattr(mp, "get_structure_by_id", fail)
    monkeypatch.setattr(
        sys, "argv", ["mlmp", "fetch", "mp-149", "--output", str(tmp_path / "Si.vasp")]
    )

    assert mp.mlmp_cli() == 1
    assert "missing dependency" in capsys.readouterr().err


def test_materials_project_packaging_and_docs_are_registered():
    pyproject = tomllib.loads(
        (mp.Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    docs = (mp.Path(__file__).parents[1] / "docs" / "materials_project.md").read_text()

    assert (
        pyproject["project"]["scripts"]["mlmp"]
        == "atomchain.materials_project:mlmp_cli"
    )
    assert (
        "mp-api" in pyproject["project"]["optional-dependencies"]["materials-project"]
    )
    assert "uv sync --extra materials-project" in docs
    assert "MP_API_KEY" in docs
