import builtins
from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.io import read

from atomchain import materials_project as mp
from atomchain.materials_project import MaterialsProjectStructure

ORIGINAL_GET_MPRESTER_CLASS = mp._get_mprester_class


def silicon_atoms():
    return Atoms(
        "Si2",
        cell=[5.43, 5.43, 5.43],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        pbc=True,
    )


class FakeRester:
    instances = []

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.materials = SimpleNamespace(summary=SimpleNamespace(search=self.search))
        self.search_calls = []
        FakeRester.instances.append(self)

    def get_structure_by_material_id(self, material_id, **kwargs):
        self.last_material_id = material_id
        self.last_structure_kwargs = kwargs
        return silicon_atoms()

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [
            SimpleNamespace(
                material_id="mp-149",
                formula_pretty="Si",
                symmetry=SimpleNamespace(number=227, symbol="Fd-3m"),
                structure=silicon_atoms(),
            )
        ]


@pytest.fixture(autouse=True)
def reset_fake_rester(monkeypatch):
    FakeRester.instances = []
    monkeypatch.setattr(mp, "_get_mprester_class", lambda: FakeRester)


def test_get_structure_by_id_returns_bare_atoms_copy(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "env-key")

    atoms = mp.get_structure_by_id("mp-149", api_key="explicit-key")

    assert isinstance(atoms, Atoms)
    assert atoms.get_chemical_formula() == "Si2"
    assert FakeRester.instances[-1].api_key == "explicit-key"
    assert FakeRester.instances[-1].last_material_id == "mp-149"
    assert FakeRester.instances[-1].last_structure_kwargs == {"final": True}


def test_get_structure_record_by_id_returns_provenance(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "env-key")

    record = mp.get_structure_record_by_id("mp-149")

    record.atoms.positions[0, 0] = 1.0

    assert record.material_id == "mp-149"
    assert record.formula == "Si"
    assert record.spacegroup_number == 227
    assert record.spacegroup_symbol == "Fd-3m"
    assert record.source == "materials_project"
    assert record.metadata["structure_type"] == "final"
    assert FakeRester.instances[-1].api_key == "env-key"
    assert FakeRester.instances[-1].search_calls[-1]["material_ids"] == ["mp-149"]
    assert FakeRester.instances[-1].last_structure_kwargs == {"final": True}


def test_structure_type_initial_and_invalid_values():
    record = mp.get_structure_record_by_id("mp-149", structure_type="initial")

    assert record.metadata["structure_type"] == "initial"
    assert FakeRester.instances[-1].last_structure_kwargs == {"final": False}

    with pytest.raises(ValueError, match="structure_type"):
        mp.get_structure_by_id("mp-149", structure_type="conventional")


def test_get_structures_by_spacegroup_maps_number_and_symbol():
    number_records = mp.get_structures_by_spacegroup(141, limit=5)
    number_call = FakeRester.instances[-1].search_calls[-1]

    symbol_records = mp.get_structures_by_spacegroup("I4_1/amd")
    symbol_call = FakeRester.instances[-1].search_calls[-1]

    assert number_records[0].material_id == "mp-149"
    assert number_call["spacegroup_number"] == 141
    assert number_call["num_chunks"] == 1
    assert number_call["chunk_size"] == 5
    assert symbol_records[0].spacegroup_number == 227
    assert symbol_call["spacegroup_symbol"] == "I4_1/amd"


def test_query_structures_forwards_filters_and_default_fields():
    records = mp.query_structures(
        elements=["Zr", "Si"], energy_above_hull=(0, 0.05), limit=10
    )
    call = FakeRester.instances[-1].search_calls[-1]

    assert records[0].formula == "Si"
    assert call["elements"] == ["Zr", "Si"]
    assert call["energy_above_hull"] == (0, 0.05)
    assert call["fields"] == ["material_id", "formula_pretty", "symmetry", "structure"]
    assert call["num_chunks"] == 1
    assert call["chunk_size"] == 10


def test_missing_optional_dependency_message(monkeypatch):
    def missing():
        raise RuntimeError(
            "Install with `uv sync --extra materials-project` or `pip install atomchain[materials-project]`."
        )

    monkeypatch.setattr(mp, "_get_mprester_class", missing)

    with pytest.raises(RuntimeError, match="materials-project"):
        mp.query_structures()


def test_real_optional_dependency_import_error_message(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mp_api.client":
            raise ImportError("no mp-api")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(mp, "_get_mprester_class", ORIGINAL_GET_MPRESTER_CLASS)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="uv sync --extra materials-project"):
        mp._get_mprester_class()


def test_write_structure_records_and_manifest(tmp_path):
    records = [
        MaterialsProjectStructure(
            material_id="mp-149",
            formula="Si",
            atoms=silicon_atoms(),
            spacegroup_number=227,
            spacegroup_symbol="Fd-3m",
        )
    ]

    written = mp.write_structure_records(records, tmp_path, fmt="vasp")
    manifest_path = mp.write_manifest(
        written,
        tmp_path / "manifest.yaml",
        {"command": "query", "filters": {"elements": ["Si"]}},
    )

    assert written[0].structure_file == "mp-149_Si.vasp"
    assert (tmp_path / "mp-149_Si.vasp").exists()
    assert read(tmp_path / "mp-149_Si.vasp").get_chemical_formula() == "Si2"

    manifest = manifest_path.read_text()
    assert "source: materials_project" in manifest
    assert "material_id: mp-149" in manifest
    assert "structure_file: mp-149_Si.vasp" in manifest

    with pytest.raises(FileExistsError):
        mp.write_structure_records(records, tmp_path, fmt="vasp")

    mp.write_structure_records(records, tmp_path, fmt="vasp", overwrite=True)


def test_write_empty_manifest(tmp_path):
    manifest_path = mp.write_manifest(
        [], tmp_path / "manifest.yaml", {"command": "spacegroup", "spacegroup": 141}
    )

    text = manifest_path.read_text()
    assert "results: []" in text


def test_parse_filter_values():
    assert mp.parse_filter_pair("is_stable=true") == ("is_stable", True)
    assert mp.parse_filter_pair("nelements=2") == ("nelements", 2)
    assert mp.parse_filter_pair("band_gap=0.1,1.5") == ("band_gap", (0.1, 1.5))
    assert mp.parse_filter_pair("formula=ZrSi") == ("formula", "ZrSi")

    with pytest.raises(ValueError, match="key=value"):
        mp.parse_filter_pair("bad")
