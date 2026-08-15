"""Tests for the minimal fit-and-reuse API (story-012)."""

import numpy as np
import pytest
import yaml
from ase import Atoms

from atomchain import bondvalence_workflow as bvw
from tests.test_bondvalence_fit import TRUE_R0, _fit_context


def _run_fit(tmp_path):
    context = _fit_context(tmp_path)
    metadata = bvw.StageMetadata(name="fit")
    metadata.outputs = bvw.run_fit_stage(context, metadata)
    context.manifest["stages"]["fit"] = metadata.to_dict()
    return context


def test_fitted_parameters_are_self_contained(tmp_path):
    _run_fit(tmp_path)
    data = yaml.safe_load(
        (tmp_path / "bv" / "fit" / "fitted_parameters.yaml").read_text("utf-8")
    )
    assert data["format"] == "atomchain-bv-model-v1"
    assert data["pairs"]["Na-Cl"]["r0"] == pytest.approx(TRUE_R0, abs=1e-3)
    assert data["contacts"]["cutoff"] == pytest.approx(3.6)
    assert data["oxidation_states"] == {"Na": 1.0, "Cl": -1.0}
    assert data["dataset_fingerprint"]


def test_load_and_evaluate_round_trip(tmp_path):
    context = _run_fit(tmp_path)
    model = bvw.load_bond_valence_model(
        context.output_dir / "fit" / "fitted_parameters.yaml"
    )
    assert model.contact_policy.cutoff == pytest.approx(3.6)
    assert model.oxidation_states == {"Na": 1.0, "Cl": -1.0}
    identifiers = [p.identifier for p in model.parameter_set.parameters]
    assert identifiers == ["Na-Cl"]

    dimer = Atoms("NaCl", positions=[[0, 0, 0], [TRUE_R0, 0, 0]])
    evaluation = bvw.evaluate_bond_valence(model, dimer)
    assert evaluation["gii"] == pytest.approx(0.0, abs=1e-6)
    assert len(evaluation["site_sums"]) == 2
    assert evaluation["per_site"][0]["element"] == "Na"
    assert evaluation["per_site"][0]["target"] == pytest.approx(1.0)
    assert np.asarray(evaluation["dipoles"]).shape == (2, 3)
    assert "units" in evaluation


def test_evaluate_reports_nonzero_mismatch(tmp_path):
    context = _run_fit(tmp_path)
    model = bvw.load_bond_valence_model(
        context.output_dir / "fit" / "fitted_parameters.yaml"
    )
    stretched = Atoms("NaCl", positions=[[0, 0, 0], [TRUE_R0 + 0.2, 0, 0]])
    evaluation = bvw.evaluate_bond_valence(model, stretched)
    assert evaluation["gii"] > 0.1
    assert all(site["mismatch"] < 0 for site in evaluation["per_site"])


def test_loader_rejects_incomplete_model(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("pairs: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bv-model"):
        bvw.load_bond_valence_model(path)
    path.write_text(
        yaml.safe_dump(
            {
                "format": "atomchain-bv-model-v1",
                "pairs": {
                    "Na-Cl": {
                        "selector": ["Na", "Cl"],
                        "r0": 2.3,
                        "b": 0.37,
                        "source": "t",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contacts"):
        bvw.load_bond_valence_model(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("contacts: {cutoff: 3.6}\n")
    with pytest.raises(ValueError, match="oxidation"):
        bvw.load_bond_valence_model(path)


def test_loader_rejects_malformed_pair_records(tmp_path):
    def write(payload):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    base = {
        "format": "atomchain-bv-model-v1",
        "contacts": {"cutoff": 3.6},
        "oxidation_states": {"Na": 1, "Cl": -1},
    }
    bad_selector = dict(
        base, pairs={"Na-Cl": {"selector": ["Na", "Cl", "X"], "r0": 2.3, "b": 0.37}}
    )
    with pytest.raises(ValueError, match="two-element"):
        bvw.load_bond_valence_model(write(bad_selector))
    bad_number = dict(
        base,
        pairs={
            "Na-Cl": {
                "selector": ["Na", "Cl"],
                "r0": "not-a-number",
                "b": 0.37,
            }
        },
    )
    with pytest.raises(ValueError, match="non-numeric"):
        bvw.load_bond_valence_model(write(bad_number))
    missing_field = dict(base, pairs={"Na-Cl": {"selector": ["Na", "Cl"], "b": 0.37}})
    with pytest.raises(ValueError, match="missing field 'r0'"):
        bvw.load_bond_valence_model(write(missing_field))


def test_evaluate_missing_species_is_actionable(tmp_path):
    from ase.build import bulk

    context = _run_fit(tmp_path)
    model = bvw.load_bond_valence_model(
        context.output_dir / "fit" / "fitted_parameters.yaml"
    )
    with pytest.raises(ValueError, match="F"):
        bvw.evaluate_bond_valence(model, bulk("NaF", "rocksalt", a=4.6, cubic=True))


def test_round_trip_multipoles_and_provenance(tmp_path):
    context = _run_fit(tmp_path)
    path = context.output_dir / "fit" / "fitted_parameters.yaml"
    data = yaml.safe_load(path.read_text("utf-8"))
    assert data["provenance"]
    model = bvw.load_bond_valence_model(path)
    dimer = Atoms("NaCl", positions=[[0, 0, 0], [2.3, 0, 0]])
    evaluation = bvw.evaluate_bond_valence(model, dimer)
    assert np.asarray(evaluation["quadrupoles"]).shape == (2, 3, 3)
    assert np.asarray(evaluation["invariants"]).shape == (2, 5)
    assert evaluation["max_rank"] == 4


def test_reused_fit_checkpoint_yields_loadable_model(tmp_path, monkeypatch):
    """Engine-level: a reused fit checkpoint must still satisfy the API."""
    from tests.test_bondvalence_fit import SEED
    from tests.test_bondvalence_stages import SpringCalculator

    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(SEED.lstrip(), encoding="utf-8")
    parent = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [2.3, 0, 0]],
        cell=np.eye(3) * 12.0,
        pbc=(False, False, False),
    )
    from ase.io import write as ase_write

    structure = tmp_path / "POSCAR"
    ase_write(structure, parent, format="vasp", vasp5=True, direct=True)
    config = bvw.BVWorkflowConfig(
        structure=structure,
        seed_file=seed_path,
        output_dir=tmp_path / "bv",
        relax=bvw.RelaxStageConfig(fmax=0.05),
        sampling=bvw.BVSamplingStageConfig(
            rattle_amplitudes=(0.05,),
            frames_per_amplitude=2,
            strain_range=(-0.02, 0.02),
            strain_points=3,
            energy_window=10.0,
        ),
        validation=bvw.BVValidationStageConfig(gii_threshold=2.0),
    )
    monkeypatch.setattr(
        bvw,
        "init_calc",
        lambda model_type, model_path=None: SpringCalculator(parent.positions, k=5.0),
    )
    bvw.run_bondvalence_workflow(config)
    second = bvw.run_bondvalence_workflow(config)
    assert second.manifest["stages"]["fit"]["status"] == "reused"
    model = bvw.load_bond_valence_model(second.fitted_parameters)
    evaluation = bvw.evaluate_bond_valence(model, parent)
    assert evaluation["gii"] < 0.2
