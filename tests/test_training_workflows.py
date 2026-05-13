import sys
import types

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from atomchain.multibinit import (
    train_multibinit_model,
    validate_trained_multibinit_model,
)
from atomchain.training.artifacts import generate_multibinit_training_artifacts
from atomchain.training.evaluate import evaluate_training_frames
from atomchain.training.samplers import (
    generate_training_trajectory,
    sample_md_frames,
    sample_metastable_frames,
    sample_metastable_linear_combinations,
    sample_phonon_mode_frames,
)
from atomchain.training.validation import (
    validate_fitted_model,
    validate_metastable_energy_differences,
)


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = -float(len(atoms))
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = np.arange(6, dtype=float) * 1e-3


class OffsetCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self, energy_offset=0.0, force_offset=0.0, stress_offset=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.energy_offset = float(energy_offset)
        self.force_offset = float(force_offset)
        self.stress_offset = float(stress_offset)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = -float(len(atoms)) + self.energy_offset
        self.results["forces"] = np.full((len(atoms), 3), self.force_offset)
        self.results["stress"] = np.arange(6, dtype=float) * 1e-3 + self.stress_offset


def test_evaluate_training_frames_attaches_results_and_provenance():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    evaluated = evaluate_training_frames(
        [atoms], calculator=ConstantCalculator(), provenance={"source": "unit"}
    )
    assert evaluated[0].info["source"] == "unit"
    assert evaluated[0].get_potential_energy() == -len(atoms)
    assert evaluated[0].get_forces().shape == (len(atoms), 3)
    assert evaluated[0].get_stress(voigt=True).shape == (6,)


def test_validate_fitted_model_compares_energy_forces_and_stress(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    reference = evaluate_training_frames([atoms], calculator=ConstantCalculator())
    result = validate_fitted_model(
        reference,
        calculator=OffsetCalculator(
            energy_offset=0.2, force_offset=0.03, stress_offset=0.004
        ),
        output=tmp_path / "validation.json",
    )

    assert result.nframes == 1
    assert result.metrics["energy_ev"].mae == pytest.approx(0.2)
    assert result.metrics["forces_ev_ang"].mae == pytest.approx(0.03)
    assert result.metrics["stress_ev_ang3"].mae == pytest.approx(0.004)
    assert (tmp_path / "validation.json").exists()


def test_validate_fitted_model_loads_hist_test_set(tmp_path):
    from atomchain.io.hist import write_abinit_hist

    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    reference = evaluate_training_frames([atoms], calculator=ConstantCalculator())
    hist = tmp_path / "test_HIST.nc"
    write_abinit_hist(reference, hist)

    result = validate_fitted_model(hist, calculator=OffsetCalculator(energy_offset=0.1))

    assert result.nframes == 1
    assert result.metrics["energy_ev"].mae == pytest.approx(0.1)


def test_validate_metastable_energy_differences_compares_relative_energies():
    parent = bulk("Al", "fcc", a=4.05, cubic=True)
    metastable = parent.copy()
    metastable.positions += 0.01
    parent.calc = OffsetCalculator(energy_offset=0.0)
    metastable.calc = OffsetCalculator(energy_offset=0.5)

    result = validate_metastable_energy_differences(
        parent,
        [metastable],
        calculator=OffsetCalculator(energy_offset=0.25),
    )

    assert result.nstates == 1
    assert result.states[0]["reference_delta_ev"] == pytest.approx(0.5)
    assert result.states[0]["model_delta_ev"] == pytest.approx(0.0)
    assert result.metric.mae == pytest.approx(0.5)


def test_md_sampler_records_provenance():
    frames = sample_md_frames(bulk("Al", "fcc", a=4.05), steps=2, seed=1)
    assert len(frames) == 3
    assert all(f.info["source"] == "md" for f in frames)


def test_phonon_sampler_records_mode_metadata():
    frames = sample_phonon_mode_frames(
        bulk("Al", "fcc", a=4.05), amplitudes=[0.1], directions=[0]
    )
    assert len(frames) == 1
    assert frames[0].info["source"] == "phonon_modes"
    assert frames[0].info["amplitude"] == 0.1


def test_metastable_sampler_and_combinations():
    parent = bulk("Al", "fcc", a=4.05, cubic=True)
    meta = parent.copy()
    meta.positions += 0.01
    results = {"results": [{"id": 1, "combined_label": "M1", "atoms": meta}]}
    frames = sample_metastable_frames(results)
    combos = sample_metastable_linear_combinations(parent, results, weights=[0.5])
    assert frames[0].info["source"] == "metastable"
    assert combos[0].info["source"] == "metastable_linear_combination"
    np.testing.assert_allclose(combos[0].positions, parent.positions + 0.005)


def test_generate_training_trajectory_without_evaluation(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    output = tmp_path / "training.traj"
    frames = generate_training_trajectory(
        atoms, sources=["phonon_modes"], evaluate=False, output=output
    )
    assert frames
    assert output.exists()


def test_artifact_bundle_with_existing_evaluated_trajectory(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    frames = evaluate_training_frames([atoms], calculator=ConstantCalculator())
    traj = tmp_path / "frames.traj"
    write(str(traj), frames)
    data = generate_multibinit_training_artifacts(
        atoms,
        traj,
        calculator=ConstantCalculator(),
        ddb="out.ddb",
        hist="out_HIST.nc",
        output_dir=tmp_path,
    )
    assert (tmp_path / "out_HIST.nc").exists()
    assert data["hist"].endswith("out_HIST.nc")
    assert data["ddb"] is None
    assert data["ddb_written"] is False


def test_artifact_bundle_accepts_single_atoms_input(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    frames = evaluate_training_frames([atoms], calculator=ConstantCalculator())
    data = generate_multibinit_training_artifacts(
        atoms,
        frames[0],
        ddb="out.ddb",
        hist="single_HIST.nc",
        output_dir=tmp_path,
    )
    assert (tmp_path / "single_HIST.nc").exists()
    assert data["hist"].endswith("single_HIST.nc")


def test_pymultibinit_training_wrapper_delegates(monkeypatch, tmp_path):
    module = types.ModuleType("pymultibinit")
    training = types.ModuleType("pymultibinit.training")

    def fake_train_multibinit_model(**kwargs):
        config = tmp_path / "model.conf"
        config.write_text("model", encoding="utf-8")
        return {
            "model_config": str(config),
            "output_dir": kwargs["output_dir"],
            "artifacts": {"config": str(config)},
        }

    training.train_multibinit_model = fake_train_multibinit_model
    monkeypatch.setitem(sys.modules, "pymultibinit", module)
    monkeypatch.setitem(sys.modules, "pymultibinit.training", training)
    result = train_multibinit_model("a.ddb", "a_HIST.nc", output_dir=tmp_path)
    assert result.model_config.endswith("model.conf")
    assert validate_trained_multibinit_model(result.model_config, result.artifacts)


def test_pymultibinit_training_wrapper_accepts_result_object(monkeypatch, tmp_path):
    module = types.ModuleType("pymultibinit")
    training = types.ModuleType("pymultibinit.training")

    class FakeResult:
        def to_dict(self):
            config = tmp_path / "model.conf"
            config.write_text("model", encoding="utf-8")
            return {
                "model_config": str(config),
                "output_dir": str(tmp_path),
                "artifacts": {"config": str(config)},
            }

    training.train_multibinit_model = lambda **kwargs: FakeResult()
    monkeypatch.setitem(sys.modules, "pymultibinit", module)
    monkeypatch.setitem(sys.modules, "pymultibinit.training", training)
    result = train_multibinit_model("a.ddb", "a_HIST.nc", output_dir=tmp_path)
    assert result.model_config.endswith("model.conf")


def test_trained_model_validation_resolves_relative_artifacts(tmp_path):
    (tmp_path / "model.conf").write_text("model", encoding="utf-8")
    assert validate_trained_multibinit_model(
        "model.conf", {"config": "model.conf"}, output_dir=tmp_path
    )


def test_pymultibinit_training_wrapper_missing_backend(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pymultibinit", None)
    monkeypatch.setitem(sys.modules, "pymultibinit.training", None)
    with pytest.raises(ImportError, match="pymultibinit training API"):
        train_multibinit_model("a.ddb", "a_HIST.nc", output_dir=tmp_path)
