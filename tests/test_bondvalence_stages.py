"""Tests for the bond-valence workflow data-generation stages (stories 003-005)."""

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

from atomchain import bondvalence_workflow as bvw

SEED_NACL = """
provenance: "test tables"
oxidation_states: {Na: 1, Cl: -1}
contacts: {cutoff: 3.6}
pairs:
  - selector: [Na, Cl]
    r0: {initial: 2.30, lower: 1.0, upper: 3.0}
    b: {initial: 0.37, lower: 0.1, upper: 1.0, transform: positive}
"""


class SpringCalculator(Calculator):
    """Restoring harmonic forces toward reference positions (no real physics)."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, reference, k=5.0):
        super().__init__()
        self.reference = np.asarray(reference, dtype=float)
        self.k = float(k)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = atoms.positions - self.reference
        energy = 0.5 * self.k * np.sum(displacement**2)
        forces = -self.k * displacement
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = np.zeros(6)


def _seed_path(tmp_path):
    path = tmp_path / "seed.yaml"
    path.write_text(textwrap.dedent(SEED_NACL), encoding="utf-8")
    return path


def _context(tmp_path, **overrides):
    parent = bulk("NaCl", "rocksalt", a=5.6)
    config = bvw.BVWorkflowConfig(
        structure=parent,
        seed_file=_seed_path(tmp_path),
        output_dir=tmp_path / "bv",
        **overrides,
    )
    context = bvw.BVWorkflowContext(config, parent)
    context.reference_calculator = SpringCalculator(parent.positions)
    return context


def test_relax_stage_writes_artifacts_and_updates_context(tmp_path):
    context = _context(tmp_path)
    atoms = context.parent_atoms.copy()
    rng = np.random.default_rng(7)
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    context.parent_atoms = atoms
    metadata = bvw.StageMetadata(name="relax")
    outputs = bvw.run_relax_stage(context, metadata)
    relaxed = context.relaxed_atoms
    relaxed.calc = context.reference_calculator
    assert np.linalg.norm(relaxed.get_forces(), axis=1).max() < 0.05
    assert (tmp_path / "bv" / "relax" / "relaxed.vasp").exists()
    data = json.loads(
        (tmp_path / "bv" / "relax" / "relaxation.json").read_text("utf-8")
    )
    assert "energy_per_atom" in data
    assert "final_fmax" in data
    assert "calculator" in data
    assert outputs["relaxed"] == "relax/relaxed.vasp"


def test_relax_stage_runs_contact_audit(tmp_path):
    # Cutoff 0.5 A is far too small: Na-Cl has zero contacts -> audit warning.
    context = _context(tmp_path)
    seed = Path(context.config.seed_file)
    seed.write_text(
        seed.read_text("utf-8").replace("cutoff: 3.6", "cutoff: 0.5"), "utf-8"
    )
    metadata = bvw.StageMetadata(name="relax")
    bvw.run_relax_stage(context, metadata)
    assert any("Na-Cl" in warning for warning in metadata.warnings)


def test_reference_calculator_initialized_once(tmp_path, monkeypatch):
    from ase.calculators.emt import EMT

    calls = []

    def fake_init_calc(model_type, model_path=None):
        calls.append(model_type)
        return EMT()

    monkeypatch.setattr(bvw, "init_calc", fake_init_calc)
    context = _context(tmp_path)
    context.reference_calculator = None
    first = bvw._reference_calculator(context)
    second = bvw._reference_calculator(context)
    assert calls == ["mace"]
    assert first is second


# ---------------------------------------------------------------------------
# Sampling stage (story-004)


def _sampled_context(tmp_path, **sampling_overrides):
    context = _context(tmp_path)
    context.relaxed_atoms = context.parent_atoms.copy()
    sampling = context.config.sampling
    for key, value in sampling_overrides.items():
        setattr(sampling, key, value)
    return context


def _read_manifest(result_dir):
    return yaml.safe_load(
        (result_dir / "sampling" / "frame_manifest.yaml").read_text("utf-8")
    )


def test_sampling_counts_groups_and_manifest_fields(tmp_path):
    context = _sampled_context(
        tmp_path,
        rattle_amplitudes=(0.03, 0.06),
        frames_per_amplitude=3,
        strain_range=(-0.02, 0.02),
        strain_points=5,
    )
    metadata = bvw.StageMetadata(name="sample")
    outputs = bvw.run_sample_stage(context, metadata)
    frames = _read_manifest(tmp_path / "bv")["frames"]
    assert len(frames) == 1 + 2 * 3 + 5
    groups = {frame["group"] for frame in frames}
    assert groups == {"relaxed", "rattle-a0.03", "rattle-a0.06", "strain"}
    by_group = {}
    for frame in frames:
        by_group.setdefault(frame["group"], []).append(frame)
    assert len(by_group["rattle-a0.03"]) == 3
    assert len(by_group["strain"]) == 5
    for frame in frames:
        assert frame["frame_id"]
        if frame["kind"] == "rattle":
            assert frame["amplitude"] in (0.03, 0.06)
        if frame["kind"] == "strain":
            assert -0.02 <= frame["strain"] <= 0.02
    assert outputs["frames"] == "sampling/frames.traj"
    assert (tmp_path / "bv" / "sampling" / "frames.traj").exists()


def test_strain_frames_scale_cell_preserve_fractional_coords(tmp_path):
    context = _sampled_context(
        tmp_path, rattle_amplitudes=(), frames_per_amplitude=0, strain_points=3
    )
    bvw.run_sample_stage(context, bvw.StageMetadata(name="sample"))
    frames = read(tmp_path / "bv" / "sampling" / "frames.traj", ":")
    strain_frames = [
        frame
        for frame in frames
        if frame.info.get("kind") == "strain"
        and abs(frame.info.get("strain", 1.0)) > 1e-12
    ]
    assert strain_frames
    parent = context.relaxed_atoms
    for frame in strain_frames:
        eps = frame.info["strain"]
        assert np.allclose(frame.cell, (1.0 + eps) * np.asarray(parent.cell))
        assert np.allclose(
            frame.positions @ np.linalg.inv(np.asarray(frame.cell)),
            parent.positions @ np.linalg.inv(np.asarray(parent.cell)),
            atol=1e-8,
        )


def test_sampling_deterministic_under_seed(tmp_path):
    outputs = []
    for out_name, seed in (("run-a", 1234), ("run-b", 1234), ("run-c", 99)):
        (tmp_path / out_name).mkdir()
        context = _sampled_context(tmp_path / out_name)
        context.config.output_dir = tmp_path / out_name
        context.output_dir = tmp_path / out_name
        context.config.seed = seed
        bvw.run_sample_stage(context, bvw.StageMetadata(name="sample"))
        outputs.append(read(tmp_path / out_name / "sampling" / "frames.traj", ":"))
    same_seed_a, same_seed_b, other_seed = outputs
    assert len(same_seed_a) == len(same_seed_b) == len(other_seed)
    for frame_a, frame_b in zip(same_seed_a, same_seed_b):
        assert np.allclose(frame_a.positions, frame_b.positions)
    rattled = [
        (frame_a.positions, frame_c.positions)
        for frame_a, frame_c in zip(same_seed_a, other_seed)
        if frame_a.info.get("kind") == "rattle"
    ]
    assert rattled
    assert any(
        not np.allclose(positions_a, positions_c)
        for positions_a, positions_c in rattled
    )


def test_empty_amplitudes_yields_relaxed_and_strain_only(tmp_path):
    context = _sampled_context(
        tmp_path, rattle_amplitudes=(), frames_per_amplitude=5, strain_points=3
    )
    bvw.run_sample_stage(context, bvw.StageMetadata(name="sample"))
    frames = _read_manifest(tmp_path / "bv")["frames"]
    assert {frame["group"] for frame in frames} == {"relaxed", "strain"}


# ---------------------------------------------------------------------------
# Evaluation stage (story-005)


class EnergyMapCalculator(Calculator):
    """Deterministic per-frame energies keyed by frame_id."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, mapping):
        super().__init__()
        self.mapping = dict(mapping)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = float(self.mapping[atoms.info["frame_id"]])
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = np.zeros(6)


def _evaluated_context(tmp_path, energies, **sampling_overrides):
    context = _sampled_context(
        tmp_path,
        rattle_amplitudes=(0.03,),
        frames_per_amplitude=2,
        strain_points=2,
        **sampling_overrides,
    )
    sample_metadata = bvw.StageMetadata(name="sample")
    sample_metadata.outputs = bvw.run_sample_stage(context, sample_metadata)
    context.manifest["stages"]["sample"] = sample_metadata.to_dict()
    context.reference_calculator = EnergyMapCalculator(energies)
    return context


def test_evaluate_stage_filters_by_energy_window(tmp_path):
    natoms = len(bulk("NaCl", "rocksalt", a=5.6))
    context = _evaluated_context(
        tmp_path,
        energies={
            "relaxed": 0.0,
            "rattle-0-0": natoms * 0.02,  # delta 0.02 -> kept
            "rattle-0-1": natoms * 0.10,  # delta 0.10 -> dropped (window 0.05)
            "strain-0": natoms * 0.05,  # delta exactly at window -> kept
            "strain-1": natoms * 0.30,  # dropped
        },
    )
    metadata = bvw.StageMetadata(name="evaluate")
    outputs = bvw.run_evaluate_stage(context, metadata)
    filter_data = json.loads(
        (tmp_path / "bv" / "sampling" / "filter.json").read_text("utf-8")
    )
    by_id = {item["frame_id"]: item for item in filter_data["frames"]}
    assert by_id["relaxed"]["kept"] is True
    assert by_id["rattle-0-0"]["kept"] is True
    assert by_id["rattle-0-1"]["kept"] is False
    assert by_id["strain-0"]["kept"] is True  # inclusive boundary
    assert by_id["strain-1"]["kept"] is False
    assert by_id["rattle-0-1"]["delta_vs_parent"] == pytest.approx(0.10)
    assert filter_data["group_stats"]["rattle-a0.03"] == {"kept": 1, "dropped": 1}
    kept = read(tmp_path / "bv" / "sampling" / "evaluated.traj", ":")
    assert {atoms.info["frame_id"] for atoms in kept} == {
        "relaxed",
        "rattle-0-0",
        "strain-0",
    }
    assert outputs["evaluated"] == "sampling/evaluated.traj"
    assert outputs["filter"] == "sampling/filter.json"


def test_evaluate_stage_warns_on_fully_dropped_group(tmp_path):
    natoms = len(bulk("NaCl", "rocksalt", a=5.6))
    context = _evaluated_context(
        tmp_path,
        energies={
            "relaxed": 0.0,
            "rattle-0-0": natoms * 0.01,
            "rattle-0-1": natoms * 0.02,
            "strain-0": natoms * 0.50,
            "strain-1": natoms * 0.60,
        },
    )
    metadata = bvw.StageMetadata(name="evaluate")
    bvw.run_evaluate_stage(context, metadata)
    assert any("strain" in warning for warning in metadata.warnings)


def test_evaluate_stage_requires_sample_outputs(tmp_path):
    context = _context(tmp_path)
    metadata = bvw.StageMetadata(name="evaluate")
    with pytest.raises(RuntimeError, match="sample"):
        bvw.run_evaluate_stage(context, metadata)


class NeverConvergingCalculator(Calculator):
    """Forces of fixed magnitude 0.1 eV/A that can never satisfy fmax."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        angle = 0.05 * self.calls
        forces = np.zeros((len(atoms), 3))
        forces[:, 0] = 0.1 * np.cos(angle)
        forces[:, 1] = 0.1 * np.sin(angle)
        self.results["energy"] = 0.0
        self.results["forces"] = forces
        self.results["stress"] = np.zeros(6)


def test_relax_stage_fails_when_not_converged(tmp_path):
    context = _context(tmp_path)
    context.reference_calculator = NeverConvergingCalculator()
    with pytest.raises(RuntimeError, match="did not converge"):
        bvw.run_relax_stage(context, bvw.StageMetadata(name="relax"))


def test_relax_records_effective_calculator_provenance(tmp_path):
    context = _context(tmp_path)
    bvw.run_relax_stage(context, bvw.StageMetadata(name="relax"))
    data = json.loads(
        (tmp_path / "bv" / "relax" / "relaxation.json").read_text("utf-8")
    )
    assert data["calculator"] == {
        "calculator": "SpringCalculator",
        "source": "user-provided instance",
    }


def test_atoms_only_relax_preserves_cell(tmp_path):
    context = _context(tmp_path)
    original_cell = np.array(context.parent_atoms.cell)
    atoms = context.parent_atoms.copy()
    atoms.positions += np.random.default_rng(3).normal(
        scale=0.05, size=atoms.positions.shape
    )
    context.parent_atoms = atoms
    bvw.run_relax_stage(context, bvw.StageMetadata(name="relax"))
    assert np.allclose(np.asarray(context.relaxed_atoms.cell), original_cell)


def test_relax_cell_branch_completes(tmp_path):
    context = _context(tmp_path, relax=bvw.RelaxStageConfig(fmax=0.05, relax_cell=True))
    bvw.run_relax_stage(context, bvw.StageMetadata(name="relax"))
    assert context.relaxed_atoms is not None
