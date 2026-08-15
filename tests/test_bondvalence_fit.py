"""Tests for the bond-valence workflow fit stage (story-006)."""

import json
import textwrap

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import Trajectory

from atomchain import bondvalence_workflow as bvw

# True generating parameters for the analytic fixture.
TRUE_R0 = 2.30
TRUE_B = 0.37


def _fixture_frames():
    """Structures with exactly zero residual at the true (r0, b).

    In every structure each site's bond-valence sum equals |z| = 1 v.u.:
    a dimer (coordination 1, d = r0), a periodic 1D chain (coordination 2,
    d = r0 + b ln 2), and a periodic 2D square layer (coordination 4,
    d = r0 + b ln 4). The three distinct coordination distances make
    (r0, b) identifiable.
    """
    d1 = TRUE_R0
    d2 = TRUE_R0 + TRUE_B * np.log(2.0)
    d4 = TRUE_R0 + TRUE_B * np.log(4.0)
    vacuum = 12.0
    dimer = Atoms("NaCl", positions=[[0, 0, 0], [d1, 0, 0]], pbc=(False,) * 3)
    dimer.info = {"frame_id": "dimer", "group": "dimer", "kind": "dimer"}
    chain = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [d2, 0, 0]],
        cell=[[2 * d2, 0, 0], [0, vacuum, 0], [0, 0, vacuum]],
        pbc=(True, False, False),
    )
    chain.info = {"frame_id": "chain", "group": "chain", "kind": "chain"}
    # Square lattice with constant d4*sqrt(2) and Cl at the cell center: the
    # Na-Cl distance is d4 while same-species neighbors sit d4*sqrt(2) apart,
    # beyond the cutoff.
    lattice = d4 * np.sqrt(2.0)
    layer = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [lattice / 2, lattice / 2, 0]],
        cell=[[lattice, 0, 0], [0, lattice, 0], [0, 0, vacuum]],
        pbc=(True, True, False),
    )
    layer.info = {"frame_id": "layer", "group": "layer", "kind": "layer"}
    return [dimer, chain, layer]


SEED = """
provenance: "analytic fixture table"
oxidation_states: {Na: 1, Cl: -1}
contacts: {cutoff: 3.6}
pairs:
  - selector: [Na, Cl]
    r0: {initial: 2.35, lower: 2.0, upper: 2.6}
    b: {initial: 0.30, lower: 0.2, upper: 0.6, transform: positive}
"""


def _fit_context(tmp_path):
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(textwrap.dedent(SEED), encoding="utf-8")
    output = tmp_path / "bv"
    sampling_dir = output / "sampling"
    sampling_dir.mkdir(parents=True)
    evaluated_path = sampling_dir / "evaluated.traj"
    with Trajectory(evaluated_path, "w") as trajectory:
        for atoms in _fixture_frames():
            trajectory.write(atoms)
    config = bvw.BVWorkflowConfig(
        structure=Atoms("NaCl2", positions=[[0, 0, 0], [2.3, 0, 0], [0, 2.3, 0]]),
        seed_file=seed_path,
        output_dir=output,
    )
    context = bvw.BVWorkflowContext(config, config.structure)
    evaluate_metadata = bvw.StageMetadata(name="evaluate")
    evaluate_metadata.outputs = {
        "evaluated": "sampling/evaluated.traj",
        "filter": "sampling/filter.json",
    }
    context.manifest["stages"]["evaluate"] = evaluate_metadata.to_dict()
    return context


def test_fit_stage_recovers_analytic_parameters(tmp_path):
    context = _fit_context(tmp_path)
    metadata = bvw.StageMetadata(name="fit")
    outputs = bvw.run_fit_stage(context, metadata)
    fitted = yaml.safe_load(
        (tmp_path / "bv" / "fit" / "fitted_parameters.yaml").read_text("utf-8")
    )
    pair = fitted["pairs"]["Na-Cl"]
    assert pair["r0"] == pytest.approx(TRUE_R0, abs=1e-3)
    assert pair["b"] == pytest.approx(TRUE_B, abs=1e-3)
    assert pair["source"] == "analytic fixture table"
    diagnostics = json.loads(
        (tmp_path / "bv" / "fit" / "diagnostics.json").read_text("utf-8")
    )
    assert diagnostics["converged"] is True
    assert diagnostics["termination"] in {"ftol", "xtol", "gtol"}
    assert diagnostics["sensitivity"]["rank"] >= 1
    split = json.loads((tmp_path / "bv" / "fit" / "split.json").read_text("utf-8"))
    training_groups = set(split["training_group_ids"])
    validation_groups = set(split["validation_group_ids"])
    assert training_groups and validation_groups
    assert not training_groups & validation_groups
    assert outputs["fitted_parameters"] == "fit/fitted_parameters.yaml"
    assert outputs["diagnostics"] == "fit/diagnostics.json"
    assert outputs["split"] == "fit/split.json"


def test_fit_stage_records_dataset_provenance(tmp_path):
    context = _fit_context(tmp_path)
    bvw.run_fit_stage(context, bvw.StageMetadata(name="fit"))
    diagnostics = json.loads(
        (tmp_path / "bv" / "fit" / "diagnostics.json").read_text("utf-8")
    )
    assert "analytic fixture table" in diagnostics["dataset_provenance"]
    assert "mace" in diagnostics["dataset_provenance"]


def test_fit_stage_warns_when_training_lacks_strain(tmp_path):
    context = _fit_context(tmp_path)
    metadata = bvw.StageMetadata(name="fit")
    bvw.run_fit_stage(context, metadata)
    assert any("strain" in warning for warning in metadata.warnings)


def test_fit_stage_requires_evaluate_outputs(tmp_path):
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(textwrap.dedent(SEED), encoding="utf-8")
    config = bvw.BVWorkflowConfig(
        structure=Atoms("NaCl2", positions=[[0, 0, 0], [2.3, 0, 0], [0, 2.3, 0]]),
        seed_file=seed_path,
        output_dir=tmp_path / "bv2",
    )
    context = bvw.BVWorkflowContext(config, config.structure)
    with pytest.raises(RuntimeError, match="evaluate"):
        bvw.run_fit_stage(context, bvw.StageMetadata(name="fit"))


def test_fit_stage_survives_rank_deficiency(tmp_path):
    """Two identical dimers: r0 and b are perfectly correlated (rank 1)."""
    from ase.io import Trajectory

    context = _fit_context(tmp_path)
    # Replace the evaluated frames with two identical dimer structures.
    dimer = Atoms("NaCl", positions=[[0, 0, 0], [TRUE_R0, 0, 0]])
    evaluated = tmp_path / "bv" / "sampling" / "evaluated.traj"
    with Trajectory(evaluated, "w") as trajectory:
        for name in ("a", "b"):
            atoms = dimer.copy()
            atoms.info = {
                "frame_id": f"dimer-{name}",
                "group": f"dimer-{name}",
                "kind": "dimer",
            }
            trajectory.write(atoms)
    # Seed initial values at the truth so the residual is zero and the fit
    # terminates immediately despite the degenerate Jacobian.
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(
        textwrap.dedent(SEED)
        .replace("initial: 2.35", "initial: 2.30")
        .replace("initial: 0.30", "initial: 0.37"),
        encoding="utf-8",
    )
    metadata = bvw.StageMetadata(name="fit")
    bvw.run_fit_stage(context, metadata)  # must not raise
    diagnostics = json.loads(
        (tmp_path / "bv" / "fit" / "diagnostics.json").read_text("utf-8")
    )
    assert diagnostics["sensitivity"]["rank"] < 2
    # Infinite condition numbers are serialized as None (JSON-safe), and the
    # stage still wrote every artifact.
    assert diagnostics["sensitivity"]["condition_number"] is None
    assert (tmp_path / "bv" / "fit" / "fitted_parameters.yaml").exists()
    assert (tmp_path / "bv" / "fit" / "split.json").exists()


def test_fit_stage_provenance_records_sampling(tmp_path):
    context = _fit_context(tmp_path)
    context.config.sampling.rattle_amplitudes = (0.04, 0.08)
    bvw.run_fit_stage(context, bvw.StageMetadata(name="fit"))
    diagnostics = json.loads(
        (tmp_path / "bv" / "fit" / "diagnostics.json").read_text("utf-8")
    )
    provenance = diagnostics["dataset_provenance"]
    assert "rattle=[0.04, 0.08]" in provenance
    assert "energy_window=" in provenance
