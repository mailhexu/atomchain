"""Tests for the bond-valence workflow validation stage (story-007)."""

import json
import math
import textwrap

import numpy as np
import pytest
import yaml
from ase import Atoms

from atomchain import bondvalence_workflow as bvw
from tests.test_bondvalence_fit import SEED, TRUE_B, TRUE_R0, _fixture_frames


def _workflow_context(tmp_path):
    """Context whose fit stage has run on the coordination fixture."""
    output = tmp_path / "bv"
    config = bvw.BVWorkflowConfig(
        structure=Atoms(
            "NaCl",
            positions=[[0, 0, 0], [TRUE_R0, 0, 0]],
            cell=np.eye(3) * 12.0,
            pbc=(False, False, False),
        ),
        seed_file=tmp_path / "seed.yaml",
        output_dir=output,
    )
    (tmp_path / "seed.yaml").write_text(textwrap.dedent(SEED), encoding="utf-8")
    context = bvw.BVWorkflowContext(config, config.structure)
    from tests.test_bondvalence_stages import SpringCalculator

    context.reference_calculator = SpringCalculator(context.parent_atoms.positions)
    relax_metadata = bvw.StageMetadata(name="relax")
    relax_metadata.outputs = bvw.run_relax_stage(context, relax_metadata)
    context.manifest["stages"]["relax"] = relax_metadata.to_dict()
    # Sampling bookkeeping is minimal: fit needs evaluate outputs only.
    from ase.io import Trajectory

    sampling_dir = output / "sampling"
    sampling_dir.mkdir(parents=True)
    with Trajectory(sampling_dir / "evaluated.traj", "w") as trajectory:
        for atoms in _fixture_frames():
            trajectory.write(atoms)
    evaluate_metadata = bvw.StageMetadata(name="evaluate")
    evaluate_metadata.outputs = {"evaluated": "sampling/evaluated.traj"}
    context.manifest["stages"]["evaluate"] = evaluate_metadata.to_dict()
    fit_metadata = bvw.StageMetadata(name="fit")
    fit_metadata.outputs = bvw.run_fit_stage(context, fit_metadata)
    context.manifest["stages"]["fit"] = fit_metadata.to_dict()
    return context


def _metrics(tmp_path):
    return json.loads(
        (tmp_path / "bv" / "validation" / "metrics.json").read_text("utf-8")
    )


def test_validate_passes_with_zero_residual_fit(tmp_path):
    context = _workflow_context(tmp_path)
    metadata = bvw.StageMetadata(name="validate")
    outputs = bvw.run_validate_stage(context, metadata)
    metrics = _metrics(tmp_path)
    assert metrics["passed"] is True
    assert metrics["aggregate_rms_dv"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["threshold"] == pytest.approx(
        context.config.validation.gii_threshold
    )
    assert metrics["n_validation_structures"] >= 1
    assert outputs["metrics"] == "validation/metrics.json"
    assert (tmp_path / "bv" / "validation" / "dv_distribution.png").exists()
    assert (tmp_path / "bv" / "validation" / "dv_vs_distance.png").exists()


def test_validate_gate_arithmetic_is_hand_computed(tmp_path):
    context = _workflow_context(tmp_path)
    # Perturb the fitted r0 so every site mismatch is |exp((r0'-d)/b) - 1|.
    fitted_path = tmp_path / "bv" / "fit" / "fitted_parameters.yaml"
    fitted = yaml.safe_load(fitted_path.read_text("utf-8"))
    r0_perturbed = 2.40
    fitted["pairs"]["Na-Cl"]["r0"] = r0_perturbed
    fitted_path.write_text(yaml.safe_dump(fitted), encoding="utf-8")
    split = json.loads((tmp_path / "bv" / "fit" / "split.json").read_text("utf-8"))
    validation_groups = set(split["validation_group_ids"])

    from ase.io import read

    frames = read(tmp_path / "bv" / "sampling" / "evaluated.traj", ":")
    expected_dv = []
    expected_per_frame = {}
    for atoms in frames:
        if atoms.info["group"] not in validation_groups:
            continue
        kind = atoms.info["kind"]
        distance = {
            "dimer": TRUE_R0,
            "chain": TRUE_R0 + TRUE_B * np.log(2.0),
            "layer": TRUE_R0 + TRUE_B * np.log(4.0),
        }[kind]
        coordination = {"dimer": 1, "chain": 2, "layer": 4}[kind]
        mismatch = coordination * math.exp((r0_perturbed - distance) / TRUE_B) - 1.0
        expected_dv.extend([mismatch] * len(atoms))
        expected_per_frame[atoms.info["frame_id"]] = abs(mismatch)
    expected_rms = float(np.sqrt(np.mean(np.square(expected_dv))))

    metadata = bvw.StageMetadata(name="validate")
    bvw.run_validate_stage(context, metadata)
    metrics = _metrics(tmp_path)
    assert metrics["aggregate_rms_dv"] == pytest.approx(expected_rms, rel=1e-6)
    assert metrics["passed"] is (expected_rms < metrics["threshold"])
    for entry in metrics["per_structure"]:
        assert entry["gii"] == pytest.approx(
            expected_per_frame[entry["frame_id"]], rel=1e-6
        )


def test_validate_reports_identifiability_without_gating(tmp_path):
    context = _workflow_context(tmp_path)
    metadata = bvw.StageMetadata(name="validate")
    bvw.run_validate_stage(context, metadata)
    metrics = _metrics(tmp_path)
    identifiability = metrics["identifiability"]
    assert "rank" in identifiability
    assert "covariance_available" in identifiability
    assert "active_bounds" in identifiability
    # Identifiability fields are informational only.
    assert metrics["passed"] is True


def test_validate_requires_fit_outputs(tmp_path):
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(textwrap.dedent(SEED), encoding="utf-8")
    config = bvw.BVWorkflowConfig(
        structure=Atoms(
            "NaCl",
            positions=[[0, 0, 0], [TRUE_R0, 0, 0]],
            cell=np.eye(3) * 12.0,
            pbc=(False, False, False),
        ),
        seed_file=seed_path,
        output_dir=tmp_path / "bv2",
    )
    context = bvw.BVWorkflowContext(config, config.structure)
    with pytest.raises(RuntimeError, match="fit"):
        bvw.run_validate_stage(context, bvw.StageMetadata(name="validate"))


def test_validate_disabled_skips_metrics(tmp_path):
    context = _workflow_context(tmp_path)
    context.config.validation.enabled = False
    outputs = bvw.run_validate_stage(context, bvw.StageMetadata(name="validate"))
    assert outputs["skipped"] is True
    assert not (tmp_path / "bv" / "validation" / "metrics.json").exists()


def test_metrics_include_held_out_data_cost_and_identifiability_warnings(tmp_path):
    context = _workflow_context(tmp_path)
    metadata = bvw.StageMetadata(name="validate")
    bvw.run_validate_stage(context, metadata)
    metrics = _metrics(tmp_path)
    assert "held_out_data_cost" in metrics
    assert metrics["held_out_data_cost"] == pytest.approx(0.0, abs=1e-6)
    assert isinstance(metrics["identifiability"].get("warnings"), list)


def test_rank_deficient_fit_emits_warning_without_gating(tmp_path):
    from ase.io import Trajectory

    context = _workflow_context(tmp_path)
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
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(
        textwrap.dedent(SEED)
        .replace("initial: 2.35", "initial: 2.30")
        .replace("initial: 0.30", "initial: 0.37"),
        encoding="utf-8",
    )
    fit_metadata = bvw.StageMetadata(name="fit")
    fit_metadata.outputs = bvw.run_fit_stage(context, fit_metadata)
    context.manifest["stages"]["fit"] = fit_metadata.to_dict()
    metadata = bvw.StageMetadata(name="validate")
    bvw.run_validate_stage(context, metadata)
    warnings_text = " ".join(metadata.warnings)
    assert "rank deficient" in warnings_text or "covariance" in warnings_text
    # Diagnostics never gate.
    assert _metrics(tmp_path)["passed"] in (True, False)


def test_disabled_validation_removes_stale_artifacts(tmp_path):
    context = _workflow_context(tmp_path)
    metadata = bvw.StageMetadata(name="validate")
    bvw.run_validate_stage(context, metadata)
    assert (tmp_path / "bv" / "validation" / "metrics.json").exists()
    context.config.validation.enabled = False
    skipped_metadata = bvw.StageMetadata(name="validate")
    outputs = bvw.run_validate_stage(context, skipped_metadata)
    assert outputs["skipped"] is True
    assert skipped_metadata.status == "skipped"
    assert not (tmp_path / "bv" / "validation" / "metrics.json").exists()
    assert not (tmp_path / "bv" / "validation" / "dv_distribution.png").exists()
