import json
import sys
import types

import numpy as np
import pytest
import tomllib
import yaml
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from atomchain import multibinit_workflow as mw


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, force_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.force_value = float(force_value)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = -float(len(atoms))
        self.results["forces"] = np.full((len(atoms), 3), self.force_value)
        self.results["stress"] = np.zeros(6)


def test_workflow_dry_run_returns_stage_plan(tmp_path):
    result = mw.run_model_build_workflow(
        mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path),
        dry_run=True,
    )

    assert [stage["name"] for stage in result.manifest["stage_plan"]] == mw.STAGE_ORDER
    assert result.passed is None


def test_multibinit_config_paths_are_sibling_relative(tmp_path):
    config_path = tmp_path / "model" / "config" / "mlmb_model.conf"
    ddb = tmp_path / "model" / "ddb" / "source_DDB"

    assert mw._config_relative_path(config_path, ddb) == "../ddb/source_DDB"


def test_stage_checkpoint_outputs_are_relative(tmp_path):
    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path), bulk("Al")
    )
    artifact = context.stage_dir("model") / "fitted.xml"
    artifact.write_text("<model />", encoding="utf-8")

    metadata = mw._run_stage(
        context,
        "train",
        lambda _context, _metadata: {"model_files": [artifact]},
        force=True,
    )

    stage_data = yaml.safe_load(
        (tmp_path / "train" / "stage.yaml").read_text(encoding="utf-8")
    )
    assert metadata.outputs["model_files"] == ["model/fitted.xml"]
    assert stage_data["outputs"]["model_files"] == ["model/fitted.xml"]


def test_random_sampling_accepts_grouped_amplitudes(tmp_path):
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=bulk("Al"),
            output_dir=tmp_path,
            sampling=mw.SamplingStageConfig(
                train_size=4,
                test_size=2,
                random_amplitude=[0.05, 0.1, 0.2],
            ),
        ),
        bulk("Al"),
    )

    frames = mw._sample_random_frames(context)

    assert [item.metadata["amplitude"] for item in frames] == [
        0.05,
        0.1,
        0.2,
        0.05,
        0.1,
        0.2,
    ]
    assert [item.split for item in frames] == ["train"] * 4 + ["test"] * 2


def test_metastable_trajectory_frames_are_sampled_for_training(tmp_path):
    parent = bulk("Al")
    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=parent, output_dir=tmp_path), parent
    )
    meta_dir = context.stage_dir("metastable")
    traj_dir = meta_dir / "relaxation_trajectories"
    traj_dir.mkdir(parents=True)
    trajectory = traj_dir / "relax_001.traj"
    frames = [parent.copy(), parent.copy()]
    frames[1].positions += 0.01
    write(trajectory, frames)
    records = [{"id": 1, "trajectory_file": "relaxation_trajectories/relax_001.traj"}]
    aggregate = meta_dir / "metastable_relaxation_trajectories.traj"
    write(aggregate, mw._metastable_trajectory_atoms(meta_dir, records))
    context.manifest["stages"]["metastable"] = {
        "outputs": {
            "trajectories": context.relative_path(aggregate),
            "records": records,
        }
    }

    sampled = mw._sample_metastable_frames(context)

    assert len(sampled) == 2
    assert {item.source for item in sampled} == {"metastable_trajectory"}
    assert {item.split for item in sampled} == {"train"}


def test_training_supercell_expands_evaluated_quantities(tmp_path):
    parent = bulk("Al")
    frame = parent.copy()
    frame.calc = SinglePointCalculator(
        frame,
        energy=-2.0,
        forces=np.ones((len(frame), 3)),
        stress=np.arange(6, dtype=float),
    )
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            sampling=mw.SamplingStageConfig(supercell=(2, 1, 1)),
        ),
        parent,
    )

    expanded = mw._frames_for_training_supercell(context, [frame])[0]

    assert len(expanded) == 2 * len(parent)
    assert expanded.get_potential_energy() == pytest.approx(-4.0)
    np.testing.assert_allclose(expanded.get_forces(), np.ones((2 * len(parent), 3)))
    np.testing.assert_allclose(
        expanded.get_stress(voigt=True), np.arange(6, dtype=float)
    )
    assert mw._training_ncell(context) == (2, 1, 1)


def test_metastable_target_supercell_uses_record_matrix(tmp_path):
    parent = bulk("Al")
    atoms = parent.repeat((2, 1, 1))
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            sampling=mw.SamplingStageConfig(supercell=(2, 2, 1)),
        ),
        parent,
    )
    record = {"supercell_matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}

    expanded = mw._metastable_atoms_in_target_supercell(
        context, atoms, record, (2, 2, 1)
    )

    assert expanded is not None
    assert len(expanded) == 4 * len(parent)
    assert expanded.info["source_supercell"] == (2, 1, 1)
    assert expanded.info["validation_supercell"] == (2, 2, 1)


def test_reorder_like_reference_supercell_matches_symbols_and_positions():
    reference = bulk("NaCl", "rocksalt", a=5.6).repeat((2, 1, 1))
    shuffled = reference[[1, 0, 3, 2]]

    reordered = mw._reorder_like_reference_supercell(shuffled, reference)

    assert reordered.get_chemical_symbols() == reference.get_chemical_symbols()
    np.testing.assert_allclose(
        reordered.get_scaled_positions(wrap=True),
        reference.get_scaled_positions(wrap=True),
    )


def test_ddb_stage_disables_phonon_parallel_by_default(tmp_path, monkeypatch):
    calls = []

    def fake_write_ddb_from_finite_difference(*args, **kwargs):
        calls.append(kwargs)
        path = kwargs["filename"]
        path.write_text("ddb", encoding="utf-8")
        return path

    ddb_mod = types.ModuleType("atomchain.ddb")
    ddb_mod.write_ddb_from_finite_difference = fake_write_ddb_from_finite_difference
    monkeypatch.setitem(sys.modules, "atomchain.ddb", ddb_mod)
    monkeypatch.setattr(
        mw, "_reference_calculator", lambda context: ConstantCalculator()
    )
    monkeypatch.setattr(
        mw,
        "_write_phonon_band_comparison",
        lambda context, ddb_path, ddb_dir: {
            "figure": "ddb/phonon_band_comparison.png",
            "data": "ddb/phonon_band_comparison.json",
        },
    )

    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path), bulk("Al")
    )
    outputs = mw.run_ddb_stage(context, mw.StageMetadata(name="ddb", status="running"))

    assert outputs["ddb"] == "ddb/model.ddb"
    assert outputs["phonon_band_comparison"] == "ddb/phonon_band_comparison.png"
    assert calls[0]["phonon_kwargs"]["parallel"] is False


def test_ddb_harmonic_validation_records_elastic_and_metrics(tmp_path, monkeypatch):
    parent = bulk("Al", cubic=True)
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            ddb=mw.DdbStageConfig(harmonic_validation_frames=2),
        ),
        parent,
    )
    ddb = tmp_path / "ddb" / "model.ddb"
    ddb.parent.mkdir(parents=True)
    ddb.write_text("ddb", encoding="utf-8")

    parser_mod = types.ModuleType("pymultibinit.pyeffpot.ddb_parser_complete")
    parser_mod.read_ddb = lambda path: types.SimpleNamespace(
        elastic_constants=np.eye(6), strain_coupling=np.ones((6, 3, len(parent)))
    )
    potential_mod = types.ModuleType("pymultibinit.potential")
    calculator_mod = types.ModuleType("pymultibinit.calculator")

    class FakePotential:
        @classmethod
        def from_pyeffpot(cls, *args, **kwargs):
            return "potential"

    class FakeCalculator(ConstantCalculator):
        def __init__(self, potential):
            super().__init__()
            self.potential = potential

    potential_mod.MultibinitPotential = FakePotential
    calculator_mod.MultibinitCalculator = FakeCalculator
    monkeypatch.setitem(sys.modules, "pymultibinit.potential", potential_mod)
    monkeypatch.setitem(sys.modules, "pymultibinit.calculator", calculator_mod)
    monkeypatch.setitem(
        sys.modules, "pymultibinit.pyeffpot.ddb_parser_complete", parser_mod
    )

    result = mw._write_ddb_harmonic_validation(
        context, ddb, ddb.parent, ConstantCalculator()
    )

    data = mw.json.loads((tmp_path / result["metrics"]).read_text(encoding="utf-8"))
    assert data["elastic"]["present"] is True
    assert data["internal_strain"]["present"] is True
    assert data["summary"]["force_rmse_ev_ang"] == pytest.approx(0.0)
    assert data["summary"]["stress_rmse_ev_ang3"] == pytest.approx(0.0)


def test_workflow_runs_mocked_stages_and_reuses_checkpoints(tmp_path):
    calls = []

    def fake_stage(context, metadata):
        calls.append(metadata.name)
        output = context.stage_dir(metadata.name) / f"{metadata.name}.txt"
        output.write_text(metadata.name, encoding="utf-8")
        return {"artifact": context.relative_path(output)}

    stages = {name: fake_stage for name in mw.STAGE_ORDER[:-1]}
    config = mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path)

    result = mw.run_model_build_workflow(config, stage_functions=stages)
    rerun = mw.run_model_build_workflow(config, stage_functions=stages)
    forced = mw.run_model_build_workflow(
        config, stage_functions=stages, force_stage=["sample"]
    )

    assert result.report_md.exists()
    assert (tmp_path / "manifest.yaml").exists()
    assert all((tmp_path / name / "stage.yaml").exists() for name in mw.STAGE_ORDER)
    assert rerun.manifest["stages"]["sample"]["status"] == "reused"
    assert forced.manifest["stages"]["sample"]["status"] == "completed"
    assert calls.count("sample") == 2


def test_force_stage_reruns_downstream_checkpoints(tmp_path):
    calls = []

    def fake_stage(context, metadata):
        calls.append(metadata.name)
        output = context.stage_dir(metadata.name) / f"{metadata.name}.txt"
        output.write_text(metadata.name, encoding="utf-8")
        return {"artifact": context.relative_path(output)}

    stages = {name: fake_stage for name in mw.STAGE_ORDER[:-1]}
    config = mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path)

    mw.run_model_build_workflow(config, stage_functions=stages)
    rerun = mw.run_model_build_workflow(
        config, stage_functions=stages, force_stage=["sample"]
    )

    assert rerun.manifest["stages"]["ddb"]["status"] == "reused"
    assert rerun.manifest["stages"]["sample"]["status"] == "completed"
    assert rerun.manifest["stages"]["evaluate"]["status"] == "completed"
    assert rerun.manifest["stages"]["train"]["status"] == "completed"
    assert calls.count("sample") == 2
    assert calls.count("evaluate") == 2


def test_workflow_records_partial_failure(tmp_path):
    def fail_train(context, metadata):
        raise RuntimeError("training failed")

    stages = {
        name: (fail_train if name == "train" else lambda context, metadata: {})
        for name in mw.STAGE_ORDER[:-1]
    }

    result = mw.run_model_build_workflow(
        mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path),
        stage_functions=stages,
        continue_on_error=True,
    )

    assert result.manifest["stages"]["train"]["status"] == "failed"
    assert result.manifest["stages"]["validate"]["status"] == "skipped"
    assert "training failed" in result.report_md.read_text(encoding="utf-8")


def test_stop_after_persists_manifest_and_report(tmp_path):
    stages = {name: (lambda context, metadata: {}) for name in mw.STAGE_ORDER[:-1]}

    result = mw.run_model_build_workflow(
        mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path),
        stage_functions=stages,
        stop_after="sample",
    )

    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))

    assert result.report_md.exists()
    assert manifest["stages"]["sample"]["status"] == "completed"
    assert manifest["stages"]["evaluate"]["status"] == "skipped"
    assert manifest["stages"]["report"]["status"] == "completed"


def test_non_strict_metastable_failure_is_skipped(tmp_path, monkeypatch):
    import atomchain.metastable as metastable

    parent = bulk("Al")
    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=parent, output_dir=tmp_path), parent
    )

    def fail_explore(*args, **kwargs):
        raise RuntimeError("metastable failed")

    monkeypatch.setattr(metastable, "explore_metastable_states", fail_explore)
    monkeypatch.setattr(
        mw, "_reference_calculator", lambda context: ConstantCalculator()
    )

    metadata = mw.StageMetadata(name="metastable", status="running")
    outputs = mw.run_metastable_stage(context, metadata)

    assert metadata.status == "skipped"
    assert outputs["enabled"] is True
    assert "metastable failed" in outputs["warning"]


def test_sampling_stage_combines_random_and_metastable_sources(tmp_path):
    parent = bulk("Al", cubic=True)
    meta = parent.copy()
    meta.positions += 0.01
    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=parent, output_dir=tmp_path), parent
    )
    context.manifest["stages"]["metastable"] = {
        "outputs": {"records": [{"id": "m1", "atoms": meta, "status": "success"}]}
    }

    outputs = mw.run_sampling_stage(
        context, mw.StageMetadata(name="sample", status="running")
    )

    manifest = yaml.safe_load(
        (tmp_path / "sampling" / "frame_manifest.yaml").read_text(encoding="utf-8")
    )
    train_frames = read(tmp_path / "sampling" / "train_raw.traj", ":")

    assert outputs["train_raw"] == "sampling/train_raw.traj"
    assert {frame["source"] for frame in manifest["frames"]} >= {"random", "metastable"}
    assert len(train_frames) >= 1


def test_sampling_stage_runs_registered_rattle_source(tmp_path):
    parent = bulk("Al", cubic=True)
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            sampling=mw.SamplingStageConfig(
                train_size=0,
                test_size=0,
                include_random=False,
                include_metastable=False,
                sources=[
                    {"name": "rattle", "count": 2, "stdev": 0.01, "split": "test"}
                ],
            ),
            metastable=mw.MetastableStageConfig(enabled=False),
        ),
        parent,
    )

    outputs = mw.run_sampling_stage(
        context, mw.StageMetadata(name="sample", status="running")
    )
    manifest = yaml.safe_load(
        (tmp_path / "sampling" / "frame_manifest.yaml").read_text(encoding="utf-8")
    )
    test_frames = read(tmp_path / "sampling" / "test_raw.traj", ":")

    assert outputs["n_train"] == 0
    assert outputs["n_test"] == 2
    assert len(test_frames) == 2
    assert {frame["source"] for frame in manifest["frames"]} == {"rattle"}
    assert manifest["frames"][0]["metadata"]["stdev"] == pytest.approx(0.01)
    assert manifest["frames"][0]["frame_id"].startswith("source-00-")


def test_repeated_registered_sources_have_unique_frame_ids(tmp_path):
    parent = bulk("Al", cubic=True)
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            sampling=mw.SamplingStageConfig(
                train_size=0,
                test_size=0,
                include_random=False,
                include_metastable=False,
                sources=[
                    {"name": "rattle", "count": 1, "stdev": 0.01, "split": "train"},
                    {"name": "rattle", "count": 1, "stdev": 0.02, "split": "train"},
                ],
            ),
            metastable=mw.MetastableStageConfig(enabled=False),
        ),
        parent,
    )

    mw.run_sampling_stage(context, mw.StageMetadata(name="sample", status="running"))
    manifest = yaml.safe_load(
        (tmp_path / "sampling" / "frame_manifest.yaml").read_text(encoding="utf-8")
    )
    frame_ids = [frame["frame_id"] for frame in manifest["frames"]]

    assert frame_ids == ["source-00-rattle-0000", "source-01-rattle-0000"]
    assert [frame["metadata"]["source_index"] for frame in manifest["frames"]] == [0, 1]


def test_sampling_stage_runs_registered_metastable_interpolation_source(tmp_path):
    parent = bulk("Al", cubic=True)
    meta = parent.copy()
    meta.set_scaled_positions(parent.get_scaled_positions() + [0.1, 0.0, 0.0])
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            sampling=mw.SamplingStageConfig(
                train_size=0,
                test_size=0,
                include_random=False,
                include_metastable=False,
                sources=[{"name": "metastable_interpolation", "lambdas": [0, 1, 3]}],
            ),
        ),
        parent,
    )
    context.manifest["stages"]["metastable"] = {
        "outputs": {"records": [{"id": "m1", "atoms": meta, "status": "success"}]}
    }

    outputs = mw.run_sampling_stage(
        context, mw.StageMetadata(name="sample", status="running")
    )
    manifest = yaml.safe_load(
        (tmp_path / "sampling" / "frame_manifest.yaml").read_text(encoding="utf-8")
    )

    assert outputs["n_train"] == 3
    assert [frame["metadata"]["lambda"] for frame in manifest["frames"]] == [
        0.0,
        1.0,
        3.0,
    ]
    assert manifest["frames"][0]["source"] == "metastable_interpolation"


def test_evaluate_train_validate_report_default_workflow(tmp_path, monkeypatch):
    parent = bulk("Al", cubic=True)

    def fake_ddb(context, metadata):
        ddb = context.stage_dir("ddb") / "model.ddb"
        sidecar = context.stage_dir("ddb") / "model.ddb.yaml"
        comparison = context.stage_dir("ddb") / "phonon_band_comparison.png"
        ddb.write_text("ddb", encoding="utf-8")
        sidecar.write_text("qgrid: [1, 1, 1]\n", encoding="utf-8")
        comparison.write_text("png", encoding="utf-8")
        return {
            "ddb": context.relative_path(ddb),
            "metadata": context.relative_path(sidecar),
            "phonon_band_comparison": context.relative_path(comparison),
        }

    def fake_train(context, metadata):
        model = context.stage_dir("model") / "fitted.xml"
        diagnostics = context.stage_dir("model") / "diagnostics.json"
        model.write_text("<model />", encoding="utf-8")
        diagnostics.write_text("{}", encoding="utf-8")
        return {
            "backend": "python",
            "model_files": [context.relative_path(model)],
            "diagnostics": context.relative_path(diagnostics),
        }

    monkeypatch.setattr(
        mw, "init_calc", lambda model_type, model_path=None: ConstantCalculator()
    )
    monkeypatch.setattr(
        mw,
        "_model_calculator_from_training",
        lambda context: ConstantCalculator(force_value=0.005),
    )
    stages = {"ddb": fake_ddb, "train": fake_train}

    result = mw.run_model_build_workflow(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            sampling=mw.SamplingStageConfig(
                train_size=2, test_size=1, random_amplitude=0.01
            ),
            metastable=mw.MetastableStageConfig(enabled=False),
        ),
        stage_functions=stages,
    )

    assert result.passed is True
    assert result.validation_metrics["metrics"]["forces_ev_ang"][
        "rmse"
    ] == pytest.approx(0.005)
    assert (tmp_path / "training" / "train_HIST.nc").exists()
    assert (tmp_path / "validation" / "metrics.json").exists()
    report = result.report_md.read_text(encoding="utf-8")
    assert "## Procedure" in report
    assert "## Sampling Data" in report
    assert "## Training Data" in report
    assert "## Validation Data" in report
    assert "## Figures" in report
    assert "validation/metrics.json" in report
    assert "![ddb_vs_phonopy_phonon_bands](ddb/phonon_band_comparison.png)" in report
    assert "![force_parity](validation/force_parity.png)" in report
    assert "metrics.forces_ev_ang.rmse" in report


def test_report_markdown_includes_full_input_parameters(tmp_path):
    parent = bulk("Al", cubic=True)
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            ddb=mw.DdbStageConfig(qgrid=(3, 3, 2), phonon_kwargs={"parallel": False}),
            metastable=mw.MetastableStageConfig(
                enabled=True,
                nmax=1,
                options={
                    "amplitude": 0.5,
                    "phonon_ndim": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                    "relax_kwargs": {"fmax": 0.05},
                },
            ),
            sampling=mw.SamplingStageConfig(
                include_random=False,
                include_metastable=False,
                sources=[{"name": "rattle", "count": 3, "stdev": 0.05}],
                supercell=(2, 2, 2),
            ),
            training=mw.TrainingStageConfig(
                options={
                    "basis_ncell": [2, 2, 2],
                    "selection": "screened_greedy",
                    "feature_backend": "memmap",
                    "env": {"API_TOKEN": "secret", "SAFE_VALUE": "visible"},
                }
            ),
            validation=mw.ValidationStageConfig(
                force_rmse_threshold=0.02,
                metastable_records="metastable/manifest.yaml",
            ),
        ),
        parent,
    )

    report = "\n".join(mw._report_markdown(context, tmp_path / "report.yaml"))

    assert "## Input Parameters" in report
    assert "`ddb.qgrid`" in report
    assert "`[3, 3, 2]`" in report
    assert "`metastable.options.phonon_ndim`" in report
    assert "`sampling.sources`" in report
    assert "`training.options.basis_ncell`" in report
    assert "`training.options.env.API_TOKEN` | `<redacted>`" in report
    assert "`training.options.env.SAFE_VALUE` | `visible`" in report
    assert "`validation.metastable_records` | `metastable/manifest.yaml`" in report


def test_report_markdown_includes_complete_metastable_exploration(tmp_path):
    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path), bulk("Al")
    )
    context.manifest["stages"]["metastable"] = {
        "outputs": {
            "structures": "metastable/metastable_states.traj",
            "trajectories": "metastable/metastable_relaxation_trajectories.traj",
            "records": [
                {
                    "id": index,
                    "status": "success",
                    "attempt": index,
                    "amplitude": 0.1 * index,
                    "pre_relax_max_force": float(index),
                    "force_screen_reductions": index - 1,
                    "delta_e_per_fu": -0.01 * index,
                    "combined_label": f"GM4-{index}",
                    "opd_label": "(a,0,0)",
                    "opd_is_maximal": True,
                    "polarization_direction": [1.0, 0.0, 0.0],
                    "source_modes": [
                        {
                            "kpoint_label": "GM",
                            "band_index": index,
                            "frequency": -4.0 - index,
                            "bcs_label": "GM4-",
                        }
                    ],
                    "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "initial_structure_file": f"initial_structures/initial_{index:03d}.vasp",
                    "relaxed_structure_file": f"relaxed_structures/relaxed_{index:03d}.vasp",
                    "trajectory_file": f"relaxation_trajectories/relax_{index:03d}.traj",
                    "structure_file": f"relaxed_structures/relaxed_{index:03d}.vasp",
                    "relaxed_group_id": 1,
                    "relaxed_group_members": [1, 2, 3],
                    "relaxed_group_size": 3,
                    "relaxed_group_representative": index == 1,
                    "spacegroup_name": "P4mm",
                    "spacegroup_number": 99,
                }
                for index in range(1, 13)
            ]
            + [
                {
                    "id": 99,
                    "status": "failed",
                    "error_stage": "relax",
                    "error_message": "did not converge",
                    "attempt": 3,
                    "combined_label": "failed-mode",
                }
            ],
        }
    }

    report = "\n".join(mw._report_metastable_data(context))

    assert "Candidate records: `13`" in report
    assert "Successful candidates: `12`" in report
    assert "Failed/skipped candidates: `1`" in report
    assert "relaxed_structures/relaxed_012.vasp" in report
    assert "relaxation_trajectories/relax_012.traj" in report
    assert "failed-mode" in report
    assert "did not converge" in report
    assert "### Metastable Candidate 12" in report
    assert "`source_modes.0.frequency`" in report
    assert "`polarization_direction`" in report
    assert "`relaxed_group_members`" in report


def test_report_relativizes_absolute_artifact_paths(tmp_path):
    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=bulk("Al"), output_dir=tmp_path), bulk("Al")
    )
    absolute_model = tmp_path / "model" / "fitted.xml"
    absolute_basis = tmp_path / "model" / "basis.xml"
    context.manifest["stages"]["train"] = {
        "outputs": {
            "model_files": [str(absolute_model)],
            "diagnostics": {"basis_xml": str(absolute_basis)},
        }
    }

    report = "\n".join(mw._report_markdown(context, tmp_path / "report.yaml"))
    normalized = mw._relativize_manifest_paths(context, context.manifest)

    assert f"| `train` | `{tmp_path}" not in report
    assert f"basis_xml: `{tmp_path}" not in report
    assert "`model/fitted.xml`" in report
    assert "basis_xml: `model/basis.xml`" in report
    assert normalized["stages"]["train"]["outputs"]["model_files"] == [
        "model/fitted.xml"
    ]
    assert normalized["config"]["output_dir"] == str(tmp_path)


def test_training_stage_dispatches_python_and_binary_backends(tmp_path, monkeypatch):
    parent = bulk("Al")
    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=parent, output_dir=tmp_path), parent
    )
    ddb = context.stage_dir("ddb") / "model.ddb"
    hist = context.stage_dir("training") / "train_HIST.nc"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")
    context.manifest["stages"] = {
        "ddb": {"outputs": {"ddb": context.relative_path(ddb)}},
        "evaluate": {"outputs": {"train_hist": context.relative_path(hist)}},
    }

    py_calls = []
    bin_calls = []
    training_mod = types.ModuleType("pymultibinit.training")

    def fake_fit_multibinit_model_python(**kwargs):
        py_calls.append(kwargs)
        output_xml = tmp_path / "model" / "python.xml"
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        output_xml.write_text("xml", encoding="utf-8")
        return types.SimpleNamespace(
            output_xml=str(output_xml),
            to_dict=lambda: {
                "output_xml": str(output_xml),
                "coefficients": [0.0, 1.5, 0.0, -2.0],
                "ncoeff": 4,
            },
        )

    def fake_train_multibinit_model(**kwargs):
        bin_calls.append(kwargs)
        return {"artifacts": {"model": "binary.xml"}, "model_config": "model.conf"}

    training_mod.fit_multibinit_model_python = fake_fit_multibinit_model_python
    training_mod.train_multibinit_model = fake_train_multibinit_model
    monkeypatch.setitem(sys.modules, "pymultibinit.training", training_mod)
    monkeypatch.setattr(
        mw,
        "_python_fit_config",
        lambda context: types.SimpleNamespace(ncoeff=20, regularization=1e-8),
    )
    fixed_model = object()
    monkeypatch.setattr(mw, "_python_fixed_model", lambda context, ddb: fixed_model)
    basis_xml = tmp_path / "model" / "basis.xml"
    basis_xml.parent.mkdir(parents=True, exist_ok=True)
    basis_xml.write_text("<model />", encoding="utf-8")
    monkeypatch.setattr(mw, "_basis_xml", lambda context, model_dir: basis_xml)

    context.config.training.backend = "python"
    py_outputs = mw.run_training_stage(
        context, mw.StageMetadata(name="train", status="running")
    )
    context.config.training.backend = "binary"
    bin_outputs = mw.run_training_stage(
        context, mw.StageMetadata(name="train", status="running")
    )

    assert py_calls and py_outputs["backend"] == "python"
    assert py_calls[0]["basis_xml"] == str(basis_xml)
    assert py_calls[0]["config"].ncoeff == 20
    assert py_calls[0]["config"].regularization == pytest.approx(1e-8)
    assert py_calls[0]["fixed_model"] is fixed_model
    assert "coefficients" not in py_outputs["diagnostics"]
    assert py_outputs["diagnostics"]["coefficient_count"] == 4
    assert py_outputs["diagnostics"]["nonzero_coefficient_count"] == 2
    assert py_outputs["diagnostics"]["nonzero_coefficient_indices"] == [1, 3]
    assert bin_calls and bin_outputs["backend"] == "binary"
    assert "ncoeff" not in bin_calls[0]
    assert "regularization" not in bin_calls[0]


def test_basis_xml_reuses_only_matching_fingerprint(tmp_path, monkeypatch):
    parent = bulk("Al")
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            training=mw.TrainingStageConfig(
                options={
                    "basis_ncell": [1, 1, 1],
                    "use_symmetry": False,
                    "power_range": [3, 4],
                }
            ),
        ),
        parent,
    )
    model_dir = context.stage_dir("model")
    basis = model_dir / "basis.xml"
    diagnostics = model_dir / "basis_pair_diagnostics.json"
    basis.write_text("existing", encoding="utf-8")
    cutoff = mw._basis_cutoff(context)
    ncell = mw._basis_ncell(context)
    fingerprint = mw._basis_request_fingerprint(
        context, cutoff, ncell, context.config.training.options
    )
    diagnostics.write_text(
        json.dumps(
            {
                "ncell": [1, 1, 1],
                "cutoff": cutoff,
                "n_factors": 1,
                "n_symmetry_operations": 1,
                "symmetry_closed": True,
                "missing_mapped_factors_count": 0,
                "max_pair_distance": 1.0,
                "basis_request_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    training_mod = types.ModuleType("pymultibinit.training")
    training_mod.displacement_pair_diagnostics = lambda *args, **kwargs: calls.append(
        "diagnostics"
    ) or {
        "ncell": [1, 1, 1],
        "cutoff": cutoff,
        "n_factors": 1,
        "n_symmetry_operations": 1,
        "symmetry_closed": True,
        "missing_mapped_factors_count": 0,
        "max_pair_distance": 1.0,
    }
    training_mod.generate_displacement_basis = (
        lambda *args, **kwargs: calls.append("generate") or []
    )
    training_mod.with_fortran_text_labels = lambda basis, symbols: basis
    training_mod.write_fitted_xml = lambda path, basis: calls.append(
        "write"
    ) or path.write_text("new", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "pymultibinit.training", training_mod)

    assert mw._basis_xml(context, model_dir) == basis
    assert calls == []

    context.config.training.options["power_range"] = [2, 2]
    assert mw._basis_xml(context, model_dir) == basis
    assert calls == ["diagnostics", "generate", "write"]


def test_basis_xml_recomputed_nonclosed_diagnostics_still_raise(tmp_path, monkeypatch):
    parent = bulk("Al")
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=parent,
            output_dir=tmp_path,
            training=mw.TrainingStageConfig(
                options={
                    "basis_ncell": [1, 1, 1],
                    "use_symmetry": True,
                    "require_basis_symmetry_closed": True,
                }
            ),
        ),
        parent,
    )
    model_dir = context.stage_dir("model")
    basis = model_dir / "basis.xml"
    diagnostics = model_dir / "basis_pair_diagnostics.json"
    basis.write_text("existing", encoding="utf-8")
    cutoff = mw._basis_cutoff(context)
    ncell = mw._basis_ncell(context)
    fingerprint = mw._basis_request_fingerprint(
        context, cutoff, ncell, context.config.training.options
    )
    nonclosed = {
        "ncell": [1, 1, 1],
        "cutoff": cutoff,
        "n_factors": 1,
        "n_symmetry_operations": 1,
        "symmetry_closed": False,
        "missing_mapped_factors_count": 1,
        "max_pair_distance": 1.0,
        "basis_request_fingerprint": fingerprint,
    }
    diagnostics.write_text(json.dumps(nonclosed), encoding="utf-8")
    training_mod = types.ModuleType("pymultibinit.training")
    training_mod.displacement_pair_diagnostics = lambda *args, **kwargs: dict(nonclosed)
    training_mod.generate_displacement_basis = lambda *args, **kwargs: []
    training_mod.with_fortran_text_labels = lambda basis, symbols: basis
    training_mod.write_fitted_xml = lambda path, basis: None
    monkeypatch.setitem(sys.modules, "pymultibinit.training", training_mod)

    with pytest.raises(ValueError, match="not closed"):
        mw._basis_xml(context, model_dir)


def test_sensitive_config_values_are_redacted_from_artifacts():
    config = mw.WorkflowConfig(
        structure=bulk("Al"),
        training=mw.TrainingStageConfig(
            options={"env": {"API_TOKEN": "secret", "SAFE_VALUE": "visible"}}
        ),
    )

    data = mw._config_to_dict(config)

    env = data["training"]["options"]["env"]
    assert env["API_TOKEN"] == "<redacted>"
    assert env["SAFE_VALUE"] == "visible"


def test_metastable_record_paths_are_confined(tmp_path):
    base = tmp_path / "metastable"
    base.mkdir()
    assert mw._confined_path(base, "relaxed/ok.vasp") == base / "relaxed" / "ok.vasp"
    assert mw._confined_path(base, "../outside.vasp") is None
    assert mw._confined_path(base, tmp_path / "outside.vasp") is None


def test_metastable_atoms_from_records_confines_structure_paths(tmp_path):
    base = tmp_path / "metastable"
    base.mkdir()
    inside = base / "inside.vasp"
    outside = tmp_path / "outside.vasp"
    write(inside, bulk("Al"), format="vasp")
    write(outside, bulk("Cu"), format="vasp")

    atoms = mw._metastable_atoms_from_records(
        [
            {"id": "outside", "structure_file": "../outside.vasp"},
            {"id": "inside", "structure_file": "inside.vasp"},
        ],
        base,
    )

    assert len(atoms) == 1
    assert atoms[0].info["metastable_id"] == "inside"


def test_python_training_validation_uses_pyeffpot_without_libabinit(
    tmp_path, monkeypatch
):
    parent = bulk("Al")
    context = mw.WorkflowContext(
        mw.WorkflowConfig(structure=parent, output_dir=tmp_path), parent
    )
    ddb = context.stage_dir("ddb") / "model.ddb"
    coeff = context.stage_dir("model") / "fitted.xml"
    ddb.write_text("ddb", encoding="utf-8")
    coeff.write_text("xml", encoding="utf-8")
    context.manifest["stages"]["train"] = {
        "outputs": {
            "backend": "python",
            "ddb": context.relative_path(ddb),
            "coeff_file": context.relative_path(coeff),
            "model_config": "model/config.conf",
            "model_files": ["model/config.conf", "model/fitted.xml"],
        }
    }

    calls = []
    calculator_mod = types.ModuleType("pymultibinit.calculator")
    potential_mod = types.ModuleType("pymultibinit.potential")

    class FakeCalculator:
        def __init__(self, potential):
            self.potential = potential

    class FakePotential:
        @classmethod
        def from_pyeffpot(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "potential"

    calculator_mod.MultibinitCalculator = FakeCalculator
    potential_mod.MultibinitPotential = FakePotential
    monkeypatch.setitem(sys.modules, "pymultibinit.calculator", calculator_mod)
    monkeypatch.setitem(sys.modules, "pymultibinit.potential", potential_mod)
    monkeypatch.setattr(
        mw,
        "init_calc",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loaded lib path")
        ),
    )

    calc = mw._model_calculator_from_training(context)

    assert isinstance(calc, FakeCalculator)
    assert calls[0][0] == (str(ddb),)
    assert calls[0][1]["xml_file"] == str(coeff)


def test_mlmbmodel_cli_dry_run_config_and_packaging(tmp_path, monkeypatch, capsys):
    structure = tmp_path / "POSCAR"
    write(structure, bulk("Al"))
    config = tmp_path / "workflow.yaml"
    config.write_text("sampling:\n  train_size: 3\n  test_size: 2\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlmbmodel",
            str(structure),
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )

    assert mw.mlmbmodel_cli() == 0
    assert "prepare" in capsys.readouterr().out

    pyproject = tomllib.loads(
        (mw.Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    assert (
        pyproject["project"]["scripts"]["mlmbmodel"]
        == "atomchain.multibinit_workflow:mlmbmodel_cli"
    )


def test_docs_are_registered():
    root = mw.Path(__file__).parents[1]
    docs = (root / "docs" / "multibinit_workflow.md").read_text(encoding="utf-8")
    index = (root / "docs" / "README.md").read_text(encoding="utf-8")

    assert "mlmbmodel" in docs
    assert "atomchain.multibinit_workflow" in docs
    assert "multibinit_workflow.md" in index
