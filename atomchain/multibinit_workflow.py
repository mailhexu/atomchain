"""End-to-end MULTIBINIT model-building workflow orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from atomchain.init_model import init_calc
from atomchain.io.hist import write_abinit_hist
from atomchain.training.evaluate import evaluate_training_frames
from atomchain.training.validation import (
    validate_fitted_model,
    validate_metastable_energy_differences,
)

try:
    from pymultibinit.training import FORTRAN_ANCHORED_GENERATOR_TAG
except (ImportError, AttributeError):
    # Older optional pymultibinit installations do not expose the provenance tag.
    FORTRAN_ANCHORED_GENERATOR_TAG = "fortran_anchored_v2"


STAGE_ORDER = [
    "prepare",
    "ddb",
    "metastable",
    "sample",
    "evaluate",
    "train",
    "validate",
    "report",
]
STAGE_SIGNATURE_VERSIONS = {"train": 5, "validate": 5}


@dataclass
class DdbStageConfig:
    phonon_supercell: tuple[int, int, int] = (2, 2, 2)
    qgrid: tuple[int, int, int] = (2, 2, 2)
    include_elastic: bool = True
    include_internal_strain: bool = True
    strain_amplitude: float = 1e-3
    difference: str = "central"
    validate_parse: bool = True
    phonon_kwargs: dict[str, Any] = field(default_factory=dict)
    validate_harmonic: bool = True
    harmonic_validation_frames: int = 5
    harmonic_displacement_amplitude: float = 1e-4
    harmonic_strain_amplitude: float = 1e-5


@dataclass
class MetastableStageConfig:
    enabled: bool = True
    nmax: int = 2
    max_cell_size: int = 80
    strict: bool = False
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingStageConfig:
    train_size: int = 20
    test_size: int = 5
    random_amplitude: float | list[float] = 0.02
    include_random: bool = True
    include_metastable: bool = True
    sources: list[dict[str, Any]] = field(default_factory=list)
    supercell: tuple[int, int, int] | None = None


@dataclass
class TrainingStageConfig:
    backend: Literal["python", "binary"] = "python"
    ncoeff: int = 20
    regularization: float = 1e-8
    basis_xml: str | None = None
    output_xml: str = "fitted.nc"
    binary_config: str | None = None
    executable: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationStageConfig:
    enabled: bool = True
    force_rmse_threshold: float = 0.01
    include_stress: bool = True
    metastable_energy_differences: bool = True
    metastable_records: str | None = None
    metastable_energy_tol: float = 1e-3


@dataclass
class WorkflowConfig:
    structure: str | Path | Atoms
    model: str = "mace-r2scan"
    model_path: str | None = None
    output_dir: str | Path = "multibinit_model_workflow"
    seed: int = 1234
    ddb: DdbStageConfig = field(default_factory=DdbStageConfig)
    metastable: MetastableStageConfig = field(default_factory=MetastableStageConfig)
    sampling: SamplingStageConfig = field(default_factory=SamplingStageConfig)
    training: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    validation: ValidationStageConfig = field(default_factory=ValidationStageConfig)


@dataclass
class StageMetadata:
    name: str
    status: str = "pending"
    signature: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    elapsed_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _sanitize(asdict(self))


@dataclass
class SampledFrame:
    atoms: Atoms
    split: Literal["train", "test"]
    source: str
    frame_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    output_dir: Path
    manifest: dict[str, Any]
    report_md: Path
    report_yaml: Path
    model_files: list[Path]
    validation_metrics: dict[str, Any]
    passed: bool | None


class WorkflowContext:
    """Mutable workflow context shared by stages."""

    def __init__(self, config: WorkflowConfig, parent_atoms: Atoms):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.parent_atoms = parent_atoms.copy()
        self.manifest: dict[str, Any] = {
            "format": "atomchain-multibinit-workflow-v1",
            "config": _config_to_dict(config),
            "stages": {},
        }
        self.reference_calculator = None
        self.reference_stress_ha_bohr3: np.ndarray | None = None

    def stage_dir(self, name: str) -> Path:
        mapping = {
            "evaluate": "training",
            "validate": "validation",
            "sample": "sampling",
        }
        path = self.output_dir / mapping.get(name, name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def checkpoint_dir(self, name: str) -> Path:
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def relative_path(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.output_dir.resolve()).as_posix()

    def path_from_manifest(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.output_dir / path


StageFunction = Callable[[WorkflowContext, StageMetadata], dict[str, Any]]


def run_model_build_workflow(
    config: WorkflowConfig | None = None,
    *,
    dry_run: bool = False,
    stage_functions: dict[str, StageFunction] | None = None,
    force_stage: list[str] | None = None,
    force_all: bool = False,
    stop_after: str | None = None,
    continue_on_error: bool = False,
    **kwargs: Any,
) -> WorkflowResult:
    """Run the full MULTIBINIT model-building workflow."""

    cfg = _coerce_config(config, kwargs)
    parent = _load_parent_atoms(cfg.structure)
    context = WorkflowContext(cfg, parent)
    if dry_run:
        context.manifest["stage_plan"] = [{"name": name} for name in STAGE_ORDER]
        return _result_from_context(context, passed=None)

    context.output_dir.mkdir(parents=True, exist_ok=True)
    _restore_prepare_state(context)
    stage_map = _default_stage_functions()
    stage_map.update(stage_functions or {})
    forced = set(force_stage or [])
    unknown_forced = forced - set(STAGE_ORDER)
    if unknown_forced:
        raise ValueError(
            f"Unknown workflow stage(s): {', '.join(sorted(unknown_forced))}"
        )
    force_from = min((STAGE_ORDER.index(name) for name in forced), default=None)
    failed = False
    for name in STAGE_ORDER:
        if (
            stop_after is not None
            and name != "report"
            and STAGE_ORDER.index(name) > STAGE_ORDER.index(stop_after)
        ):
            context.manifest["stages"][name] = StageMetadata(
                name=name, status="skipped", warnings=["Skipped by stop_after"]
            ).to_dict()
            _write_manifest(context)
            continue
        if failed and name != "report":
            context.manifest["stages"][name] = StageMetadata(
                name=name,
                status="skipped",
                warnings=["Skipped after previous stage failure"],
            ).to_dict()
            continue
        metadata = _run_stage(
            context,
            name,
            stage_map[name],
            force=(
                force_all
                or name == "report"
                or (force_from is not None and STAGE_ORDER.index(name) >= force_from)
            ),
        )
        context.manifest["stages"][name] = metadata.to_dict()
        _write_manifest(context)
        if metadata.status == "failed":
            failed = True
            if not continue_on_error and name != "report":
                # Still write a partial report before surfacing the failure.
                report_metadata = _run_stage(
                    context, "report", stage_map["report"], force=True
                )
                context.manifest["stages"]["report"] = report_metadata.to_dict()
                _write_manifest(context)
                raise RuntimeError(metadata.error)
    return _result_from_context(context)


def run_prepare_stage(
    context: WorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    outdir = context.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    relax_parent = context.config.training.options.get("relax_parent", False)
    if relax_parent:
        context.parent_atoms = _relax_parent(context)

    _compute_reference_stress(context)

    parent_path = outdir / "parent.vasp"
    config_path = outdir / "config.yaml"
    write(parent_path, context.parent_atoms, format="vasp")
    _write_yaml(_config_to_dict(context.config), config_path)
    outputs = {
        "parent_structure": context.relative_path(parent_path),
        "config": context.relative_path(config_path),
    }
    if context.reference_stress_ha_bohr3 is not None:
        outputs["reference_stress_ha_bohr3"] = (
            context.reference_stress_ha_bohr3.tolist()
        )
    return outputs


def _restore_prepare_state(context: WorkflowContext) -> None:
    """Restore relaxed parent and reference stress from a cached prepare stage.

    Stage reuse returns cached metadata without re-executing prepare, so a fresh
    process would otherwise carry the unrelaxed input structure and a missing
    reference stress into downstream stages (basis anchoring, stress validation).
    """
    checkpoint = context.checkpoint_dir("prepare") / "stage.yaml"
    if not checkpoint.exists():
        return
    try:
        previous = yaml.safe_load(checkpoint.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    if previous.get("status") not in ("completed", "reused"):
        return
    outputs = previous.get("outputs") or {}
    parent_path = context.path_from_manifest(outputs.get("parent_structure"))
    if parent_path is not None and parent_path.exists():
        try:
            context.parent_atoms = read(parent_path)
        except Exception:
            return
    refstress = outputs.get("reference_stress_ha_bohr3")
    if refstress is not None:
        context.reference_stress_ha_bohr3 = np.array(refstress, dtype=float)


def _relax_parent(context: WorkflowContext) -> Atoms:
    from ase.filters import UnitCellFilter
    from ase.optimize import BFGS

    calc = _reference_calculator(context)
    atoms = context.parent_atoms.copy()
    atoms.calc = calc
    fmax = float(context.config.training.options.get("relax_fmax", 0.01))
    smax_gpa = float(context.config.training.options.get("relax_smax_gpa", 0.5))
    smax_ev_ang3 = smax_gpa / 160.21766
    uf = UnitCellFilter(atoms)
    opt = BFGS(uf, logfile=None)
    for _ in range(200):
        opt.run(fmax=fmax, steps=1)
        stress = atoms.get_stress(voigt=True)
        if (
            np.max(np.abs(stress[:3])) < smax_ev_ang3
            and np.max(np.abs(stress[3:])) < smax_ev_ang3
        ):
            break
    atoms.calc = None
    return atoms


def _compute_reference_stress(context: WorkflowContext) -> None:
    calc = _reference_calculator(context)
    atoms = context.parent_atoms.copy()
    atoms.calc = calc
    stress_ev_ang3 = atoms.get_stress(voigt=True)
    BOHR3_TO_ANG3 = 0.529177**3
    HA_TO_EV = 27.211386
    stress_ha_bohr3 = np.array(stress_ev_ang3, dtype=float) * BOHR3_TO_ANG3 / HA_TO_EV
    context.reference_stress_ha_bohr3 = stress_ha_bohr3


def run_ddb_stage(context: WorkflowContext, metadata: StageMetadata) -> dict[str, Any]:
    from atomchain.ddb import write_ddb_from_finite_difference

    calc = _reference_calculator(context)
    ddb_dir = context.stage_dir("ddb")
    ddb_path = ddb_dir / "model.ddb"
    written = write_ddb_from_finite_difference(
        context.parent_atoms,
        calc=calc,
        filename=ddb_path,
        phonon_ndim=context.config.ddb.phonon_supercell,
        qgrid=context.config.ddb.qgrid,
        strain_amplitude=context.config.ddb.strain_amplitude,
        include_stress=context.config.ddb.include_elastic,
        include_strain_phonon=context.config.ddb.include_internal_strain,
        difference=context.config.ddb.difference,
        cache_dir=ddb_dir / "fd_cache",
        phonon_kwargs={"parallel": False, **context.config.ddb.phonon_kwargs},
    )
    sidecar = Path(f"{ddb_path}.yaml")
    outputs = {
        "ddb": context.relative_path(written),
        "metadata": context.relative_path(sidecar) if sidecar.exists() else None,
        "qgrid": list(context.config.ddb.qgrid),
        "include_elastic": context.config.ddb.include_elastic,
        "include_internal_strain": context.config.ddb.include_internal_strain,
    }
    if context.config.ddb.validate_parse:
        outputs["pymultibinit_parse"] = _validate_ddb_parse(written)
    comparison = _write_phonon_band_comparison(context, written, ddb_dir)
    if comparison.get("warning"):
        metadata.warnings.append(comparison["warning"])
    if comparison.get("figure"):
        outputs["phonon_band_comparison"] = comparison["figure"]
    if comparison.get("data"):
        outputs["phonon_band_data"] = comparison["data"]
    harmonic = _write_ddb_harmonic_validation(context, written, ddb_dir, calc)
    if harmonic.get("warning"):
        metadata.warnings.append(harmonic["warning"])
    if harmonic.get("metrics"):
        outputs["harmonic_validation"] = harmonic["metrics"]
    return outputs


def run_metastable_stage(
    context: WorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    if not context.config.metastable.enabled:
        metadata.status = "skipped"
        return {"enabled": False}
    from atomchain.metastable import explore_metastable_states

    meta_dir = context.stage_dir("metastable")
    try:
        result = explore_metastable_states(
            context.parent_atoms,
            calc=_reference_calculator(context),
            nmax=context.config.metastable.nmax,
            max_cell_size=context.config.metastable.max_cell_size,
            output_dir=meta_dir,
            **context.config.metastable.options,
        )
    except Exception as exc:
        if context.config.metastable.strict:
            raise
        metadata.status = "skipped"
        metadata.warnings.append(str(exc))
        return {
            "enabled": True,
            "output_dir": context.relative_path(meta_dir),
            "records": [],
            "warning": str(exc),
        }
    records = result.get("results", result) if isinstance(result, dict) else result
    states = _metastable_atoms_from_records(records, meta_dir)
    structures_path = meta_dir / "metastable_states.traj"
    trajectories_path = meta_dir / "metastable_relaxation_trajectories.traj"
    if states:
        write(structures_path, states)
    trajectory_frames = _metastable_trajectory_atoms(meta_dir, records)
    if trajectory_frames:
        write(trajectories_path, trajectory_frames)
    outputs = {
        "enabled": True,
        "output_dir": context.relative_path(meta_dir),
        "report_yaml": "metastable/report.yaml"
        if (meta_dir / "report.yaml").exists()
        else None,
        "report_md": "metastable/report.md"
        if (meta_dir / "report.md").exists()
        else None,
        "structures": context.relative_path(structures_path) if states else None,
        "trajectories": context.relative_path(trajectories_path)
        if trajectory_frames
        else None,
        "records": records,
    }
    return outputs


def run_sampling_stage(
    context: WorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    frames = []
    if context.config.sampling.include_random:
        frames.extend(_sample_random_frames(context))
    if context.config.sampling.include_metastable:
        frames.extend(_sample_metastable_frames(context))
    for source_index, source in enumerate(context.config.sampling.sources):
        frames.extend(
            _run_registered_sampler(context, source, source_index=source_index)
        )
    if not frames:
        raise ValueError("Sampling produced no frames")
    train = [item.atoms for item in frames if item.split == "train"]
    test = [item.atoms for item in frames if item.split == "test"]
    sampling_dir = context.stage_dir("sample")
    train_path = sampling_dir / "train_raw.traj"
    test_path = sampling_dir / "test_raw.traj"
    manifest_path = sampling_dir / "frame_manifest.yaml"
    write(train_path, train)
    write(test_path, test)
    manifest = {
        "frames": [
            {
                "frame_id": item.frame_id,
                "split": item.split,
                "source": item.source,
                "metadata": _sanitize(item.metadata),
            }
            for item in frames
        ]
    }
    _write_yaml(manifest, manifest_path)
    return {
        "train_raw": context.relative_path(train_path),
        "test_raw": context.relative_path(test_path),
        "frame_manifest": context.relative_path(manifest_path),
        "n_train": len(train),
        "n_test": len(test),
    }


def run_evaluate_stage(
    context: WorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    sample_outputs = context.manifest["stages"].get("sample", {}).get("outputs", {})
    train_raw = context.path_from_manifest(sample_outputs.get("train_raw"))
    test_raw = context.path_from_manifest(sample_outputs.get("test_raw"))
    if train_raw is None or test_raw is None:
        raise ValueError("Sampling outputs are missing train/test trajectories")
    train_frames = _read_frames(train_raw)
    test_frames = _read_frames(test_raw)
    training_dir = context.stage_dir("evaluate")
    train_traj = training_dir / "train.traj"
    test_traj = training_dir / "test.traj"
    train_hist = training_dir / "train_HIST.nc"
    test_hist = training_dir / "test_HIST.nc"
    calc = _reference_calculator(context)
    evaluated_train = evaluate_training_frames(
        train_frames, calculator=calc, provenance={"split": "train"}
    )
    evaluated_test = evaluate_training_frames(
        test_frames, calculator=calc, provenance={"split": "test"}
    )
    evaluated_train = _frames_for_training_supercell(context, evaluated_train)
    evaluated_test = _frames_for_training_supercell(context, evaluated_test)
    write(train_traj, evaluated_train)
    write(test_traj, evaluated_test)
    write_abinit_hist(evaluated_train, train_hist, metadata=f"{train_hist}.yaml")
    write_abinit_hist(evaluated_test, test_hist, metadata=f"{test_hist}.yaml")
    return {
        "train_traj": context.relative_path(train_traj),
        "test_traj": context.relative_path(test_traj),
        "train_hist": context.relative_path(train_hist),
        "test_hist": context.relative_path(test_hist),
        "train_hist_metadata": context.relative_path(f"{train_hist}.yaml"),
        "test_hist_metadata": context.relative_path(f"{test_hist}.yaml"),
    }


def run_training_stage(
    context: WorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    ddb = context.path_from_manifest(
        context.manifest["stages"].get("ddb", {}).get("outputs", {}).get("ddb")
    )
    train_hist = context.path_from_manifest(
        context.manifest["stages"]
        .get("evaluate", {})
        .get("outputs", {})
        .get("train_hist")
    )
    if ddb is None or train_hist is None:
        raise ValueError("Training requires DDB and train HIST outputs")
    model_dir = context.stage_dir("model")
    backend = context.config.training.backend
    if backend == "python":
        from pymultibinit.training import fit_multibinit_model_python

        output_xml = model_dir / context.config.training.output_xml
        fit_config = _python_fit_config(context)
        basis_xml = _basis_xml(context, model_dir, fit_config=fit_config)
        kwargs = _python_fit_kwargs(context, train_hist=train_hist)
        kwargs.setdefault("fixed_model", _python_fixed_model(context, ddb))
        raw = fit_multibinit_model_python(
            ddb=str(ddb),
            hist=str(train_hist),
            basis_xml=str(basis_xml),
            output_xml=str(output_xml),
            config=fit_config,
            **kwargs,
        )
        data = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw or {})
        diagnostics = _compact_python_fit_diagnostics(data)
        model_files = [data.get("output_xml") or str(output_xml)]
        model_config = _write_multibinit_model_config(
            context, model_dir, ddb=ddb, coeff_file=Path(model_files[0])
        )
        return {
            "backend": backend,
            "model_config": context.relative_path(model_config),
            "ddb": context.relative_path(ddb),
            "coeff_file": _rel_or_abs(context, model_files[0]),
            "model_files": [
                context.relative_path(model_config),
                *[_rel_or_abs(context, item) for item in model_files],
            ],
            "diagnostics": diagnostics,
            "basis_pair_diagnostics": context.relative_path(
                model_dir / "basis_pair_diagnostics.json"
            )
            if (model_dir / "basis_pair_diagnostics.json").exists()
            else None,
        }
    if backend == "binary":
        from pymultibinit.training import train_multibinit_model

        kwargs = _binary_training_kwargs(context)
        raw = train_multibinit_model(
            ddb=str(ddb),
            hist=str(train_hist),
            config=context.config.training.binary_config,
            output_dir=str(model_dir),
            executable=context.config.training.executable,
            **kwargs,
        )
        data = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw or {})
        artifacts = data.get("artifacts", {}) or {}
        model_files = [data.get("model_config"), *artifacts.values()]
        return {
            "backend": backend,
            "model_files": [_rel_or_abs(context, item) for item in model_files if item],
            "diagnostics": data,
        }
    raise ValueError(f"Unsupported training backend: {backend}")


def _compact_python_fit_diagnostics(data: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(data)
    coefficients = diagnostics.pop("coefficients", None)
    if coefficients is not None:
        diagnostics["coefficient_count"] = len(coefficients)
        nonzero = [
            index for index, value in enumerate(coefficients) if abs(float(value)) > 0.0
        ]
        diagnostics["nonzero_coefficient_count"] = len(nonzero)
        diagnostics["nonzero_coefficient_indices"] = nonzero[:200]
        diagnostics["nonzero_coefficient_indices_truncated"] = len(nonzero) > 200
    return diagnostics


def run_validate_stage(
    context: WorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    if not context.config.validation.enabled:
        metadata.status = "skipped"
        return {"enabled": False}
    evaluate_outputs = context.manifest["stages"].get("evaluate", {}).get("outputs", {})
    test_traj = context.path_from_manifest(evaluate_outputs.get("test_traj"))
    if test_traj is None:
        raise ValueError("Validation requires evaluated test trajectory")
    validation_dir = context.stage_dir("validate")
    metrics_path = validation_dir / "metrics.json"
    model_calc = _model_calculator_from_training(context)
    result = validate_fitted_model(
        test_traj,
        calculator=model_calc,
        output=metrics_path,
        evaluated_trajectory=validation_dir / "model_test.traj",
        include_stress=context.config.validation.include_stress,
    )
    metrics = result.to_dict()
    force_rmse = metrics["metrics"]["forces_ev_ang"]["rmse"]
    passed = force_rmse <= context.config.validation.force_rmse_threshold
    metrics["passed"] = passed
    metrics["thresholds"] = {
        "force_rmse_ev_ang": context.config.validation.force_rmse_threshold
    }
    _write_json(metrics, metrics_path)
    plots = _write_validation_plots(
        test_traj, validation_dir / "model_test.traj", metrics, validation_dir
    )
    outputs = {
        "enabled": True,
        "metrics": context.relative_path(metrics_path),
        "passed": passed,
        "plots": plots,
    }
    if context.config.validation.metastable_energy_differences:
        records, record_base_dir = _validation_metastable_records(context)
        if records:
            table_path = validation_dir / "metastable_energy_table.json"
            table = _validate_representative_metastable_energies(
                context,
                records,
                record_base_dir,
                model_calc,
                output=table_path,
            )
            outputs["metastable_energy_table"] = context.relative_path(table_path)
            metrics["metastable_energy_table"] = table
            states = _metastable_atoms_from_records(records, record_base_dir)
            if (
                states
                and context.parent_atoms.calc is not None
                and all(item.calc is not None for item in states)
            ):
                meta_path = validation_dir / "metastable_energy_differences.json"
                meta_result = validate_metastable_energy_differences(
                    context.parent_atoms,
                    states,
                    calculator=model_calc,
                    output=meta_path,
                )
                outputs["metastable_energy_differences"] = context.relative_path(
                    meta_path
                )
                metrics["metastable_energy_differences"] = meta_result.to_dict()
            _write_json(metrics, metrics_path)
    return outputs


def run_report_stage(
    context: WorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    report_yaml = context.output_dir / "report.yaml"
    report_md = context.output_dir / "report.md"
    context.manifest = _relativize_manifest_paths(context, context.manifest)
    context.manifest["stages"]["report"] = StageMetadata(
        name="report",
        status="completed",
        outputs={
            "report_yaml": context.relative_path(report_yaml),
            "report_md": context.relative_path(report_md),
        },
    ).to_dict()
    _write_yaml(context.manifest, report_yaml)
    lines = _report_markdown(context, report_yaml)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report_yaml": context.relative_path(report_yaml),
        "report_md": context.relative_path(report_md),
    }


def _report_markdown(context: WorkflowContext, report_yaml: Path) -> list[str]:
    validation_outputs = (
        context.manifest.get("stages", {}).get("validate", {}).get("outputs", {})
    )
    metrics = _read_report_metrics(context, validation_outputs)
    lines = ["# MULTIBINIT Model Workflow Report", ""]
    lines.extend(_report_summary(context, metrics, report_yaml))
    lines.extend(_report_input_parameters(context))
    lines.extend(_report_procedure(context))
    lines.extend(_report_stage_table(context))
    lines.extend(_report_artifacts(context))
    lines.extend(_report_ddb_data(context))
    lines.extend(_report_metastable_data(context))
    lines.extend(_report_sampling_data(context))
    lines.extend(_report_training_data(context))
    lines.extend(_report_validation_data(metrics, validation_outputs))
    lines.extend(_report_figures(context, validation_outputs))
    return lines


def _report_summary(
    context: WorkflowContext, metrics: dict[str, Any], report_yaml: Path
) -> list[str]:
    passed = metrics.get("passed")
    status = "not evaluated" if passed is None else ("passed" if passed else "failed")
    cfg = context.config
    return [
        "## Summary",
        "",
        f"- Structure: `{_sanitize(cfg.structure)}`",
        f"- Reference model: `{cfg.model}`",
        f"- Output directory: `{context.output_dir}`",
        f"- Validation status: **{status}**",
        f"- Machine-readable data: `{context.relative_path(report_yaml)}`",
        "",
        "## Configuration",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| DDB q-grid | `{tuple(cfg.ddb.qgrid)}` |",
        f"| Phonon supercell | `{tuple(cfg.ddb.phonon_supercell)}` |",
        f"| Elastic/internal strain | `{cfg.ddb.include_elastic}` / `{cfg.ddb.include_internal_strain}` |",
        f"| Metastable enabled/nmax | `{cfg.metastable.enabled}` / `{cfg.metastable.nmax}` |",
        f"| Train/test size | `{cfg.sampling.train_size}` / `{cfg.sampling.test_size}` |",
        f"| Training backend | `{cfg.training.backend}` |",
        f"| Validation force RMSE threshold | `{cfg.validation.force_rmse_threshold}` |",
        "",
    ]


def _report_procedure(context: WorkflowContext) -> list[str]:
    return [
        "## Procedure",
        "",
        "1. Prepare the parent structure and serialized workflow configuration.",
        "2. Build a finite-difference DDB with phonon, elastic, and internal-strain response blocks.",
        "3. Explore symmetry-mode metastable structures when enabled.",
        "4. Generate random and metastable-derived train/test frames with provenance.",
        "5. Evaluate train/test frames with the reference calculator and write trajectories plus `HIST.nc` files.",
        f"6. Fit the model with the `{context.config.training.backend}` training backend.",
        "7. Validate the fitted model on held-out data and collect metrics/figures.",
        "8. Write this report and the full YAML manifest.",
        "",
    ]


def _report_input_parameters(context: WorkflowContext) -> list[str]:
    rows = _flatten_report_values(context.manifest.get("config", {}))
    lines = [
        "## Input Parameters",
        "",
        "Full sanitized workflow inputs used to generate this report.",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    if not rows:
        lines.append("| none | none |")
    for key, value in rows:
        lines.append(
            f"| `{_escape_markdown_table_cell(key)}` | `{_escape_markdown_table_cell(_format_report_value(value))}` |"
        )
    lines.append("")
    return lines


def _report_stage_table(context: WorkflowContext) -> list[str]:
    lines = [
        "## Stage Results",
        "",
        "| Stage | Status | Key Outputs |",
        "|---|---|---|",
    ]
    for name in STAGE_ORDER:
        stage = context.manifest.get("stages", {}).get(name, {})
        status = stage.get("status", "pending")
        outputs = stage.get("outputs", {})
        key_outputs = ", ".join(_output_links(context, outputs)[:4]) or ""
        error = stage.get("error")
        if error:
            key_outputs = f"Error: {error}"
        lines.append(f"| `{name}` | `{status}` | {key_outputs} |")
    lines.append("")
    return lines


def _report_artifacts(context: WorkflowContext) -> list[str]:
    lines = ["## Artifact Index", "", "| Stage | Artifact |", "|---|---|"]
    found = False
    for name in STAGE_ORDER:
        outputs = context.manifest.get("stages", {}).get(name, {}).get("outputs", {})
        for link in _output_links(context, outputs):
            found = True
            lines.append(f"| `{name}` | {link} |")
    if not found:
        lines.append("| none | none |")
    lines.append("")
    return lines


def _report_ddb_data(context: WorkflowContext) -> list[str]:
    outputs = context.manifest.get("stages", {}).get("ddb", {}).get("outputs", {})
    harmonic_path = context.path_from_manifest(outputs.get("harmonic_validation"))
    lines = ["## DDB Harmonic Validation", ""]
    if outputs.get("ddb"):
        lines.append(f"- DDB: `{outputs['ddb']}`")
    if outputs.get("phonon_band_comparison"):
        lines.append(f"- Phonon-band comparison: `{outputs['phonon_band_comparison']}`")
    if outputs.get("harmonic_validation"):
        lines.append(f"- Harmonic validation JSON: `{outputs['harmonic_validation']}`")
    if harmonic_path is not None and harmonic_path.exists():
        data = json.loads(harmonic_path.read_text(encoding="utf-8"))
        components = data.get("model_components", {})
        elastic = data.get("elastic", {})
        internal = data.get("internal_strain", {})
        summary = data.get("summary", {})
        lines.extend(
            [
                f"- Model components: DDB-only `{components.get('ddb_only')}`; fitted XML `{components.get('xml_coefficients')}`",
                f"- Elastic constants present: `{elastic.get('present')}` (norm `{_format_report_value(elastic.get('norm'))}`)",
                f"- Internal-strain coupling present: `{internal.get('present')}` (norm `{_format_report_value(internal.get('norm'))}`)",
                "",
                "| Metric | Value |",
                "|---|---|",
            ]
        )
        for key, value in summary.items():
            lines.append(f"| `{key}` | `{_format_report_value(value)}` |")
    elif len(lines) == 2:
        lines.append("No DDB harmonic validation data were generated.")
    lines.append("")
    return lines


def _report_metastable_data(context: WorkflowContext) -> list[str]:
    outputs = (
        context.manifest.get("stages", {}).get("metastable", {}).get("outputs", {})
    )
    records = [
        record for record in outputs.get("records", []) if isinstance(record, dict)
    ]
    lines = ["## Metastable Data", ""]
    lines.append(f"- Candidate records: `{len(records)}`")
    if outputs.get("structures"):
        lines.append(f"- Structure trajectory: `{outputs['structures']}`")
    if outputs.get("trajectories"):
        lines.append(f"- Relaxation trajectories: `{outputs['trajectories']}`")
    successful = [record for record in records if record.get("status") == "success"]
    incomplete = [record for record in records if record.get("status") != "success"]
    lines.append(f"- Successful candidates: `{len(successful)}`")
    lines.append(f"- Failed/skipped candidates: `{len(incomplete)}`")
    if records:
        ordered = sorted(
            records,
            key=lambda item: (
                item.get("status") != "success",
                _format_report_value(item.get("id", "")),
            ),
        )
        lines.extend(
            [
                "",
                "### Metastable Candidate Index",
                "",
                "| ID | Status | Attempt | Amplitude | Label | Space Group | Delta E/FU (eV) | Max Force | Structure | Trajectory | Error |",
                "|---:|---|---:|---:|---|---|---:|---:|---|---|---|",
            ]
        )
        for record in ordered:
            lines.append(
                "| {id} | `{status}` | {attempt} | {amp} | {label} | {sg} | {energy} | {force} | `{structure}` | `{traj}` | {error} |".format(
                    id=record.get("id", ""),
                    status=_escape_markdown_table_cell(record.get("status", "")),
                    attempt=_format_report_value(record.get("attempt", "")),
                    amp=_format_report_value(record.get("amplitude", "")),
                    label=_escape_markdown_table_cell(record.get("combined_label", "")),
                    sg=_escape_markdown_table_cell(_format_spacegroup(record)),
                    energy=_format_report_value(record.get("delta_e_per_fu")),
                    force=_format_report_value(record.get("pre_relax_max_force", "")),
                    structure=record.get("structure_file", ""),
                    traj=record.get("trajectory_file", ""),
                    error=_escape_markdown_table_cell(
                        record.get("error_message", "")
                        or record.get("error_stage", "")
                        or ""
                    ),
                )
            )
        lines.extend(["", "### Metastable Candidate Details", ""])
        detail_keys = {
            "id",
            "status",
            "attempt",
            "amplitude",
            "pre_relax_max_force",
            "force_screen_reductions",
            "energy",
            "energy_per_fu",
            "delta_e_per_fu",
            "initial_spacegroup_number",
            "initial_spacegroup_name",
            "spacegroup_number",
            "spacegroup_name",
            "combined_label",
            "opd_label",
            "opd_is_maximal",
            "polarization_direction",
            "supercell_matrix",
            "n_atoms",
            "initial_structure_file",
            "relaxed_structure_file",
            "trajectory_file",
            "structure_file",
            "initial_atoms",
            "atoms",
            "is_still_imaginary",
            "relaxed_group_id",
            "relaxed_group_members",
            "relaxed_group_size",
            "relaxed_group_representative",
            "error_stage",
            "error_message",
            "source_modes",
        }
        for record in records:
            lines.extend(
                [
                    f"### Metastable Candidate {record.get('id', '')}",
                    "",
                    "| Field | Value |",
                    "|---|---|",
                ]
            )
            details = {
                key: value for key, value in record.items() if key in detail_keys
            }
            for key, value in _flatten_report_values(details):
                lines.append(
                    f"| `{_escape_markdown_table_cell(key)}` | `{_escape_markdown_table_cell(_format_report_value(value))}` |"
                )
            lines.append("")
    lines.append("")
    return lines


def _report_sampling_data(context: WorkflowContext) -> list[str]:
    outputs = context.manifest.get("stages", {}).get("sample", {}).get("outputs", {})
    return [
        "## Sampling Data",
        "",
        f"- Training frames: `{outputs.get('n_train', 0)}`",
        f"- Test frames: `{outputs.get('n_test', 0)}`",
        f"- Training trajectory: `{outputs.get('train_raw')}`",
        f"- Test trajectory: `{outputs.get('test_raw')}`",
        f"- Frame manifest: `{outputs.get('frame_manifest')}`",
        "",
    ]


def _report_training_data(context: WorkflowContext) -> list[str]:
    outputs = context.manifest.get("stages", {}).get("train", {}).get("outputs", {})
    diagnostics = outputs.get("diagnostics", {}) if isinstance(outputs, dict) else {}
    lines = ["## Training Data", ""]
    lines.append(f"- Backend: `{outputs.get('backend')}`")
    if outputs.get("model_config"):
        lines.append(f"- Model config: `{outputs['model_config']}`")
    for model_file in outputs.get("model_files", []):
        lines.append(f"- Model file: `{model_file}`")
    if isinstance(diagnostics, dict):
        for key in ("ncoeff", "nframes", "basis_xml"):
            if key in diagnostics:
                lines.append(
                    f"- {key}: `{_format_report_value(_relative_report_value(context, diagnostics[key]))}`"
                )
    lines.append("")
    return lines


def _report_validation_data(
    metrics: dict[str, Any], validation_outputs: dict[str, Any]
) -> list[str]:
    lines = ["## Validation Data", ""]
    if validation_outputs.get("metrics"):
        lines.append(f"- Metrics JSON: `{validation_outputs['metrics']}`")
    if validation_outputs.get("metastable_energy_differences"):
        lines.append(
            f"- Metastable energy differences: `{validation_outputs['metastable_energy_differences']}`"
        )
    if validation_outputs.get("metastable_energy_table"):
        lines.append(
            f"- Representative metastable energy table: `{validation_outputs['metastable_energy_table']}`"
        )
    rows = _flatten_report_scalars(metrics)
    if rows:
        lines.extend(["", "| Metric | Value |", "|---|---|"])
        for key, value in rows:
            lines.append(f"| `{key}` | `{_format_report_value(value)}` |")
    table_rows = _metastable_table_report_rows(metrics)
    if table_rows:
        lines.extend(
            [
                "",
                "### Representative Metastable Energies",
                "",
                "| ID | Space Group | Source Supercell | Validation Supercell | Group Size | Source dE/FU (eV) | Model dE/FU (eV) | Error (eV/FU) | Label |",
                "|---:|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in table_rows:
            lines.append(
                "| {id} | {sg} | `{source_sc}` | `{validation_sc}` | {size} | {src} | {model} | {err} | {label} |".format(
                    id=row.get("metastable_id", ""),
                    sg=row.get("spacegroup", ""),
                    source_sc=row.get("supercell_matrix", ""),
                    validation_sc=row.get("validation_supercell", ""),
                    size=row.get("group_size", ""),
                    src=_format_report_value(row.get("source_delta_e_per_fu_ev")),
                    model=_format_report_value(row.get("model_delta_e_per_fu_ev")),
                    err=_format_report_value(row.get("delta_error_e_per_fu_ev")),
                    label=row.get("combined_label", ""),
                )
            )
    lines.append("")
    return lines


def _report_figures(
    context: WorkflowContext, validation_outputs: dict[str, Any]
) -> list[str]:
    plots = validation_outputs.get("plots", {}) if validation_outputs else {}
    lines = ["## Figures", ""]
    ddb_outputs = context.manifest.get("stages", {}).get("ddb", {}).get("outputs", {})
    ddb_comparison = ddb_outputs.get("phonon_band_comparison")
    if not plots and not ddb_comparison:
        lines.append("No validation figures were generated.")
    if ddb_comparison:
        lines.extend(
            [
                "### DDB Vs Phonopy Phonon Bands",
                "",
                f"![ddb_vs_phonopy_phonon_bands]({ddb_comparison})",
                "",
            ]
        )
    for name, path in plots.items():
        lines.extend(
            [f"### {name.replace('_', ' ').title()}", "", f"![{name}]({path})", ""]
        )
    if lines[-1] != "":
        lines.append("")
    return lines


def _metastable_table_report_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    table = (
        metrics.get("metastable_energy_table") if isinstance(metrics, dict) else None
    )
    if not isinstance(table, dict):
        return []
    rows = table.get("states", [])
    return [row for row in rows if isinstance(row, dict)]


def _read_report_metrics(
    context: WorkflowContext, validation_outputs: dict[str, Any]
) -> dict[str, Any]:
    metrics_path = context.path_from_manifest(validation_outputs.get("metrics"))
    if metrics_path is None or not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _output_links(context: WorkflowContext, value: Any) -> list[str]:
    links = []
    seen = set()
    for link in _iter_output_links(context, value):
        if link not in seen:
            links.append(link)
            seen.add(link)
    return links


def _iter_output_links(context: WorkflowContext, value: Any) -> list[str]:
    links = []
    if isinstance(value, dict):
        for item in value.values():
            links.extend(_iter_output_links(context, item))
    elif isinstance(value, list):
        for item in value:
            links.extend(_iter_output_links(context, item))
    elif isinstance(value, str) and _looks_like_artifact(value):
        links.append(f"`{_relative_report_value(context, value)}`")
    return links


def _relative_report_value(context: WorkflowContext, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(context.output_dir.resolve()).as_posix()
    except ValueError:
        return value


def _relativize_manifest_paths(
    context: WorkflowContext, value: Any, *, _key: str = ""
) -> Any:
    if _key == "config":
        return value
    if isinstance(value, dict):
        return {
            key: _relativize_manifest_paths(context, item, _key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relativize_manifest_paths(context, item) for item in value]
    return _relative_report_value(context, value)


def _looks_like_artifact(value: str) -> bool:
    suffixes = (
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".traj",
        ".nc",
        ".ddb",
        ".xml",
        ".conf",
        ".png",
        ".vasp",
    )
    return "/" in value or value.lower().endswith(suffixes)


def _flatten_report_scalars(
    value: Any, prefix: str = ""
) -> list[tuple[str, str | int | float | bool | None]]:
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_report_scalars(item, child))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        rows.append((prefix, value))
    return rows


def _flatten_report_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_report_values(item, child))
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(item, dict) for item in value)
    ):
        rows.append((prefix, value))
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            rows.extend(_flatten_report_values(item, child))
    elif prefix:
        rows.append((prefix, value))
    return rows


def _format_report_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_sanitize(value), sort_keys=True)
    if value is None:
        return "null"
    return str(value)


def _escape_markdown_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def mlmbmodel_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate a MULTIBINIT-compatible model workflow."
    )
    parser.add_argument("structure", nargs="?", help="Parent structure file")
    parser.add_argument("--config", help="YAML workflow configuration")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--phonon-supercell", nargs=3, type=int, default=None)
    parser.add_argument("--qgrid", nargs=3, type=int, default=None)
    parser.add_argument("--include-elastic", action="store_true")
    parser.add_argument("--include-internal-strain", action="store_true")
    parser.add_argument(
        "--metastable", dest="metastable", action="store_true", default=None
    )
    parser.add_argument("--no-metastable", dest="metastable", action="store_false")
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--training-backend", choices=["python", "binary"], default=None
    )
    parser.add_argument("--ncoeff", type=int, default=None)
    parser.add_argument("--regularization", type=float, default=None)
    parser.add_argument("--force-rmse-threshold", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-stage", action="append", default=[])
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default=None)
    args = parser.parse_args()
    try:
        config = _config_from_cli(args)
        result = run_model_build_workflow(
            config,
            dry_run=args.dry_run,
            force_stage=args.force_stage,
            force_all=args.force_all,
            stop_after=args.stop_after,
        )
        if args.dry_run:
            for stage in result.manifest["stage_plan"]:
                print(stage["name"])
        else:
            print(f"Report: {result.report_md}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _default_stage_functions() -> dict[str, StageFunction]:
    return {
        "prepare": run_prepare_stage,
        "ddb": run_ddb_stage,
        "metastable": run_metastable_stage,
        "sample": run_sampling_stage,
        "evaluate": run_evaluate_stage,
        "train": run_training_stage,
        "validate": run_validate_stage,
        "report": run_report_stage,
    }


def _run_stage(
    context: WorkflowContext, name: str, func: StageFunction, force: bool = False
) -> StageMetadata:
    signature = _stage_signature(context.config, name)
    metadata_path = context.checkpoint_dir(name) / "stage.yaml"
    if not force and metadata_path.exists():
        previous = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        if (
            previous.get("status") == "completed"
            and previous.get("signature") == signature
        ):
            previous["status"] = "reused"
            return StageMetadata(
                **{
                    key: previous[key]
                    for key in StageMetadata.__dataclass_fields__
                    if key in previous
                }
            )
    metadata = StageMetadata(
        name=name, status="running", signature=signature, started_at=time.time()
    )
    try:
        outputs = func(context, metadata)
        if metadata.status == "running":
            metadata.status = "completed"
        metadata.outputs = _relativize_manifest_paths(context, _sanitize(outputs or {}))
    except Exception as exc:
        metadata.status = "failed"
        metadata.error = str(exc)
    metadata.completed_at = time.time()
    metadata.elapsed_s = metadata.completed_at - (
        metadata.started_at or metadata.completed_at
    )
    _write_yaml(metadata.to_dict(), metadata_path)
    return metadata


def _coerce_config(
    config: WorkflowConfig | None, kwargs: dict[str, Any]
) -> WorkflowConfig:
    if config is not None and kwargs:
        raise ValueError("Pass either WorkflowConfig or keyword arguments, not both")
    if config is not None:
        return config
    if "structure" not in kwargs:
        raise ValueError("Workflow requires a parent structure")
    return WorkflowConfig(**kwargs)


def _config_from_cli(args) -> WorkflowConfig:
    data: dict[str, Any] = {}
    if args.config:
        data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.structure is not None:
        data["structure"] = args.structure
    if "structure" not in data:
        raise ValueError("A parent structure is required")
    for key in ("model", "model_path", "output_dir", "seed"):
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    cfg = _dict_to_config(data)
    if args.phonon_supercell is not None:
        cfg.ddb.phonon_supercell = tuple(args.phonon_supercell)
    if args.qgrid is not None:
        cfg.ddb.qgrid = tuple(args.qgrid)
    if args.include_elastic:
        cfg.ddb.include_elastic = True
    if args.include_internal_strain:
        cfg.ddb.include_internal_strain = True
    if args.metastable is not None:
        cfg.metastable.enabled = args.metastable
    if args.nmax is not None:
        cfg.metastable.nmax = args.nmax
    if args.train_size is not None:
        cfg.sampling.train_size = args.train_size
    if args.test_size is not None:
        cfg.sampling.test_size = args.test_size
    if args.training_backend is not None:
        cfg.training.backend = args.training_backend
    if args.ncoeff is not None:
        cfg.training.ncoeff = args.ncoeff
    if args.regularization is not None:
        cfg.training.regularization = args.regularization
    if args.force_rmse_threshold is not None:
        cfg.validation.force_rmse_threshold = args.force_rmse_threshold
    return cfg


def _dict_to_config(data: dict[str, Any]) -> WorkflowConfig:
    data = dict(data)
    for key, cls in (
        ("ddb", DdbStageConfig),
        ("metastable", MetastableStageConfig),
        ("sampling", SamplingStageConfig),
        ("training", TrainingStageConfig),
        ("validation", ValidationStageConfig),
    ):
        if isinstance(data.get(key), dict):
            data[key] = cls(**data[key])
    return WorkflowConfig(**data)


def _load_parent_atoms(structure: str | Path | Atoms) -> Atoms:
    return structure.copy() if isinstance(structure, Atoms) else read(str(structure))


def _reference_calculator(context: WorkflowContext):
    if context.reference_calculator is None:
        context.reference_calculator = init_calc(
            context.config.model, model_path=context.config.model_path
        )
    return context.reference_calculator


def _sample_random_frames(context: WorkflowContext) -> list[SampledFrame]:
    rng = np.random.default_rng(context.config.seed)
    frames = []
    total = context.config.sampling.train_size + context.config.sampling.test_size
    amplitudes = _random_amplitudes(context.config.sampling.random_amplitude)
    for index in range(total):
        atoms = context.parent_atoms.copy()
        amplitude = amplitudes[index % len(amplitudes)]
        displacement = rng.normal(scale=amplitude, size=atoms.positions.shape)
        atoms.positions += displacement
        split = "train" if index < context.config.sampling.train_size else "test"
        frame_id = f"random-{index:04d}"
        atoms.info.update(
            {
                "source": "random",
                "frame_id": frame_id,
                "split": split,
                "seed": context.config.seed,
            }
        )
        frames.append(
            SampledFrame(
                atoms=atoms,
                split=split,
                source="random",
                frame_id=frame_id,
                metadata={
                    "seed": context.config.seed,
                    "amplitude": amplitude,
                },
            )
        )
    return frames


def _random_amplitudes(value: float | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("random_amplitude must be a number or a list of numbers")
    if np.isscalar(value):
        amplitudes = (float(value),)
    else:
        amplitudes = tuple(float(item) for item in value)
    if not amplitudes:
        raise ValueError("random_amplitude list must not be empty")
    if any(not np.isfinite(item) or item < 0.0 for item in amplitudes):
        raise ValueError("random_amplitude values must be finite and non-negative")
    return amplitudes


def _sample_metastable_frames(context: WorkflowContext) -> list[SampledFrame]:
    metastable_outputs = (
        context.manifest.get("stages", {}).get("metastable", {}).get("outputs", {})
    )
    structures = context.path_from_manifest(metastable_outputs.get("structures"))
    if structures is not None and structures.exists():
        states = _read_frames(structures)
    else:
        meta_dir = context.path_from_manifest(metastable_outputs.get("output_dir"))
        states = _metastable_atoms_from_records(
            metastable_outputs.get("records", []), meta_dir
        )
    frames = []
    trajectories = context.path_from_manifest(metastable_outputs.get("trajectories"))
    if trajectories is not None and trajectories.exists():
        for index, atoms in enumerate(_read_frames(trajectories)):
            if len(atoms) != len(context.parent_atoms):
                continue
            item = atoms.copy()
            frame_id = f"metastable-trajectory-{index:04d}"
            item.info.update(
                {
                    "source": "metastable_trajectory",
                    "frame_id": frame_id,
                    "split": "train",
                }
            )
            frames.append(
                SampledFrame(
                    atoms=item,
                    split="train",
                    source="metastable_trajectory",
                    frame_id=frame_id,
                    metadata={"trajectory_index": index},
                )
            )
    for index, atoms in enumerate(states):
        if len(atoms) != len(context.parent_atoms):
            continue
        split = "train" if index % 2 == 0 else "test"
        frame_id = f"metastable-{index:04d}"
        atoms.info.update(
            {
                "source": "metastable",
                "frame_id": frame_id,
                "split": split,
                "metastable_id": atoms.info.get("metastable_id", index),
            }
        )
        frames.append(
            SampledFrame(
                atoms=atoms,
                split=split,
                source="metastable",
                frame_id=frame_id,
                metadata={"metastable_id": atoms.info.get("metastable_id", index)},
            )
        )
    return frames


def _metastable_atoms_from_records(
    records, base_dir: Path | None = None
) -> list[Atoms]:
    atoms_list = []
    for record in records or []:
        atoms = (
            record.get("atoms")
            if isinstance(record, dict)
            else getattr(record, "atoms", None)
        )
        if (
            not isinstance(atoms, Atoms)
            and isinstance(record, dict)
            and base_dir is not None
        ):
            structure_file = record.get("structure_file") or record.get(
                "relaxed_structure_file"
            )
            if structure_file:
                structure_path = _confined_path(base_dir, structure_file)
                if structure_path is not None and structure_path.exists():
                    atoms = read(structure_path)
        if not isinstance(atoms, Atoms):
            continue
        item = atoms.copy()
        item.info.setdefault(
            "metastable_id",
            record.get("id")
            if isinstance(record, dict)
            else getattr(record, "id", None),
        )
        atoms_list.append(item)
    return atoms_list


def _training_ncell(context: WorkflowContext) -> tuple[int, int, int]:
    configured = context.config.training.options.get("ncell")
    if configured is not None:
        return tuple(int(item) for item in configured)
    if context.config.sampling.supercell is not None:
        return tuple(int(item) for item in context.config.sampling.supercell)
    return (1, 1, 1)


def _frames_for_training_supercell(
    context: WorkflowContext, frames: list[Atoms]
) -> list[Atoms]:
    target = context.config.sampling.supercell
    if target is None:
        return frames
    return [
        _evaluated_supercell(frame, tuple(int(item) for item in target))
        for frame in frames
    ]


def _evaluated_supercell(frame: Atoms, repeat: tuple[int, int, int]) -> Atoms:
    if tuple(repeat) == (1, 1, 1):
        return frame
    multiplier = int(np.prod(repeat))
    atoms = frame.repeat(repeat)
    atoms.info.update(frame.info)
    atoms.info["source_supercell"] = (1, 1, 1)
    atoms.info["training_supercell"] = tuple(repeat)
    results = {
        "energy": float(frame.get_potential_energy()) * multiplier,
        "forces": np.tile(np.asarray(frame.get_forces(), dtype=float), (multiplier, 1)),
    }
    try:
        results["stress"] = np.asarray(frame.get_stress(voigt=True), dtype=float)
    except Exception:
        pass
    atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def _validation_metastable_records(
    context: WorkflowContext,
) -> tuple[list[dict[str, Any]], Path | None]:
    configured = context.config.validation.metastable_records
    if configured:
        path = Path(configured)
        if not path.is_absolute() and not path.exists():
            path = context.output_dir / path
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        stages = data.get("stages", {}) if isinstance(data.get("stages"), dict) else {}
        stage = (
            stages.get("metastable", {})
            if isinstance(stages.get("metastable"), dict)
            else {}
        )
        outputs = (
            stage.get("outputs", {}) if isinstance(stage.get("outputs"), dict) else {}
        )
        records = outputs.get("records", data.get("records", []))
        output_dir = outputs.get("output_dir")
        base_dir = (
            _confined_path(path.parent, output_dir)
            if output_dir
            else path.parent.resolve()
        )
        return [
            record for record in records or [] if isinstance(record, dict)
        ], base_dir

    metastable_outputs = (
        context.manifest["stages"].get("metastable", {}).get("outputs", {})
    )
    records = metastable_outputs.get("records") or []
    output_dir = metastable_outputs.get("output_dir")
    base_dir = context.path_from_manifest(output_dir) if output_dir else None
    return [record for record in records or [] if isinstance(record, dict)], base_dir


def _validate_representative_metastable_energies(
    context: WorkflowContext,
    records: list[dict[str, Any]],
    base_dir: Path | None,
    model_calc,
    output: Path | None = None,
) -> dict[str, Any]:
    representatives = _representative_metastable_records(
        records, energy_tol=context.config.validation.metastable_energy_tol
    )
    target_supercell = _training_ncell(context)
    target_multiplier = int(np.prod(target_supercell))
    parent = context.parent_atoms.repeat(target_supercell)
    parent.calc = model_calc
    model_parent_energy = float(parent.get_potential_energy()) / target_multiplier
    rows = []
    skipped = []
    errors = []
    for record in representatives:
        atoms = _atoms_from_metastable_record(record, base_dir)
        if atoms is None:
            skipped.append(
                {"metastable_id": record.get("id"), "reason": "missing_structure"}
            )
            continue
        expanded = _metastable_atoms_in_target_supercell(
            context, atoms, record, target_supercell
        )
        if expanded is None:
            skipped.append(
                {
                    "metastable_id": record.get("id"),
                    "reason": "incompatible_supercell",
                    "n_atoms": len(atoms),
                    "target_supercell": target_supercell,
                    "supercell_matrix": record.get("supercell_matrix"),
                }
            )
            continue
        expanded.calc = model_calc
        model_energy_per_fu = float(expanded.get_potential_energy()) / target_multiplier
        model_delta = model_energy_per_fu - model_parent_energy
        source_delta = record.get("delta_e_per_fu")
        source_energy = record.get("energy_per_fu", record.get("energy"))
        error = None if source_delta is None else model_delta - float(source_delta)
        if error is not None:
            errors.append(error)
        rows.append(
            {
                "metastable_id": record.get("id"),
                "combined_label": record.get("combined_label"),
                "spacegroup_number": record.get("spacegroup_number"),
                "spacegroup_name": record.get("spacegroup_name"),
                "spacegroup": _format_spacegroup(record),
                "group_size": record.get("relaxed_group_size", 1),
                "group_members": record.get(
                    "relaxed_group_members", [record.get("id")]
                ),
                "source_energy_per_fu_ev": source_energy,
                "source_delta_e_per_fu_ev": source_delta,
                "model_energy_per_fu_ev": model_energy_per_fu,
                "model_delta_e_per_fu_ev": model_delta,
                "delta_error_e_per_fu_ev": error,
                "structure_file": record.get("structure_file")
                or record.get("relaxed_structure_file"),
                "supercell_matrix": record.get("supercell_matrix"),
                "validation_supercell": target_supercell,
            }
        )
    result = {
        "description": "Representative metastable states grouped by space group and source energy.",
        "energy_tolerance_ev_per_fu": context.config.validation.metastable_energy_tol,
        "nrecords": len(records),
        "nrepresentatives": len(rows),
        "nskipped": len(skipped),
        "model_parent_energy_ev_per_fu": model_parent_energy,
        "validation_supercell": target_supercell,
        "metric": _simple_metric(errors),
        "states": rows,
        "skipped": skipped,
    }
    if output is not None:
        _write_json(result, output)
    return result


def _representative_metastable_records(
    records: list[dict[str, Any]], energy_tol: float
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    sizes: dict[tuple[Any, Any, Any], int] = {}
    for record in records:
        if record.get("status") != "success":
            continue
        energy = record.get("energy_per_fu", record.get("energy"))
        sg = record.get("spacegroup_number")
        n_atoms = record.get("n_atoms")
        bucket = (
            None if energy is None else round(float(energy) / energy_tol) * energy_tol
        )
        key = (sg, bucket, n_atoms)
        sizes[key] = sizes.get(key, 0) + 1
        if key not in groups:
            groups[key] = record
    representatives = list(groups.values())
    for record in representatives:
        energy = record.get("energy_per_fu", record.get("energy"))
        bucket = (
            None if energy is None else round(float(energy) / energy_tol) * energy_tol
        )
        key = (record.get("spacegroup_number"), bucket, record.get("n_atoms"))
        record.setdefault("relaxed_group_size", sizes.get(key, 1))
    representatives.sort(key=lambda item: item.get("delta_e_per_fu", float("inf")))
    return representatives


def _atoms_from_metastable_record(
    record: dict[str, Any], base_dir: Path | None
) -> Atoms | None:
    atoms = record.get("atoms")
    if isinstance(atoms, Atoms):
        return atoms.copy()
    structure_file = record.get("structure_file") or record.get(
        "relaxed_structure_file"
    )
    if structure_file and base_dir is not None:
        structure_path = _confined_path(base_dir, structure_file)
        if structure_path is None:
            return None
        if structure_path.exists():
            item = read(structure_path)
            item.info.update(
                {
                    "metastable_id": record.get("id"),
                    "combined_label": record.get("combined_label"),
                    "spacegroup_number": record.get("spacegroup_number"),
                    "spacegroup_name": record.get("spacegroup_name"),
                }
            )
            return item
    return None


def _confined_path(base_dir: Path, relative_path: Any) -> Path | None:
    base = Path(base_dir).resolve()
    candidate = Path(str(relative_path))
    if candidate.is_absolute():
        return None
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved


def _metastable_atoms_in_target_supercell(
    context: WorkflowContext,
    atoms: Atoms,
    record: dict[str, Any],
    target: tuple[int, int, int],
) -> Atoms | None:
    source = _record_supercell_diagonal(record, len(atoms), len(context.parent_atoms))
    if source is None:
        return None
    if any(src <= 0 or dst % src != 0 for src, dst in zip(source, target)):
        return None
    repeat = tuple(dst // src for src, dst in zip(source, target))
    expanded = atoms.repeat(repeat) if repeat != (1, 1, 1) else atoms.copy()
    expected_natoms = len(context.parent_atoms) * int(np.prod(target))
    if len(expanded) != expected_natoms:
        return None
    expanded = _reorder_like_reference_supercell(
        expanded, context.parent_atoms.repeat(target)
    )
    expanded.info.update(atoms.info)
    expanded.info["source_supercell"] = source
    expanded.info["validation_supercell"] = target
    return expanded


def _reorder_like_reference_supercell(atoms: Atoms, reference: Atoms) -> Atoms:
    if len(atoms) != len(reference):
        raise ValueError("Cannot reorder structures with different atom counts")
    scaled = atoms.get_scaled_positions(wrap=True)
    ref_scaled = reference.get_scaled_positions(wrap=True)
    symbols = atoms.get_chemical_symbols()
    ref_symbols = reference.get_chemical_symbols()
    unused = set(range(len(atoms)))
    order = []
    for ref_symbol, ref_pos in zip(ref_symbols, ref_scaled):
        candidates = [index for index in unused if symbols[index] == ref_symbol]
        if not candidates:
            raise ValueError(f"No unmatched atom with symbol {ref_symbol!r}")
        distances = []
        for index in candidates:
            delta = scaled[index] - ref_pos
            delta -= np.round(delta)
            distances.append((float(np.dot(delta, delta)), index))
        _, selected = min(distances, key=lambda item: item[0])
        order.append(selected)
        unused.remove(selected)
    reordered = Atoms(
        symbols=ref_symbols,
        positions=atoms.positions[order],
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    reordered.info.update(atoms.info)
    return reordered


def _record_supercell_diagonal(
    record: dict[str, Any], n_atoms: int, parent_natoms: int
) -> tuple[int, int, int] | None:
    matrix = record.get("supercell_matrix")
    if matrix is not None:
        array = np.asarray(matrix, dtype=int)
        if array.shape == (3, 3) and np.all(array == np.diag(np.diag(array))):
            return tuple(int(abs(item)) for item in np.diag(array))
    if n_atoms == parent_natoms:
        return (1, 1, 1)
    return None


def _format_spacegroup(record: dict[str, Any]) -> str:
    name = record.get("spacegroup_name") or ""
    number = record.get("spacegroup_number")
    return f"{name} ({number})" if number is not None else str(name)


def _simple_metric(values) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {"count": 0, "mae": 0.0, "rmse": 0.0, "max_abs": 0.0}
    return {
        "count": int(array.size),
        "mae": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "max_abs": float(np.max(np.abs(array))),
    }


def _metastable_trajectory_atoms(meta_dir: Path, records) -> list[Atoms]:
    atoms_list = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        traj_file = record.get("trajectory_file")
        if not traj_file:
            continue
        traj_path = meta_dir / str(traj_file)
        if not traj_path.exists():
            continue
        for step_index, atoms in enumerate(_read_frames(traj_path)):
            item = atoms.copy()
            item.info.update(
                {
                    "metastable_id": record.get("id"),
                    "metastable_trajectory_file": str(traj_file),
                    "metastable_trajectory_step": step_index,
                }
            )
            atoms_list.append(item)
    return atoms_list


def _run_registered_sampler(
    context: WorkflowContext, source: dict[str, Any], source_index: int = 0
) -> list[SampledFrame]:
    from atomchain.training_set_generation import (
        SamplingContext,
        generate_frames_from_source,
    )

    name = str(source.get("name", "")).strip().lower().replace("-", "_")
    calculator = (
        _reference_calculator(context)
        if name in {"contour", "contour_exploration", "npt", "npt_md"}
        else None
    )
    metastable_outputs = (
        context.manifest.get("stages", {}).get("metastable", {}).get("outputs", {})
    )
    structures = context.path_from_manifest(metastable_outputs.get("structures"))
    if structures is not None and structures.exists():
        metastable_structures = _read_frames(structures)
        metastable_records = []
    else:
        metastable_structures = []
        metastable_records = metastable_outputs.get("records", [])
    sampler_dir = context.stage_dir("sample") / f"{source_index:02d}_{name or 'source'}"
    sampler_dir.mkdir(parents=True, exist_ok=True)
    source_payload = dict(source)
    generated = generate_frames_from_source(
        SamplingContext(
            parent_atoms=context.parent_atoms,
            seed=context.config.seed,
            calculator=calculator,
            output_dir=sampler_dir,
            metastable_records=metastable_records,
            metastable_structures=metastable_structures,
        ),
        source_payload,
    )
    frames = []
    default_split = source.get("split", "train")
    for item in generated:
        split = item.split or default_split
        if split not in {"train", "test"}:
            raise ValueError(
                f"Sampler source {name!r} produced invalid split: {split!r}"
            )
        atoms = item.atoms.copy()
        frame_id = f"source-{source_index:02d}-{item.frame_id}"
        atoms.info.update(
            {
                "source": item.source,
                "frame_id": frame_id,
                "split": split,
                "source_index": source_index,
            }
        )
        atoms.info.update(_sanitize(item.metadata))
        metadata = dict(_sanitize(item.metadata))
        metadata["source_index"] = source_index
        frames.append(
            SampledFrame(
                atoms=atoms,
                split=split,
                source=item.source,
                frame_id=frame_id,
                metadata=metadata,
            )
        )
    return frames


def _read_frames(path: Path) -> list[Atoms]:
    loaded = read(path, ":")
    return [loaded] if isinstance(loaded, Atoms) else list(loaded)


def _model_calculator_from_training(context: WorkflowContext):
    train_outputs = (
        context.manifest.get("stages", {}).get("train", {}).get("outputs", {})
    )
    if train_outputs.get("backend") == "python":
        ddb = context.path_from_manifest(train_outputs.get("ddb"))
        coeff_file = context.path_from_manifest(train_outputs.get("coeff_file"))
        if ddb is not None and coeff_file is not None:
            from pymultibinit.calculator import MultibinitCalculator
            from pymultibinit.potential import MultibinitPotential

            potential = MultibinitPotential.from_pyeffpot(
                str(ddb),
                xml_file=str(coeff_file),
                ncell=_training_ncell(context),
                dipdip=_bool_option(context.config.training.options, "dipdip", True),
                asr=_bool_option(context.config.training.options, "asr", False),
                auto_match_atoms=False,
                reference_stress_ha_bohr3=context.reference_stress_ha_bohr3,
            )
            return MultibinitCalculator(potential=potential)
    model_config = train_outputs.get("model_config")
    if model_config:
        return init_calc(
            "multibinit", model_path=str(context.path_from_manifest(model_config))
        )
    model_files = train_outputs.get("model_files", [])
    if not model_files:
        raise ValueError("Training did not produce model files for validation")
    model_path = context.path_from_manifest(model_files[0])
    return init_calc("multibinit", model_path=str(model_path))


def _write_multibinit_model_config(
    context: WorkflowContext, model_dir: Path, *, ddb: Path, coeff_file: Path
) -> Path:
    options = context.config.training.options
    ncell = _training_ncell(context)
    ngqpt = tuple(context.config.ddb.qgrid)
    config_path = model_dir / "config.conf"
    lines = [
        f"ddb_file: {_config_relative_path(config_path, ddb)}",
        f"coeff_file: {_config_relative_path(config_path, coeff_file)}",
        f"ncell: {ncell[0]} {ncell[1]} {ncell[2]}",
        f"ngqpt: {ngqpt[0]} {ngqpt[1]} {ngqpt[2]}",
        f"dipdip: {int(options.get('dipdip', 1))}",
        f"asr: {int(options.get('asr', 0))}",
    ]
    if options.get("sys_file") is not None:
        lines.insert(1, f"sys_file: {options['sys_file']}")
    if options.get("lib_path") is not None:
        lines.append(f"lib_path: {options['lib_path']}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _config_relative_path(config_path: Path, target: Path) -> str:
    target = Path(target)
    if not target.is_absolute():
        target = target.resolve()
    return os.path.relpath(target, config_path.parent.resolve()).replace(os.sep, "/")


def _python_fit_config(context: WorkflowContext):
    from pymultibinit.training import PythonFitConfig

    options = context.config.training.options
    selection = options.get(
        "selection", "greedy" if context.config.training.ncoeff is not None else "all"
    )
    include_pure_strain, max_strain_power = _pure_strain_options(options)
    return PythonFitConfig(
        ncell=_training_ncell(context),
        fit_on=tuple(options.get("fit_on", (True, True, True))),
        fit_factors=tuple(options.get("fit_factors", (1.0, 1.0, 1.0))),
        regularization=context.config.training.regularization,
        selection=selection,
        ncoeff=context.config.training.ncoeff,
        cutoff=options.get("cutoff"),
        power_range=tuple(options.get("power_range", (3, 4))),
        feature_backend=options.get("feature_backend", "auto"),
        feature_memmap_dir=options.get("feature_memmap_dir"),
        candidate_pool_size=options.get("candidate_pool_size"),
        feature_chunk_size=int(options.get("feature_chunk_size", 512)),
        screening_frame_count=options.get("screening_frame_count"),
        min_pure_strain_ratio=_min_pure_strain_ratio(options, selection),
        include_pure_strain=include_pure_strain,
        max_strain_power=max_strain_power,
    )


def _pure_strain_options(options: Mapping[str, Any]) -> tuple[bool, int]:
    return (
        _bool_option(options, "include_pure_strain", True),
        int(options.get("max_strain_power", 4)),
    )


def _min_pure_strain_ratio(options: Mapping[str, Any], selection: str) -> float:
    if "min_pure_strain_ratio" in options:
        return float(options["min_pure_strain_ratio"])
    return 0.05 if selection in ("greedy", "screened_greedy") else 0.0


def _basis_xml(
    context: WorkflowContext, model_dir: Path, fit_config: Any | None = None
) -> Path:
    if context.config.training.basis_xml is not None:
        path = Path(context.config.training.basis_xml)
        return path if path.is_absolute() else context.output_dir / path

    from pymultibinit.training import (
        generate_fortran_anchored_basis,
        with_fortran_text_labels,
        write_basis_netcdf,
    )

    options = context.config.training.options
    cutoff = _basis_cutoff(context)
    basis_ncell = _basis_ncell(context)
    include_pure_strain, max_strain_power = (
        _pure_strain_options(options)
        if fit_config is None
        else (fit_config.include_pure_strain, fit_config.max_strain_power)
    )
    fingerprint = _basis_request_fingerprint(
        context,
        cutoff,
        basis_ncell,
        options,
        pure_strain_options=(include_pure_strain, max_strain_power),
    )
    diagnostics_path = model_dir / "basis_pair_diagnostics.json"
    basis_path = model_dir / "basis.nc"
    legacy_xml = model_dir / "basis.xml"
    for candidate in (basis_path, legacy_xml):
        if candidate.exists() and diagnostics_path.exists():
            try:
                previous = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            except Exception:
                previous = None
            if (
                isinstance(previous, dict)
                and previous.get("basis_request_fingerprint") == fingerprint
            ):
                return candidate

    symrel, tnons, _atom_mappings = _basis_symmetry(context)
    atoms = context.parent_atoms
    include_strain_coupling = _bool_option(options, "include_strain_coupling", False)
    max_nbody = options.get("max_nbody")
    if max_nbody is not None:
        max_nbody = int(max_nbody)
    basis = generate_fortran_anchored_basis(
        xcart=atoms.get_positions(),
        xred=atoms.get_scaled_positions(wrap=True),
        cutoff=cutoff,
        symrel=symrel,
        ncell=basis_ncell,
        rprimd=atoms.cell.array,
        tnons=tnons,
        power_range=tuple(options.get("power_range", (3, 4))),
        include_strain_coupling=include_strain_coupling,
        include_pure_strain=include_pure_strain,
        max_strain_power=max_strain_power,
        max_nbody=max_nbody,
    )
    basis = with_fortran_text_labels(basis, atoms.get_chemical_symbols())
    write_basis_netcdf(
        basis_path,
        basis,
        generator_version=FORTRAN_ANCHORED_GENERATOR_TAG,
        basis_fingerprint=fingerprint,
    )
    if legacy_xml.exists():
        legacy_xml.unlink()
    diagnostics = {
        "basis_request_fingerprint": fingerprint,
        "ncell": [1, 1, 1],
        "cutoff": cutoff,
        "generator_version": FORTRAN_ANCHORED_GENERATOR_TAG,
        "ncoeff": len(basis),
    }
    _write_json(diagnostics, diagnostics_path)
    return basis_path


def _basis_pair_diagnostics_satisfy_request(
    previous: Any,
    cutoff: float,
    basis_ncell: tuple[int, int, int],
    options: Mapping[str, Any],
    fingerprint: str,
) -> bool:
    if not isinstance(previous, dict):
        return False
    if previous.get("basis_request_fingerprint") != fingerprint:
        return False
    if previous.get("ncell") != list(basis_ncell):
        return False
    if previous.get("cutoff") != cutoff:
        return False
    if bool(options.get("require_basis_symmetry_closed", True)) and not previous.get(
        "symmetry_closed", False
    ):
        return False
    if (
        bool(options.get("use_symmetry", True))
        and int(previous.get("n_symmetry_operations", 0)) <= 0
    ):
        return False
    if int(previous.get("missing_mapped_factors_count", 0)) != 0:
        return False
    return True


def _basis_request_fingerprint(
    context: WorkflowContext,
    cutoff: float,
    basis_ncell: tuple[int, int, int],
    options: Mapping[str, Any],
    pure_strain_options: tuple[bool, int] | None = None,
) -> str:
    atoms = context.parent_atoms
    include_pure_strain, max_strain_power = (
        _pure_strain_options(options)
        if pure_strain_options is None
        else pure_strain_options
    )
    payload = {
        "generator": FORTRAN_ANCHORED_GENERATOR_TAG,
        "numbers": [int(item) for item in atoms.get_atomic_numbers()],
        "cell": _rounded_nested(atoms.cell.array),
        "positions": _rounded_nested(atoms.get_positions()),
        "pbc": [bool(item) for item in atoms.get_pbc()],
        "cutoff": float(cutoff),
        "basis_ncell": list(basis_ncell),
        "power_range": [int(item) for item in options.get("power_range", (3, 4))],
        "use_symmetry": bool(options.get("use_symmetry", True)),
        "symprec": float(options.get("symprec", 1e-5)),
        "include_strain_coupling": bool(options.get("include_strain_coupling", False)),
        "include_pure_strain": bool(include_pure_strain),
        "max_strain_power": int(max_strain_power),
        "max_nbody": options.get("max_nbody"),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rounded_nested(array: Any) -> Any:
    return np.round(np.asarray(array, dtype=float), decimals=12).tolist()


def _basis_pair_diagnostics_match(previous: Any, current: dict[str, Any]) -> bool:
    if not isinstance(previous, dict):
        return False
    keys = (
        "ncell",
        "cutoff",
        "n_factors",
        "n_symmetry_operations",
        "symmetry_closed",
        "missing_mapped_factors_count",
        "max_pair_distance",
        "basis_request_fingerprint",
    )
    for key in keys:
        if previous.get(key) != current.get(key):
            return False
    return True


def _basis_cutoff(context: WorkflowContext) -> float:
    options = context.config.training.options
    if options.get("cutoff") is not None:
        return float(options["cutoff"])
    if options.get("cutoff_mode") == "cell_face_diagonal":
        lengths = sorted(float(item) for item in context.parent_atoms.cell.lengths())
        return float(
            np.sqrt(lengths[-1] ** 2 + lengths[-2] ** 2)
            + float(options.get("cutoff_eps", 1e-8))
        )
    if options.get("cutoff_mode") == "cell_body_diagonal":
        return float(
            np.linalg.norm(context.parent_atoms.cell.lengths())
            + float(options.get("cutoff_eps", 1e-8))
        )
    return float(max(context.parent_atoms.cell.lengths()))


def _basis_symmetry(context: WorkflowContext):
    options = context.config.training.options
    if not bool(options.get("use_symmetry", True)):
        return None, None, None
    try:
        from pymultibinit.pyeffpot.symmetry import (
            build_atom_mapping,
            get_symmetry_from_crystal,
        )
    except Exception:
        return None, None, None
    numbers = np.asarray(context.parent_atoms.get_atomic_numbers(), dtype=int)
    symrel, tnons = get_symmetry_from_crystal(
        context.parent_atoms.cell.array,
        context.parent_atoms.get_scaled_positions(wrap=True),
        numbers,
        symprec=float(options.get("symprec", 1e-5)),
    )
    atom_mappings = build_atom_mapping(
        context.parent_atoms.get_scaled_positions(wrap=True),
        symrel,
        tnons,
        tol=float(options.get("symprec", 1e-5)),
    )
    return symrel, tnons, atom_mappings


def _python_fit_kwargs(
    context: WorkflowContext, train_hist: Path | None = None
) -> dict[str, Any]:
    supported = {"fixed_model", "weights", "validation_hist"}
    kwargs = {
        key: value
        for key, value in context.config.training.options.items()
        if key in supported
    }
    spec = kwargs.get("weights")
    if isinstance(spec, Mapping):
        if train_hist is None:
            raise ValueError("training.options.weights spec requires the train HIST")
        kwargs["weights"] = _frame_weights_from_spec(spec, train_hist)
    return kwargs


def _frame_weights_from_spec(spec: Mapping[str, Any], train_hist: Path) -> list[float]:
    """Per-frame fit weights from a declarative spec.

    mode "force_rms_boltzmann": w = exp(-F_rms^2 / (2 sigma^2)) with F_rms the
    RMS reference force of the frame in eV/Ang. Emphasizes low-force (basin and
    saddle) configurations while keeping high-force frames at reduced weight for
    harmonic identifiability.
    """
    from pymultibinit.training import read_hist_frames

    mode = spec.get("mode", "force_rms_boltzmann")
    if mode != "force_rms_boltzmann":
        raise ValueError(f"unsupported weights mode: {mode!r}")
    sigma_ev_ang = float(spec["sigma_ev_ang"])
    if sigma_ev_ang <= 0.0:
        raise ValueError("weights sigma_ev_ang must be positive")
    floor = float(spec.get("floor", 0.0))
    if not 0.0 <= floor <= 1.0:
        raise ValueError("weights floor must lie in [0, 1]")
    ha_bohr_to_ev_ang = 51.422067
    frames = read_hist_frames(str(train_hist))
    weights = []
    for frame in frames:
        forces = np.asarray(frame.forces, dtype=float)
        f_rms_ev_ang = float(np.sqrt(np.mean(forces**2))) * ha_bohr_to_ev_ang
        weight = math.exp(-0.5 * (f_rms_ev_ang / sigma_ev_ang) ** 2)
        weights.append(max(weight, floor))
    return weights


def _basis_ncell(context: WorkflowContext) -> tuple[int, int, int]:
    """Return the primitive-cell basis size.

    ``training.options.basis_ncell`` is a legacy, dead option: anchored basis
    generation always enumerates the primitive cell, so it is intentionally
    ignored to keep generation, diagnostics, and cache fingerprints consistent.
    """
    return (1, 1, 1)


def _python_fixed_model(context: WorkflowContext, ddb: Path):
    from pymultibinit.pyeffpot.potential import EffectivePotential

    options = context.config.training.options
    return EffectivePotential.from_files(
        str(ddb),
        xml_file=None,
        ncell=_training_ncell(context),
        dipdip=_bool_option(options, "dipdip", True),
        asr=_bool_option(options, "asr", False),
        reference_stress=context.reference_stress_ha_bohr3,
    )


def _binary_training_kwargs(context: WorkflowContext) -> dict[str, Any]:
    supported = {"extra_args", "timeout", "env"}
    return {
        key: value
        for key, value in context.config.training.options.items()
        if key in supported
    }


def _write_validation_plots(
    reference_traj: Path,
    model_traj: Path,
    metrics: dict[str, Any],
    validation_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_frames = _read_frames(reference_traj)
    model_frames = _read_frames(model_traj)
    data = _validation_plot_data(reference_frames, model_frames, metrics)
    plots = {}
    _write_parity_plot(
        data["reference_energy"],
        data["model_energy"],
        "Reference energy (eV)",
        "Model energy (eV)",
        validation_dir / "energy_parity.png",
        plt,
    )
    plots["energy_parity"] = "validation/energy_parity.png"
    _write_parity_plot(
        data["reference_forces"],
        data["model_forces"],
        "Reference force (eV/Angstrom)",
        "Model force (eV/Angstrom)",
        validation_dir / "force_parity.png",
        plt,
    )
    plots["force_parity"] = "validation/force_parity.png"
    if data["reference_stress"].size and data["model_stress"].size:
        _write_parity_plot(
            data["reference_stress"],
            data["model_stress"],
            "Reference stress (eV/Angstrom^3)",
            "Model stress (eV/Angstrom^3)",
            validation_dir / "stress_parity.png",
            plt,
        )
        plots["stress_parity"] = "validation/stress_parity.png"
    _write_frame_error_plot(
        metrics.get("frame_errors", []), validation_dir / "frame_error_trends.png", plt
    )
    plots["frame_error_trends"] = "validation/frame_error_trends.png"
    _write_rmse_summary_plot(
        metrics.get("metrics", {}), validation_dir / "rmse_summary.png", plt
    )
    plots["rmse_summary"] = "validation/rmse_summary.png"
    return plots


def _validation_plot_data(reference_frames, model_frames, metrics):
    reference_energy = []
    model_energy = []
    reference_forces = []
    model_forces = []
    reference_stress = []
    model_stress = []
    for ref, model in zip(reference_frames, model_frames):
        reference_energy.append(float(ref.get_potential_energy()))
        model_energy.append(float(model.get_potential_energy()))
        reference_forces.append(np.asarray(ref.get_forces(), dtype=float).reshape(-1))
        model_forces.append(np.asarray(model.get_forces(), dtype=float).reshape(-1))
        try:
            reference_stress.append(
                np.asarray(ref.get_stress(voigt=True), dtype=float).reshape(-1)
            )
            model_stress.append(
                np.asarray(model.get_stress(voigt=True), dtype=float).reshape(-1)
            )
        except Exception:
            pass
    return {
        "reference_energy": np.asarray(reference_energy, dtype=float),
        "model_energy": np.asarray(model_energy, dtype=float),
        "reference_forces": np.concatenate(reference_forces)
        if reference_forces
        else np.array([]),
        "model_forces": np.concatenate(model_forces) if model_forces else np.array([]),
        "reference_stress": np.concatenate(reference_stress)
        if reference_stress
        else np.array([]),
        "model_stress": np.concatenate(model_stress) if model_stress else np.array([]),
    }


def _write_parity_plot(x, y, xlabel: str, ylabel: str, path: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    ax.scatter(x, y, s=18, alpha=0.75)
    if x.size and y.size:
        low = float(min(np.min(x), np.min(y)))
        high = float(max(np.max(x), np.max(y)))
        if low == high:
            low -= 1.0
            high += 1.0
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Parity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_frame_error_plot(
    frame_errors: list[dict[str, Any]], path: Path, plt
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    indices = [
        item.get("frame_index", index) for index, item in enumerate(frame_errors)
    ]
    energy = [abs(item.get("energy_error_ev", 0.0)) for item in frame_errors]
    force = [item.get("max_force_error_ev_ang", 0.0) for item in frame_errors]
    if indices:
        ax.plot(indices, energy, "o-", label="|energy error| (eV)")
        ax.plot(indices, force, "s-", label="max force error (eV/Angstrom)")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Error")
    ax.set_title("Frame Error Trends")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_rmse_summary_plot(metric_data: dict[str, Any], path: Path, plt) -> None:
    labels = []
    values = []
    for name, item in metric_data.items():
        if isinstance(item, dict) and "rmse" in item:
            labels.append(name.replace("_", "\n"))
            values.append(float(item["rmse"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    if labels:
        ax.bar(labels, values, color="#4c78a8")
    ax.set_ylabel("RMSE")
    ax.set_title("Validation RMSE Summary")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_phonon_band_comparison(
    context: WorkflowContext, ddb_path: Path, ddb_dir: Path
) -> dict[str, Any]:
    phonopy_yaml = ddb_dir / "fd_cache" / "phonon_base" / "phonopy_params.yaml"
    if not phonopy_yaml.exists():
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ase.cell import Cell
        from phonopy import load

        from atomchain.phonon.plotphonopy import group_band_path

        phonon = load(phonopy_yaml=str(phonopy_yaml))
        phonon.symmetrize_force_constants()
        bandpath = Cell(phonon.primitive.cell).bandpath(npoints=51)
        xlist, kptlist, tick_positions, tick_labels = group_band_path(bandpath)
        qpoints = np.concatenate(kptlist)
        distances = np.concatenate(xlist)
        tick_labels = ["Gamma" if label == "G" else label for label in tick_labels]
        phonopy_freqs = _phonopy_band_frequencies_cm1(phonon, qpoints)
        ddb_freqs = _ddb_band_frequencies_cm1(
            ddb_path,
            qpoints,
            dipdip=_bool_option(context.config.training.options, "dipdip", True),
            asr=_bool_option(context.config.training.options, "asr", False),
        )
        figure = ddb_dir / "phonon_band_comparison.png"
        data_path = ddb_dir / "phonon_band_comparison.json"
        _plot_band_comparison(
            xlist=xlist,
            segment_lengths=[len(item) for item in kptlist],
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            phonopy_freqs=phonopy_freqs,
            ddb_freqs=ddb_freqs,
            path=figure,
            plt=plt,
        )
        _write_json(
            {
                "units": "cm-1",
                "qpoints": qpoints.tolist(),
                "distances": distances.tolist(),
                "tick_positions": np.asarray(tick_positions, dtype=float).tolist(),
                "tick_labels": tick_labels,
                "phonopy_frequencies": phonopy_freqs.tolist(),
                "ddb_frequencies": ddb_freqs.tolist(),
                "absolute_difference_summary": _frequency_difference_summary(
                    ddb_freqs, phonopy_freqs
                ),
            },
            data_path,
        )
        return {
            "figure": context.relative_path(figure),
            "data": context.relative_path(data_path),
        }
    except Exception as exc:
        return {"warning": f"Could not write DDB-vs-phonopy phonon bands: {exc}"}


def _write_ddb_harmonic_validation(
    context: WorkflowContext, ddb_path: Path, ddb_dir: Path, source_calc
) -> dict[str, Any]:
    if not context.config.ddb.validate_harmonic:
        return {}
    try:
        from pymultibinit.calculator import MultibinitCalculator
        from pymultibinit.potential import MultibinitPotential
        from pymultibinit.pyeffpot.ddb_parser_complete import read_ddb

        unitcell = read_ddb(str(ddb_path))
        elastic_norm = _safe_norm(getattr(unitcell, "elastic_constants", None))
        internal_norm = _safe_norm(getattr(unitcell, "strain_coupling", None))
        potential = MultibinitPotential.from_pyeffpot(
            str(ddb_path),
            ncell=(1, 1, 1),
            dipdip=_bool_option(context.config.training.options, "dipdip", True),
            asr=_bool_option(context.config.training.options, "asr", False),
            auto_match_atoms=False,
            reference_stress_ha_bohr3=context.reference_stress_ha_bohr3,
        )
        ddb_calc = MultibinitCalculator(potential=potential)
        frames = _tiny_harmonic_validation_frames(context)
        source_ref = context.parent_atoms.copy()
        source_ref.calc = source_calc
        ddb_ref = context.parent_atoms.copy()
        ddb_ref.calc = ddb_calc
        source_ref_values = _calculator_values(source_ref)
        ddb_ref_values = _calculator_values(ddb_ref)
        records = []
        force_errors = []
        stress_errors = []
        energy_errors = []
        for index, atoms in enumerate(frames):
            source_atoms = atoms.copy()
            source_atoms.calc = source_calc
            ddb_atoms = atoms.copy()
            ddb_atoms.calc = ddb_calc
            source_values = _calculator_values(source_atoms)
            ddb_values = _calculator_values(ddb_atoms)
            source_delta = _calculator_delta(source_values, source_ref_values)
            ddb_delta = _calculator_delta(ddb_values, ddb_ref_values)
            force_error = (ddb_delta["forces"] - source_delta["forces"]).reshape(-1)
            stress_error = (ddb_delta["stress"] - source_delta["stress"]).reshape(-1)
            source_response_energy = _harmonic_response_energy(
                source_delta, source_ref_values, atoms
            )
            ddb_response_energy = _harmonic_response_energy(
                ddb_delta, ddb_ref_values, atoms
            )
            energy_error = ddb_response_energy - source_response_energy
            force_errors.append(force_error)
            stress_errors.append(stress_error)
            energy_errors.append(energy_error)
            records.append(
                {
                    "frame_index": index,
                    "source_harmonic_response_energy_ev": source_response_energy,
                    "ddb_harmonic_response_energy_ev": ddb_response_energy,
                    "harmonic_response_energy_error_ev": energy_error,
                    "force_rmse_ev_ang": _rmse(force_error),
                    "force_max_abs_ev_ang": _max_abs(force_error),
                    "stress_rmse_ev_ang3": _rmse(stress_error),
                    "stress_max_abs_ev_ang3": _max_abs(stress_error),
                }
            )
        metrics_path = ddb_dir / "harmonic_validation.json"
        data = {
            "description": "Tiny-displacement/strain validation of DDB-only pyeffpot against source calculator",
            "model_components": {
                "ddb_only": True,
                "ddb": context.relative_path(ddb_path),
                "xml_coefficients": None,
                "fitted_model": None,
            },
            "n_frames": len(frames),
            "displacement_amplitude_ang": context.config.ddb.harmonic_displacement_amplitude,
            "strain_amplitude": context.config.ddb.harmonic_strain_amplitude,
            "dipdip": _bool_option(context.config.training.options, "dipdip", True),
            "asr": _bool_option(context.config.training.options, "asr", False),
            "elastic": {
                "requested": bool(context.config.ddb.include_elastic),
                "present": elastic_norm > 0.0,
                "norm": elastic_norm,
            },
            "internal_strain": {
                "requested": bool(context.config.ddb.include_internal_strain),
                "present": internal_norm > 0.0,
                "norm": internal_norm,
            },
            "summary": {
                "harmonic_response_energy_error_rmse_ev": _rmse(energy_errors),
                "harmonic_response_energy_error_max_abs_ev": _max_abs(energy_errors),
                "force_rmse_ev_ang": _rmse(np.concatenate(force_errors)),
                "force_max_abs_ev_ang": _max_abs(np.concatenate(force_errors)),
                "stress_rmse_ev_ang3": _rmse(np.concatenate(stress_errors)),
                "stress_max_abs_ev_ang3": _max_abs(np.concatenate(stress_errors)),
            },
            "frames": records,
        }
        _write_json(data, metrics_path)
        warning = None
        if context.config.ddb.include_elastic and elastic_norm == 0.0:
            warning = "DDB harmonic validation found no elastic constants in parsed DDB"
        return {"metrics": context.relative_path(metrics_path), "warning": warning}
    except Exception as exc:
        return {"warning": f"Could not write DDB harmonic validation: {exc}"}


def _tiny_harmonic_validation_frames(context: WorkflowContext) -> list[Atoms]:
    rng = np.random.default_rng(context.config.seed + 1701)
    frames = []
    for _ in range(max(1, int(context.config.ddb.harmonic_validation_frames))):
        atoms = context.parent_atoms.copy()
        strain = rng.normal(
            scale=context.config.ddb.harmonic_strain_amplitude, size=(3, 3)
        )
        strain = 0.5 * (strain + strain.T)
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
        reference_positions = atoms.positions.copy()
        displacement = rng.normal(
            scale=context.config.ddb.harmonic_displacement_amplitude,
            size=atoms.positions.shape,
        )
        atoms.positions += displacement
        atoms.info["harmonic_validation_strain"] = strain
        atoms.info["harmonic_validation_displacement"] = displacement
        atoms.info["harmonic_validation_reference_positions"] = reference_positions
        frames.append(atoms)
    return frames


def _calculator_values(atoms: Atoms) -> dict[str, Any]:
    return {
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces(), dtype=float),
        "stress": np.asarray(atoms.get_stress(voigt=True), dtype=float),
    }


def _calculator_delta(
    values: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    return {
        "energy": values["energy"] - reference["energy"],
        "forces": values["forces"] - reference["forces"],
        "stress": values["stress"] - reference["stress"],
    }


def _harmonic_response_energy(
    delta: dict[str, Any], reference: dict[str, Any], atoms: Atoms
) -> float:
    displacement = np.asarray(
        atoms.info.get(
            "harmonic_validation_displacement", np.zeros_like(delta["forces"])
        ),
        dtype=float,
    )
    strain = np.asarray(
        atoms.info.get("harmonic_validation_strain", np.zeros((3, 3))), dtype=float
    )
    strain_voigt = _strain_tensor_to_voigt(strain)
    force_work = float(np.sum(reference["forces"] * displacement))
    stress_work = float(atoms.get_volume() * np.dot(reference["stress"], strain_voigt))
    return float(delta["energy"] + force_work - stress_work)


def _strain_tensor_to_voigt(strain: np.ndarray) -> np.ndarray:
    return np.array(
        [
            strain[0, 0],
            strain[1, 1],
            strain[2, 2],
            strain[1, 2] + strain[2, 1],
            strain[2, 0] + strain[0, 2],
            strain[0, 1] + strain[1, 0],
        ],
        dtype=float,
    )


def _safe_norm(value: Any) -> float:
    if value is None:
        return 0.0
    array = np.asarray(value, dtype=float)
    return float(np.linalg.norm(array)) if array.size else 0.0


def _rmse(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(array**2))) if array.size else 0.0


def _max_abs(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _frequency_difference_summary(
    ddb_freqs: np.ndarray, phonopy_freqs: np.ndarray
) -> dict[str, float]:
    diff = np.abs(
        np.asarray(ddb_freqs, dtype=float) - np.asarray(phonopy_freqs, dtype=float)
    )
    return {
        "max_cm-1": float(np.max(diff)),
        "mean_cm-1": float(np.mean(diff)),
        "rms_cm-1": float(np.sqrt(np.mean(diff**2))),
    }


def _phonopy_band_frequencies_cm1(phonon, qpoints: np.ndarray) -> np.ndarray:
    freqs = []
    for qpoint in qpoints:
        dynmat = phonon.get_dynamical_matrix_at_q(qpoint)
        eigenvalues = np.linalg.eigvalsh(dynmat)
        freqs.append(
            np.sqrt(np.abs(eigenvalues))
            * np.sign(eigenvalues)
            * phonon.unit_conversion_factor
            * 33.35641
        )
    return np.asarray(freqs, dtype=float)


def _ddb_band_frequencies_cm1(
    ddb_path: Path,
    qpoints: np.ndarray,
    *,
    dipdip: bool = True,
    asr: bool = False,
) -> np.ndarray:
    from pymultibinit.pyeffpot.ddb_parser_complete import read_ddb
    from pymultibinit.pyeffpot.phonon import (
        build_unitcell_ifcs,
        compute_phonon_bands,
    )

    unitcell = read_ddb(str(ddb_path))
    if (
        getattr(unitcell, "qpoints", None) is None
        or getattr(unitcell, "dynmat", None) is None
    ):
        raise ValueError("DDB has no displacement-displacement q-point data")
    unitcell.ifcs = build_unitcell_ifcs(unitcell, dipdip=dipdip, asr=asr)
    return np.asarray(compute_phonon_bands(unitcell, qpoints), dtype=float)


def _bool_option(options: Mapping[str, Any], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _plot_band_comparison(
    *,
    xlist: list[np.ndarray],
    segment_lengths: list[int],
    tick_positions: np.ndarray,
    tick_labels: list[str],
    phonopy_freqs: np.ndarray,
    ddb_freqs: np.ndarray,
    path: Path,
    plt,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    start = 0
    for x_segment, length in zip(xlist, segment_lengths):
        stop = start + length
        for band_index in range(phonopy_freqs.shape[1]):
            label = "phonopy" if start == 0 and band_index == 0 else None
            ax.plot(
                x_segment,
                phonopy_freqs[start:stop, band_index],
                color="#4c78a8",
                linewidth=1.2,
                alpha=0.85,
                label=label,
            )
        for band_index in range(ddb_freqs.shape[1]):
            label = "DDB/pyeffpot" if start == 0 and band_index == 0 else None
            ax.plot(
                x_segment,
                ddb_freqs[start:stop, band_index],
                color="#f58518",
                linewidth=1.0,
                alpha=0.8,
                linestyle="--",
                label=label,
            )
        start = stop
    for position in tick_positions:
        ax.axvline(position, color="0.75", linewidth=0.8)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    ax.set_xlim(float(xlist[0][0]), float(xlist[-1][-1]))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("q-path")
    ax.set_ylabel("Frequency (cm$^{-1}$)")
    ax.set_title("DDB vs Phonopy Phonon Bands")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validate_ddb_parse(path: str | Path) -> dict[str, Any]:
    try:
        from pymultibinit.pyeffpot.ddb_parser_complete import read_ddb

        unitcell = read_ddb(str(path))
        return {"status": "ok", "natom": getattr(unitcell, "natom", None)}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _result_from_context(
    context: WorkflowContext, passed: bool | None | str = "auto"
) -> WorkflowResult:
    validation_metrics = {}
    validate_outputs = (
        context.manifest.get("stages", {}).get("validate", {}).get("outputs", {})
    )
    metrics_path = (
        context.path_from_manifest(validate_outputs.get("metrics"))
        if validate_outputs
        else None
    )
    if metrics_path and metrics_path.exists():
        validation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if passed == "auto":
        passed_value = validation_metrics.get("passed") if validation_metrics else None
    else:
        passed_value = passed
    train_outputs = (
        context.manifest.get("stages", {}).get("train", {}).get("outputs", {})
    )
    model_files = [
        context.path_from_manifest(item)
        for item in train_outputs.get("model_files", [])
    ]
    return WorkflowResult(
        output_dir=context.output_dir,
        manifest=context.manifest,
        report_md=context.output_dir / "report.md",
        report_yaml=context.output_dir / "report.yaml",
        model_files=[path for path in model_files if path is not None],
        validation_metrics=validation_metrics,
        passed=passed_value,
    )


def _stage_signature(config: WorkflowConfig, name: str) -> str:
    payload = {
        "stage": name,
        "stage_version": STAGE_SIGNATURE_VERSIONS.get(name, 1),
        "config": _config_to_dict(config, redact=False),
    }
    text = json.dumps(_sanitize(payload, redact=False), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_to_dict(config: WorkflowConfig, *, redact: bool = True) -> dict[str, Any]:
    data = asdict(config)
    structure = data.get("structure")
    data["structure"] = (
        "<ase.Atoms>" if isinstance(structure, Atoms) else str(structure)
    )
    data["output_dir"] = str(data["output_dir"])
    return _sanitize(data, redact=redact)


def _sanitize(value: Any, *, redact: bool = True) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Atoms):
        return {"formula": value.get_chemical_formula(), "natoms": len(value)}
    if is_dataclass(value):
        return _sanitize(asdict(value), redact=redact)
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            text_key = str(key)
            sanitized[text_key] = (
                "<redacted>"
                if redact and _is_sensitive_key(text_key)
                else _sanitize(item, redact=redact)
            )
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, redact=redact) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(
        token in lowered
        for token in (
            "secret",
            "token",
            "password",
            "credential",
            "api_key",
            "private_key",
        )
    )


def _write_yaml(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_sanitize(data), handle, sort_keys=False)
    return path


def _write_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize(data), indent=2, allow_nan=False), encoding="utf-8"
    )
    return path


def _write_manifest(context: WorkflowContext) -> Path:
    return _write_yaml(context.manifest, context.output_dir / "manifest.yaml")


def _rel_or_abs(context: WorkflowContext, value: str | Path) -> str:
    path = Path(value)
    try:
        return context.relative_path(path)
    except Exception:
        return str(value)


if __name__ == "__main__":
    raise SystemExit(mlmbmodel_cli())
