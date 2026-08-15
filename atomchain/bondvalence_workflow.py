"""End-to-end bond-valence model-building workflow orchestration (mlbvmodel)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from ase import Atoms
from ase.io import read

from atomchain.init_model import init_calc

STAGE_ORDER = [
    "prepare",
    "relax",
    "sample",
    "evaluate",
    "fit",
    "validate",
    "report",
]
STAGE_SIGNATURE_VERSIONS: dict[str, int] = {"relax": 3, "validate": 4, "fit": 4}

MANIFEST_FORMAT = "atomchain-bondvalence-workflow-v1"


@dataclass
class RelaxStageConfig:
    fmax: float = 0.01
    relax_cell: bool = False


@dataclass
class BVSamplingStageConfig:
    rattle_amplitudes: tuple[float, ...] = (0.03, 0.06, 0.10)
    frames_per_amplitude: int = 20
    strain_range: tuple[float, float] = (-0.03, 0.03)
    strain_points: int = 7
    energy_window: float = 0.05


@dataclass
class FitStageConfig:
    validation_fraction: float = 0.25
    loss: str = "linear"
    max_nfev: int = 1_000


@dataclass
class BVValidationStageConfig:
    enabled: bool = True
    gii_threshold: float = 0.2


@dataclass
class BVWorkflowConfig:
    structure: str | Path | Atoms
    seed_file: str | Path
    model: str = "mace"
    model_path: str | None = None
    output_dir: str | Path = "bondvalence_workflow"
    seed: int = 1234
    relax: RelaxStageConfig = field(default_factory=RelaxStageConfig)
    sampling: BVSamplingStageConfig = field(default_factory=BVSamplingStageConfig)
    fit: FitStageConfig = field(default_factory=FitStageConfig)
    validation: BVValidationStageConfig = field(default_factory=BVValidationStageConfig)


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
class BVWorkflowResult:
    output_dir: Path
    manifest: dict[str, Any]
    report_md: Path
    report_yaml: Path
    fitted_parameters: Path | None
    validation_metrics: dict[str, Any]
    passed: bool | None


class BVWorkflowContext:
    """Mutable workflow context shared by stages."""

    def __init__(self, config: BVWorkflowConfig, parent_atoms: Atoms):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.parent_atoms = parent_atoms.copy()
        self.relaxed_atoms: Atoms | None = None
        self.reference_calculator = None
        self.contact_policy: Any = None
        self.cutoff_suggestion: Any = None
        self.manifest: dict[str, Any] = {
            "format": MANIFEST_FORMAT,
            "config": _config_to_dict(config),
            "stages": {},
        }

    def stage_dir(self, name: str) -> Path:
        mapping = {
            "relax": "relax",
            "sample": "sampling",
            "evaluate": "sampling",
            "fit": "fit",
            "validate": "validation",
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

    def validation_enabled(self) -> bool:
        return bool(self.config.validation.enabled)

    def effective_contact_policy(self, seed: Any):
        if self.contact_policy is not None:
            return self.contact_policy
        if seed.contact_policy is not None:
            return seed.contact_policy
        raise RuntimeError(
            "seed declares cutoff: auto but no stage resolved it; "
            "the relax stage must run first"
        )

    def path_from_manifest(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.output_dir / path


StageFunction = Callable[[BVWorkflowContext, StageMetadata], dict[str, Any]]


def run_bondvalence_workflow(
    config: BVWorkflowConfig | None = None,
    *,
    dry_run: bool = False,
    stage_functions: dict[str, StageFunction] | None = None,
    force_stage: list[str] | None = None,
    force_all: bool = False,
    stop_after: str | None = None,
    continue_on_error: bool = False,
    **kwargs: Any,
) -> BVWorkflowResult:
    """Run the bond-valence parameter-fitting workflow."""
    cfg = _coerce_config(config, kwargs)
    parent = _load_parent_atoms(cfg.structure)
    context = BVWorkflowContext(cfg, parent)
    if dry_run:
        context.manifest["stage_plan"] = [{"name": name} for name in STAGE_ORDER]
        return _result_from_context(context, passed=None)

    context.output_dir.mkdir(parents=True, exist_ok=True)
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
        if name == "relax" and metadata.status == "reused":
            _hydrate_relax_context(context, metadata)
        context.manifest["stages"][name] = metadata.to_dict()
        _write_manifest(context)
        if metadata.status == "failed":
            failed = True
            if not continue_on_error:
                if name != "report":
                    # Mark downstream stages skipped, then write a partial
                    # report before surfacing the failure.
                    for later in STAGE_ORDER[STAGE_ORDER.index(name) + 1 :]:
                        if later == "report":
                            continue
                        context.manifest["stages"][later] = StageMetadata(
                            name=later,
                            status="skipped",
                            warnings=["Skipped after previous stage failure"],
                        ).to_dict()
                    report_metadata = _run_stage(
                        context, "report", stage_map["report"], force=True
                    )
                    context.manifest["stages"]["report"] = report_metadata.to_dict()
                    _write_manifest(context)
                raise RuntimeError(metadata.error)
    return _result_from_context(context)


def run_prepare_stage(
    context: BVWorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    """Normalize inputs, validate the seed file, and echo provenance."""
    from ase.io import write as ase_write

    seed = load_seed_file(
        context.config.seed_file,
        species=context.parent_atoms.get_chemical_symbols(),
    )
    for warning in seed.warnings:
        metadata.warnings.append(warning)

    parent_path = context.output_dir / "parent.vasp"
    ase_write(parent_path, context.parent_atoms, format="vasp", direct=True, vasp5=True)
    config_path = _write_yaml(
        _config_to_dict(context.config), context.output_dir / "config.yaml"
    )
    seed_echo_path = context.output_dir / "seed_echo.yaml"
    seed_echo_path.write_text(
        Path(context.config.seed_file).read_text(encoding="utf-8"), encoding="utf-8"
    )
    return {
        "parent": context.relative_path(parent_path),
        "config": context.relative_path(config_path),
        "seed_echo": context.relative_path(seed_echo_path),
    }


def run_relax_stage(
    context: BVWorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    """Relax the parent structure with the reference calculator (story-003)."""
    from ase.optimize import BFGS

    atoms = context.parent_atoms.copy()
    provided_calculator = context.reference_calculator is not None
    calculator = _reference_calculator(context)
    atoms.calc = calculator
    fmax = float(context.config.relax.fmax)
    if context.config.relax.relax_cell:
        from ase.filters import UnitCellFilter

        target = UnitCellFilter(atoms)
    else:
        target = atoms
    optimizer = BFGS(target, logfile=None)
    converged = optimizer.run(fmax=fmax, steps=500)
    final_fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    energy_per_atom = float(atoms.get_potential_energy() / len(atoms))
    atoms.calc = None
    if not converged:
        raise RuntimeError(
            f"relaxation did not converge to fmax={fmax} within 500 steps "
            f"(final atomic fmax={final_fmax:.6f})"
        )
    context.relaxed_atoms = atoms

    seed = load_seed_file(
        context.config.seed_file, species=atoms.get_chemical_symbols()
    )
    sampling = context.config.sampling
    sigma_max = max((float(a) for a in sampling.rattle_amplitudes), default=0.0)
    strain_lo, strain_hi = (float(v) for v in sampling.strain_range)
    eps_max = max(abs(strain_lo), abs(strain_hi))
    if seed.contact_policy is None:
        policy = _resolve_contact_policy(context, seed, atoms)
    else:
        policy = seed.contact_policy
        suggestion = suggest_cutoffs(atoms, seed, sigma_max=sigma_max, eps_max=eps_max)
        context.cutoff_suggestion = suggestion
        for shell in suggestion.live_shells:
            shell_margin = 3.0 * np.sqrt(2.0) * sigma_max + eps_max * shell.distance
            if abs(policy.cutoff - shell.distance) <= shell_margin:
                metadata.warnings.append(
                    f"cutoff {policy.cutoff:g} A is within the flicker margin "
                    f"({shell_margin:.2f} A) of the live {shell.family} shell "
                    f"at {shell.distance:.2f} A; contacts will flicker across "
                    "sampled frames"
                )
        if not suggestion.intervals:
            metadata.warnings.append(
                f"cutoff {policy.cutoff:g} A is unsafe: no safe interval "
                "exists for the configured rattle/strain margins ("
                + "; ".join(suggestion.warnings)
                + ")"
            )
        elif not any(
            interval.lower <= policy.cutoff <= interval.upper
            for interval in suggestion.intervals
        ):
            metadata.warnings.append(
                f"cutoff {policy.cutoff:g} A lies outside every suggested "
                "safe interval "
                + ", ".join(
                    f"[{interval.lower:.2f}, {interval.upper:.2f}]"
                    for interval in suggestion.intervals
                )
            )
        seed_families = {
            parameter.identifier for parameter in seed.parameter_set.parameters
        }
        missing_at_cutoff = sorted(
            {
                shell.family
                for shell in suggestion.shells
                if shell.distance <= policy.cutoff
            }
            - seed_families
        )
        if missing_at_cutoff:
            metadata.warnings.append(
                f"cutoff {policy.cutoff:g} A includes contact families "
                "without parameters: "
                + ", ".join(missing_at_cutoff)
                + "; add fixed near-zero pad parameters or reduce the cutoff"
            )
    outputs_extra: dict[str, Any] = {
        "cutoff": float(policy.cutoff),
        "cutoff_auto": bool(seed.cutoff_auto),
    }
    if context.cutoff_suggestion is not None:
        outputs_extra["cutoff_suggestion"] = context.cutoff_suggestion
    for warning in audit_contacts(seed, atoms, policy):
        metadata.warnings.append(warning)

    relax_dir = context.stage_dir("relax")
    relaxed_path = relax_dir / "relaxed.vasp"
    from ase.io import write as ase_write

    ase_write(relaxed_path, atoms, format="vasp", direct=True, vasp5=True)
    if provided_calculator:
        calculator_provenance = {
            "calculator": type(calculator).__name__,
            "source": "user-provided instance",
        }
    else:
        calculator_provenance = {
            "model": context.config.model,
            "model_path": context.config.model_path,
        }
    relaxation_data = {
        "calculator": calculator_provenance,
        "relax_cell": bool(context.config.relax.relax_cell),
        "fmax_target": fmax,
        "final_fmax": final_fmax,
        "energy_per_atom": energy_per_atom,
        "n_steps": int(optimizer.get_number_of_steps()),
        "n_atoms": len(atoms),
    }
    _write_json(relaxation_data, relax_dir / "relaxation.json")
    outputs = {"relaxed": context.relative_path(relaxed_path)}
    outputs.update(outputs_extra)
    return outputs


def _reference_calculator(context: BVWorkflowContext):
    if context.reference_calculator is None:
        context.reference_calculator = init_calc(
            context.config.model, model_path=context.config.model_path
        )
    return context.reference_calculator


def run_sample_stage(
    context: BVWorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    """Generate relaxed-parent, rattle, and isotropic-strain frames (story-004)."""
    from ase.io import Trajectory

    if context.relaxed_atoms is None:
        raise RuntimeError("relax stage must run before sampling")
    relaxed = context.relaxed_atoms
    sampling = context.config.sampling
    frames: list[Atoms] = []
    manifest_frames: list[dict[str, Any]] = []

    def add_frame(atoms: Atoms, frame_id: str, group: str, kind: str, **extra: Any):
        atoms.info["frame_id"] = frame_id
        atoms.info["group"] = group
        atoms.info["kind"] = kind
        atoms.info.update(extra)
        frames.append(atoms)
        record = {"frame_id": frame_id, "group": group, "kind": kind, **extra}
        manifest_frames.append(record)

    add_frame(relaxed.copy(), "relaxed", "relaxed", "relaxed")

    for amplitude_index, amplitude in enumerate(sampling.rattle_amplitudes):
        rng = np.random.default_rng([context.config.seed, amplitude_index])
        group = f"rattle-a{float(amplitude):g}"
        for frame_index in range(int(sampling.frames_per_amplitude)):
            atoms = relaxed.copy()
            atoms.positions += rng.normal(
                scale=float(amplitude), size=atoms.positions.shape
            )
            add_frame(
                atoms,
                f"rattle-{amplitude_index}-{frame_index}",
                group,
                "rattle",
                amplitude=float(amplitude),
                rng_seed_lineage=[int(context.config.seed), int(amplitude_index)],
            )

    low, high = (float(value) for value in sampling.strain_range)
    for strain_index, strain in enumerate(
        np.linspace(low, high, int(sampling.strain_points))
    ):
        atoms = relaxed.copy()
        atoms.set_cell((1.0 + float(strain)) * np.asarray(atoms.cell), scale_atoms=True)
        add_frame(
            atoms,
            f"strain-{strain_index}",
            "strain",
            "strain",
            strain=float(strain),
        )

    sampling_dir = context.stage_dir("sample")
    frames_path = sampling_dir / "frames.traj"
    with Trajectory(frames_path, "w") as trajectory:
        for atoms in frames:
            trajectory.write(atoms)
    manifest_data = {
        "n_frames": len(manifest_frames),
        "seed": int(context.config.seed),
        "rattle_amplitudes": [float(a) for a in sampling.rattle_amplitudes],
        "frames_per_amplitude": int(sampling.frames_per_amplitude),
        "strain_range": [low, high],
        "strain_points": int(sampling.strain_points),
        "frames": manifest_frames,
    }
    _write_yaml(manifest_data, sampling_dir / "frame_manifest.yaml")
    return {"frames": context.relative_path(frames_path)}


def run_evaluate_stage(
    context: BVWorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    """Single-point energies and energy-window filtering (story-005)."""
    from ase.io import Trajectory

    sample_outputs = (
        context.manifest.get("stages", {}).get("sample", {}).get("outputs", {})
    )
    frames_path = context.path_from_manifest(sample_outputs.get("frames"))
    if frames_path is None or not Path(frames_path).exists():
        raise RuntimeError("sample stage must run before evaluate")
    frames = read(frames_path, ":")
    if isinstance(frames, Atoms):
        frames = [frames]
    calc = _reference_calculator(context)
    energies: dict[str, float] = {}
    for atoms in frames:
        atoms.calc = calc
        energies[atoms.info["frame_id"]] = float(atoms.get_potential_energy())
        atoms.calc = None
    if "relaxed" not in energies:
        raise RuntimeError("sampled frames must include the relaxed parent frame")
    relaxed_n = next(
        len(atoms) for atoms in frames if atoms.info["frame_id"] == "relaxed"
    )
    reference_per_atom = energies["relaxed"] / relaxed_n
    window = float(context.config.sampling.energy_window)

    records: list[dict[str, Any]] = []
    group_stats: dict[str, dict[str, int]] = {}
    kept_frames: list[Atoms] = []
    for atoms in frames:
        frame_id = atoms.info["frame_id"]
        group = atoms.info["group"]
        energy_per_atom = energies[frame_id] / len(atoms)
        delta = energy_per_atom - reference_per_atom
        kept = bool(delta <= window)
        stats = group_stats.setdefault(group, {"kept": 0, "dropped": 0})
        stats["kept" if kept else "dropped"] += 1
        records.append(
            {
                "frame_id": frame_id,
                "group": group,
                "energy_per_atom": energy_per_atom,
                "delta_vs_parent": delta,
                "kept": kept,
            }
        )
        if kept:
            kept_frames.append(atoms)

    for group, stats in sorted(group_stats.items()):
        if group != "relaxed" and stats["kept"] == 0:
            metadata.warnings.append(
                f"structure group {group!r} was entirely dropped by the energy "
                f"window; the fit/validation split may lose this family"
            )

    sampling_dir = context.stage_dir("evaluate")
    evaluated_path = sampling_dir / "evaluated.traj"
    with Trajectory(evaluated_path, "w") as trajectory:
        for atoms in kept_frames:
            trajectory.write(atoms)
    filter_data = {
        "energy_window_per_atom": window,
        "reference_energy_per_atom": reference_per_atom,
        "n_frames": len(records),
        "n_kept": len(kept_frames),
        "group_stats": group_stats,
        "frames": records,
    }
    filter_path = _write_json(filter_data, sampling_dir / "filter.json")
    return {
        "evaluated": context.relative_path(evaluated_path),
        "filter": context.relative_path(filter_path),
        "group_stats": group_stats,
        "n_kept": len(kept_frames),
    }


def run_fit_stage(
    context: BVWorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    """Build the fit dataset, split by group, and fit parameters (story-006)."""
    easybondvalence = _require_easybondvalence()
    evaluate_outputs = (
        context.manifest.get("stages", {}).get("evaluate", {}).get("outputs", {})
    )
    evaluated_path = context.path_from_manifest(evaluate_outputs.get("evaluated"))
    if evaluated_path is None or not Path(evaluated_path).exists():
        raise RuntimeError("evaluate stage must run before fit")
    frames = read(evaluated_path, ":")
    if isinstance(frames, Atoms):
        frames = [frames]
    if not frames:
        raise RuntimeError("evaluate stage produced no kept frames")

    seed = load_seed_file(
        context.config.seed_file, species=context.parent_atoms.get_chemical_symbols()
    )
    records = [
        easybondvalence.FitStructureRecord(
            structure_id=atoms.info["frame_id"],
            split_group_id=atoms.info["group"],
            structure=_to_bv_structure(atoms),
            valences=seed.valences_for(atoms.get_chemical_symbols()),
            contact_policy=context.effective_contact_policy(seed),
        )
        for atoms in frames
    ]
    sampling = context.config.sampling
    if context.reference_calculator is not None:
        calculator_note = f"{type(context.reference_calculator).__name__} (injected)"
    else:
        calculator_note = context.config.model
    provenance = (
        f"{seed.provenance}; calculator={calculator_note}; "
        f"sampling: rattle={list(sampling.rattle_amplitudes)}x"
        f"{sampling.frames_per_amplitude}, strain={list(sampling.strain_range)}x"
        f"{sampling.strain_points}, energy_window={sampling.energy_window} eV/atom; "
        f"split seed={context.config.seed} "
        f"(atomchain bond-valence workflow)"
    )
    dataset = easybondvalence.FitDataset.from_records(records, provenance=provenance)
    validation_fraction = float(context.config.fit.validation_fraction)
    split_seed = int(context.config.seed)
    partition = easybondvalence.split_structures(
        dataset, validation_fraction=validation_fraction, seed=split_seed
    )

    training_groups = set(partition.training_group_ids)
    training_kinds = {
        atoms.info.get("kind")
        for atoms in frames
        if atoms.info["group"] in training_groups
    }
    if "strain" not in training_kinds:
        metadata.warnings.append(
            "training split contains no strained frame; R0-b identifiability "
            "may be reduced"
        )

    plan = easybondvalence.FitPlan(
        parameters=seed.fit_parameters,
        loss=context.config.fit.loss,
        max_nfev=int(context.config.fit.max_nfev),
    )
    result = easybondvalence.fit_parameters(
        partition.training, seed.parameter_set, plan
    )
    validation_score = result.score(partition.validation)

    fitted_dir = context.stage_dir("fit")
    fitted_parameters = _write_fitted_parameters(result, provenance)
    fitted_parameters["contacts"] = {
        "cutoff": context.effective_contact_policy(seed).cutoff
    }
    fitted_parameters["oxidation_states"] = dict(seed.oxidation_states)
    fitted_path = _write_yaml(fitted_parameters, fitted_dir / "fitted_parameters.yaml")

    sensitivity = result.sensitivity
    covariance = result.covariance
    uncertainties: dict[str, dict[str, float | None]] = {}
    covariance_matrix: list[list[float]] | None = None
    correlation_matrix: list[list[float]] | None = None
    if covariance.available:
        matrix = np.asarray(covariance.matrix)
        covariance_matrix = matrix.tolist()
        deviation = np.sqrt(np.diag(matrix))
        outer = np.outer(deviation, deviation)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = matrix / np.where(outer == 0.0, np.nan, outer)
        correlation_matrix = np.nan_to_num(correlation, nan=0.0).tolist()
        for position, key in enumerate(sensitivity.coordinate_keys):
            identifier, field = key.split(":", 1)
            uncertainties.setdefault(identifier, {})[field] = float(
                np.sqrt(matrix[position, position])
            )
    diagnostics = {
        "converged": bool(result.converged),
        "termination": result.termination,
        "data_cost": float(result.data_cost),
        "validation_data_cost": float(validation_score.data_cost),
        "regularization_cost": float(result.regularization_cost),
        "function_evaluations": int(result.function_evaluations),
        "jacobian_evaluations": result.jacobian_evaluations,
        "sensitivity": {
            "coordinate_keys": list(sensitivity.coordinate_keys),
            "singular_values": np.asarray(sensitivity.singular_values).tolist(),
            "rank": int(sensitivity.rank),
            "condition_number": _json_safe_number(sensitivity.condition_number),
            "condition_number_infinite": (
                sensitivity.condition_number is not None
                and not np.isfinite(float(sensitivity.condition_number))
            ),
        },
        "covariance": {
            "available": bool(covariance.available),
            "reason": covariance.reason,
            "uncertainties": uncertainties,
            "matrix": covariance_matrix,
            "correlation_matrix": correlation_matrix,
        },
        "coverage": {
            "required_parameter_keys": list(result.coverage.required_parameter_keys),
            "plan_parameter_keys": list(result.coverage.plan_parameter_keys),
            "observed_parameter_identifiers": list(
                result.coverage.observed_parameter_identifiers
            ),
            "unobserved_parameter_identifiers": list(
                result.coverage.unobserved_parameter_identifiers
            ),
            "contact_count": int(result.coverage.contact_count),
            "covered_residuals": [
                {"structure_id": item.structure_id, "site_index": int(item.site_index)}
                for item in result.coverage.covered_residuals
            ],
        },
        "environment": dict(result.environment),
        "descriptor_version": result.descriptor_version,
        "dataset_fingerprint": result.dataset_fingerprint,
        "dataset_provenance": result.dataset_provenance,
    }
    diagnostics_path = _write_json(diagnostics, fitted_dir / "diagnostics.json")

    split_data = {
        "validation_fraction": validation_fraction,
        "seed": split_seed,
        "training_group_ids": list(partition.training_group_ids),
        "validation_group_ids": list(partition.validation_group_ids),
        "training_dataset_fingerprint": partition.training.fingerprint,
        "validation_dataset_fingerprint": partition.validation.fingerprint,
        "fitted_dataset_fingerprint": result.dataset_fingerprint,
    }
    split_path = _write_json(split_data, fitted_dir / "split.json")
    return {
        "fitted_parameters": context.relative_path(fitted_path),
        "diagnostics": context.relative_path(diagnostics_path),
        "split": context.relative_path(split_path),
    }


def run_validate_stage(
    context: BVWorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    """Unweighted held-out GII gate and identifiability reporting (story-007)."""
    if not context.validation_enabled():
        metadata.status = "skipped"
        validation_dir = context.stage_dir("validate")
        for stale in ("metrics.json", "dv_distribution.png", "dv_vs_distance.png"):
            stale_path = validation_dir / stale
            if stale_path.exists():
                stale_path.unlink()
        return {"skipped": True}
    easybondvalence = _require_easybondvalence()
    fit_outputs = context.manifest.get("stages", {}).get("fit", {}).get("outputs", {})
    fitted_path = context.path_from_manifest(fit_outputs.get("fitted_parameters"))
    if fitted_path is None or not Path(fitted_path).exists():
        raise RuntimeError("fit stage must run before validate")
    split_path = context.path_from_manifest(fit_outputs.get("split"))
    diagnostics_path = context.path_from_manifest(fit_outputs.get("diagnostics"))
    evaluate_outputs = (
        context.manifest.get("stages", {}).get("evaluate", {}).get("outputs", {})
    )
    evaluated_path = context.path_from_manifest(evaluate_outputs.get("evaluated"))
    if (
        split_path is None
        or evaluated_path is None
        or not Path(evaluated_path).exists()
    ):
        raise RuntimeError("fit and evaluate outputs are required for validation")
    if diagnostics_path is None or not Path(diagnostics_path).exists():
        raise RuntimeError("fit diagnostics are required for validation")

    fitted = yaml.safe_load(Path(fitted_path).read_text(encoding="utf-8"))
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(diagnostics_path).read_text(encoding="utf-8"))
    seed = load_seed_file(
        context.config.seed_file,
        species=context.parent_atoms.get_chemical_symbols(),
    )
    parameters = [
        easybondvalence.BondValenceParameter(
            identifier=identifier,
            selector=easybondvalence.ParameterSelector(*pair["selector"]),
            r0=float(pair["r0"]),
            b=float(pair["b"]),
            source=pair["source"],
        )
        for identifier, pair in fitted["pairs"].items()
    ]
    fitted_parameter_set = easybondvalence.ParameterSet(parameters)

    frames = read(evaluated_path, ":")
    if isinstance(frames, Atoms):
        frames = [frames]
    validation_groups = set(split["validation_group_ids"])
    validation_frames = [
        atoms for atoms in frames if atoms.info["group"] in validation_groups
    ]
    if not validation_frames:
        raise RuntimeError("validation split is empty; cannot validate")

    all_dv: list[float] = []
    dv_distances: list[float] = []
    per_structure: list[dict[str, Any]] = []
    group_dv: dict[str, list[float]] = {}
    for atoms in validation_frames:
        symbols = atoms.get_chemical_symbols()
        structure = _to_bv_structure(atoms)
        valences = seed.valences_for(symbols)
        structural = easybondvalence.evaluate(
            fitted_parameter_set,
            structure,
            contacts=context.effective_contact_policy(seed),
            valences=valences,
        )
        mismatches = np.asarray(structural.mismatches, dtype=float)
        per_site_distance = np.zeros(len(symbols))
        per_site_contacts = np.zeros(len(symbols), dtype=int)
        for contact in structural.contacts:
            per_site_distance[contact.i] += contact.distance
            per_site_contacts[contact.i] += 1
            per_site_distance[contact.j] += contact.distance
            per_site_contacts[contact.j] += 1
        for site, value in enumerate(mismatches):
            all_dv.append(float(value))
            dv_distances.append(
                float(per_site_distance[site] / per_site_contacts[site])
                if per_site_contacts[site]
                else 0.0
            )
        group = atoms.info["group"]
        group_dv.setdefault(group, []).extend(float(v) for v in mismatches)
        per_structure.append(
            {
                "frame_id": atoms.info["frame_id"],
                "group": group,
                "gii": float(structural.gii),
                "n_sites": len(symbols),
                "max_abs_dv": float(np.max(np.abs(mismatches))),
            }
        )

    aggregate_rms = float(np.sqrt(np.mean(np.square(all_dv))))
    threshold = float(context.config.validation.gii_threshold)
    passed = bool(aggregate_rms < threshold)
    per_group = {
        group: float(np.sqrt(np.mean(np.square(values))))
        for group, values in sorted(group_dv.items())
    }

    active_bounds: list[str] = []
    for item in seed.fit_parameters:
        if item.status != "free":
            continue
        value = float(fitted["pairs"][item.identifier][item.field])
        if (item.lower is not None and abs(value - item.lower) <= 1e-9) or (
            item.upper is not None and abs(value - item.upper) <= 1e-9
        ):
            active_bounds.append(item.key)

    identifiability_warnings: list[str] = []
    coordinate_count = len(
        (diagnostics.get("sensitivity") or {}).get("coordinate_keys") or []
    )
    fit_rank = (diagnostics.get("sensitivity") or {}).get("rank")
    if fit_rank is not None and coordinate_count and fit_rank < coordinate_count:
        identifiability_warnings.append(
            f"fit Jacobian is rank deficient (rank {fit_rank} < "
            f"{coordinate_count} free coordinates); fitted values are not "
            "locally unique"
        )
    if (diagnostics.get("covariance") or {}).get("available") is False:
        reason = (diagnostics.get("covariance") or {}).get("reason")
        identifiability_warnings.append(
            "covariance unavailable" + (f" ({reason})" if reason else "")
        )
    if (diagnostics.get("sensitivity") or {}).get("condition_number_infinite"):
        identifiability_warnings.append(
            "fit condition number is infinite (exactly singular Jacobian)"
        )
    for warning in identifiability_warnings:
        metadata.warnings.append(warning)

    metrics = {
        "passed": passed,
        "threshold": threshold,
        "aggregate_rms_dv": aggregate_rms,
        "held_out_data_cost": diagnostics.get("validation_data_cost"),
        "n_validation_structures": len(validation_frames),
        "n_validation_sites": len(all_dv),
        "per_structure": per_structure,
        "per_group_gii": per_group,
        "identifiability": {
            "rank": diagnostics.get("sensitivity", {}).get("rank"),
            "condition_number": diagnostics.get("sensitivity", {}).get(
                "condition_number"
            ),
            "covariance_available": diagnostics.get("covariance", {}).get("available"),
            "covariance_reason": diagnostics.get("covariance", {}).get("reason"),
            "uncertainties": diagnostics.get("covariance", {}).get("uncertainties"),
            "active_bounds": active_bounds,
            "warnings": identifiability_warnings,
            "fit_converged": diagnostics.get("converged"),
            "fit_termination": diagnostics.get("termination"),
        },
    }
    if active_bounds:
        metadata.warnings.append(
            "fitted parameters at their bounds: " + ", ".join(active_bounds)
        )
    if not diagnostics.get("converged", True):
        metadata.warnings.append("the underlying fit did not converge")

    validation_dir = context.stage_dir("validate")
    plots = _write_validation_plots(validation_dir, all_dv, dv_distances)
    plots = {key: str(Path(value).resolve()) for key, value in plots.items()}
    metrics_path = _write_json(metrics, validation_dir / "metrics.json")
    outputs = {
        "metrics": context.relative_path(metrics_path),
        "passed": passed,
    }
    outputs.update(plots)
    return outputs


def _write_validation_plots(
    validation_dir: Path, all_dv: list[float], dv_distances: list[float]
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distribution_path = validation_dir / "dv_distribution.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(all_dv, bins=30)
    axis.set_xlabel(r"$\Delta V_i$ (v.u.)")
    axis.set_ylabel("site count")
    figure.tight_layout()
    figure.savefig(distribution_path, dpi=150)
    plt.close(figure)

    distance_path = validation_dir / "dv_vs_distance.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(dv_distances, all_dv, s=8, alpha=0.6)
    axis.set_xlabel("site mean contact distance (Angstrom)")
    axis.set_ylabel(r"$\Delta V_i$ (v.u.)")
    figure.tight_layout()
    figure.savefig(distance_path, dpi=150)
    plt.close(figure)
    return {
        "dv_distribution": str(distribution_path),
        "dv_vs_distance": str(distance_path),
    }


def run_report_stage(
    context: BVWorkflowContext, metadata: StageMetadata
) -> dict[str, Any]:
    """Assemble the markdown/YAML report and multipole descriptors (story-008).

    One structured payload is built and written verbatim to report.yaml; the
    markdown sections render from the same payload so the two formats cannot
    diverge.
    """
    output_dir = context.output_dir
    stages = dict(context.manifest.get("stages", {}))
    if "report" not in stages:
        # The report stage renders before its own manifest entry exists.
        stages["report"] = {"name": "report", "status": "completed"}

    def outputs_of(name: str) -> dict[str, Any]:
        return stages.get(name, {}).get("outputs", {}) or {}

    def read_json(name: str, key: str) -> dict[str, Any]:
        path = context.path_from_manifest(outputs_of(name).get(key))
        if path is None or not Path(path).exists():
            return {}
        return json.loads(Path(path).read_text(encoding="utf-8"))

    fitted = {}
    fitted_path = context.path_from_manifest(outputs_of("fit").get("fitted_parameters"))
    if fitted_path is not None and Path(fitted_path).exists():
        fitted = yaml.safe_load(Path(fitted_path).read_text(encoding="utf-8")) or {}
    diagnostics = read_json("fit", "diagnostics")
    split = read_json("fit", "split")
    metrics = read_json("validate", "metrics")
    filter_data = read_json("evaluate", "filter")

    seed_data: dict[str, Any] = {}
    seed_path = Path(context.config.seed_file)
    if seed_path.exists():
        seed_data = yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}
    literature = seed_data.get("literature") or {}
    has_literature = bool(literature)

    multipoles_path: Path | None = None
    relaxed_path = context.path_from_manifest(outputs_of("relax").get("relaxed"))
    relaxed_source: Path | Any = None
    if relaxed_path is not None and Path(relaxed_path).exists():
        relaxed_source = Path(relaxed_path)
    elif context.relaxed_atoms is not None:
        relaxed_source = context.relaxed_atoms
    if fitted and relaxed_source is not None:
        try:
            multipoles_path = _write_multipole_descriptors(
                context, fitted, relaxed_source
            )
        except Exception as exc:
            metadata.warnings.append(f"multipole evaluation failed: {exc}")
    elif fitted:
        metadata.warnings.append(
            "multipole evaluation skipped: no relaxed parent available"
        )

    seed_section: dict[str, Any] = {"provenance": seed_data.get("provenance")}
    contacts = seed_data.get("contacts") or {}
    seed_section["cutoff"] = contacts.get("cutoff")
    seed_section["oxidation_states"] = seed_data.get("oxidation_states") or {}
    seed_pairs: list[dict[str, Any]] = []
    for pair in seed_data.get("pairs") or []:
        selector = pair.get("selector") or []
        seed_pairs.append(
            {
                "identifier": "-".join(selector),
                "selector": list(selector),
                "r0": pair.get("r0") or {},
                "b": pair.get("b") or {},
            }
        )
    seed_section["pairs"] = seed_pairs
    seed_section["literature"] = literature

    artifacts: list[dict[str, str]] = []
    for name in STAGE_ORDER:
        for link in _iter_output_links(outputs_of(name)):
            artifacts.append({"stage": name, "path": link})
    if multipoles_path is not None:
        artifacts.append(
            {"stage": "report", "path": context.relative_path(multipoles_path)}
        )

    comparison: list[dict[str, Any]] = []
    for identifier, pair in sorted((fitted.get("pairs") or {}).items()):
        seed_pair = next(
            (item for item in seed_pairs if item["identifier"] == identifier),
            {},
        )
        for field_name, label in (("r0", "r0"), ("b", "b")):
            row = {
                "pair": identifier,
                "quantity": label,
                "seed": (seed_pair.get(field_name) or {}).get("initial"),
                "fitted": pair.get(field_name),
            }
            if has_literature:
                row["literature"] = (literature.get(identifier) or {}).get(field_name)
            comparison.append(row)

    relax_outputs = outputs_of("relax")
    cutoff_section: dict[str, Any] = {}
    if "cutoff" in relax_outputs:
        cutoff_section["chosen"] = relax_outputs["cutoff"]
        cutoff_section["auto"] = bool(relax_outputs.get("cutoff_auto"))
        suggestion = relax_outputs.get("cutoff_suggestion") or {}
        if suggestion:
            deduped: dict[tuple[float, float], dict[str, Any]] = {}
            for interval in suggestion.get("intervals", []):
                key = (interval["lower"], interval["upper"])
                if key not in deduped:
                    deduped[key] = {
                        "lower": interval["lower"],
                        "upper": interval["upper"],
                        "representatives": [interval["representative"]],
                        "families_inside": interval["families_inside"],
                        "missing_families": interval["missing_families"],
                    }
                else:
                    deduped[key]["representatives"].append(interval["representative"])
            cutoff_section["intervals"] = list(deduped.values())
            cutoff_section["live_shells"] = suggestion.get("live_shells", [])

    payload: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "config": context.manifest.get("config"),
        "stages": stages,
        "environment": diagnostics.get("environment"),
        "seed": seed_section,
        "sampling": {
            **filter_data,
            "split": split,
        },
        "fit": {
            "pairs": fitted.get("pairs") or {},
            "uncertainties": (diagnostics.get("covariance") or {}).get("uncertainties"),
            "converged": diagnostics.get("converged"),
            "termination": diagnostics.get("termination"),
            "data_cost": diagnostics.get("data_cost"),
            "validation_data_cost": diagnostics.get("validation_data_cost"),
            "sensitivity": diagnostics.get("sensitivity"),
            "coverage": diagnostics.get("coverage"),
            "covariance": {
                key: value
                for key, value in (diagnostics.get("covariance") or {}).items()
                if key != "matrix"
            },
            "dataset_fingerprint": diagnostics.get("dataset_fingerprint"),
            "dataset_provenance": diagnostics.get("dataset_provenance"),
        },
        "validation": metrics,
        "cutoff": cutoff_section,
        "comparison": comparison,
        "artifacts": artifacts,
        "multipoles": (
            context.relative_path(multipoles_path)
            if multipoles_path is not None
            else None
        ),
        "limitations": (
            "BVS and GII are structural diagnostics in valence units; they are "
            "not energies and do not establish thermodynamic stability. "
            "Interpret the fitted parameters within the seed provenance's "
            "chemical applicability, the stated contact policy, and "
            "oxidation-state assignment."
        ),
    }

    report_md = output_dir / "report.md"
    report_md.write_text(
        "\n".join(_report_markdown(payload, has_literature)), encoding="utf-8"
    )
    report_yaml = _write_yaml(payload, output_dir / "report.yaml")
    outputs = {
        "report_md": context.relative_path(report_md),
        "report_yaml": context.relative_path(report_yaml),
    }
    if multipoles_path is not None:
        outputs["multipoles"] = context.relative_path(multipoles_path)
    return outputs


def _report_markdown(payload: dict[str, Any], has_literature: bool) -> list[str]:
    lines: list[str] = []
    lines += ["# Bond-Valence Model Workflow Report", ""]
    lines += [
        "## Procedure",
        "",
        "Stages: " + " -> ".join(STAGE_ORDER) + ".",
        "MACE/reference-calculator energies filter sampled frames only; they "
        "never enter the fit residual.",
        "",
    ]
    config_rows = _flatten_report_values(payload.get("config") or {})
    lines += ["## Configuration", "", "| Key | Value |", "|---|---|"]
    for key, value in config_rows:
        lines.append(f"| `{key}` | `{value}` |")

    lines += [
        "",
        "## Stages",
        "",
        "| Stage | Status | Warnings | Error |",
        "|---|---|---|---|",
    ]
    for name in STAGE_ORDER:
        stage = (payload.get("stages") or {}).get(name, {})
        warnings = "; ".join(stage.get("warnings") or []) or "-"
        error = stage.get("error") or "-"
        lines.append(
            f"| {name} | {stage.get('status', 'pending')} | {warnings} | {error} |"
        )

    lines += ["", "## Artifact Index", "", "| Stage | Artifact |", "|---|---|"]
    for item in payload.get("artifacts") or []:
        lines.append(f"| {item['stage']} | [{item['path']}]({item['path']}) |")

    seed = payload.get("seed") or {}
    lines += ["", "## Seed", ""]
    lines += [
        f"- provenance: {seed.get('provenance')}",
        f"- contact cutoff: {seed.get('cutoff')} Angstrom",
        "- oxidation states: "
        + ", ".join(
            f"{element} {value:+g}"
            for element, value in (seed.get("oxidation_states") or {}).items()
        ),
        "",
        "| Pair | R0 initial [bounds] | b initial [bounds] |",
        "|---|---|---|",
    ]
    for pair in seed.get("pairs") or []:

        def fmt(spec: dict[str, Any]) -> str:
            initial = spec.get("initial")
            lower = spec.get("lower")
            upper = spec.get("upper")
            return f"{initial} [{lower}, {upper}]"

        lines.append(
            f"| {pair.get('identifier')} | {fmt(pair.get('r0') or {})} "
            f"| {fmt(pair.get('b') or {})} |"
        )

    cutoff = payload.get("cutoff") or {}
    if cutoff:
        lines += ["", "## Contact Cutoff", ""]
        source = "auto-resolved" if cutoff.get("auto") else "user-specified"
        lines.append(f"- chosen cutoff: {cutoff.get('chosen'):g} Angstrom ({source})")
        intervals = cutoff.get("intervals") or []
        if intervals:
            lines += [
                "",
                "Suggested safe intervals (clearing the flicker margin):",
                "",
                "| Interval (Angstrom) | Representative | Missing families |",
                "|---|---|---|",
            ]
            for interval in intervals:
                missing = ", ".join(interval.get("missing_families") or ()) or "-"
                representatives = (
                    ", ".join(
                        f"{value:.2f}" for value in interval.get("representatives", [])
                    )
                    or "-"
                )
                lines.append(
                    f"| [{interval['lower']:.2f}, {interval['upper']:.2f}] "
                    f"| {representatives} | {missing} |"
                )
        live_shells = cutoff.get("live_shells") or []
        if live_shells:
            lines += [
                "",
                "Live shells (weight >= theta):",
                "",
                "| Family | Distance (Angstrom) | Multiplicity | Weight (v.u.) |",
                "|---|---|---|---|",
            ]
            for shell in live_shells:
                lines.append(
                    f"| {shell['family']} | {shell['distance']:.2f} "
                    f"| {shell['multiplicity']} | {shell['weight']:.4g} |"
                )

    sampling = payload.get("sampling") or {}
    lines += ["", "## Sampling", ""]
    if sampling:
        lines += [
            f"- frames: {sampling.get('n_frames')}, kept: "
            f"{sampling.get('n_kept')} within "
            f"{sampling.get('energy_window_per_atom')} eV/atom",
            "- group statistics: "
            + ", ".join(
                f"{group} {stats['kept']}/{stats['kept'] + stats['dropped']}"
                for group, stats in sorted((sampling.get("group_stats") or {}).items())
            ),
        ]
    else:
        lines.append("- sampling/filter data unavailable")

    fit = payload.get("fit") or {}
    lines += ["", "## Fit", ""]
    if fit.get("pairs"):
        lines += ["| Pair | R0 fitted | b fitted |", "|---|---|---|"]
        for identifier, pair in sorted(fit["pairs"].items()):
            lines.append(f"| {identifier} | {pair['r0']:.6f} | {pair['b']:.6f} |")
        lines += [
            "",
            f"- converged: {fit.get('converged')} ({fit.get('termination')})",
            f"- training data cost: {fit.get('data_cost')}",
            f"- held-out data cost: {fit.get('validation_data_cost')}",
            f"- sensitivity rank: {(fit.get('sensitivity') or {}).get('rank')}",
        ]
        uncertainties = fit.get("uncertainties") or {}
        if uncertainties:
            lines += [
                "- parameter uncertainties (1 sigma): "
                + ", ".join(
                    f"{identifier}:{field_name}={value:.6f}"
                    for identifier, fields in sorted(uncertainties.items())
                    for field_name, value in fields.items()
                )
            ]
        environment = fit.get("dataset_provenance")
        if environment:
            lines += ["", f"Dataset provenance: {environment}"]
    else:
        lines.append("- fitted parameters unavailable")
    split = sampling.get("split") or {}
    if split:
        lines += [
            f"- split seed: {split.get('seed')}, validation fraction: "
            f"{split.get('validation_fraction')}",
            f"- validation groups: {', '.join(split.get('validation_group_ids') or [])}",
        ]

    validation = payload.get("validation") or {}
    lines += ["", "## Validation", ""]
    if validation:
        verdict = "PASSED" if validation.get("passed") else "FAILED"
        lines += [
            f"**Gate {verdict}**: aggregate RMS dV = "
            f"{validation.get('aggregate_rms_dv'):.6f} v.u. vs threshold "
            f"{validation.get('threshold')} v.u.",
            "",
            "| Group | GII (v.u.) |",
            "|---|---|",
        ]
        for group, gii in sorted((validation.get("per_group_gii") or {}).items()):
            lines.append(f"| {group} | {gii:.6f} |")
        identifiability = validation.get("identifiability") or {}
        lines += [
            "",
            f"- identifiability: rank={identifiability.get('rank')}, condition="
            f"{identifiability.get('condition_number')}, covariance="
            f"{identifiability.get('covariance_available')}, active bounds="
            f"{identifiability.get('active_bounds')}",
        ]
        for warning in identifiability.get("warnings") or []:
            lines.append(f"- warning: {warning}")
    else:
        lines.append("- validation metrics unavailable")

    comparison = payload.get("comparison") or []
    lines += ["", "## Parameter Comparison", ""]
    if comparison:
        if has_literature:
            lines += [
                "| Pair | Quantity | Seed | Fitted | Literature |",
                "|---|---|---|---|---|",
            ]
        else:
            lines += ["| Pair | Quantity | Seed | Fitted |", "|---|---|---|---|"]
        for row in comparison:
            base = (
                f"| {row['pair']} | {row['quantity']} | {row['seed']} "
                f"| {row['fitted']:.6f}"
                if isinstance(row["fitted"], (int, float))
                else f"| {row['pair']} | {row['quantity']} | {row['seed']} "
                f"| {row['fitted']}"
            )
            if has_literature:
                lines.append(base + f" | {row.get('literature', '-')} |")
            else:
                lines.append(base + " |")

    if payload.get("multipoles"):
        lines += [
            "",
            "## Multipole Descriptors",
            "",
            f"Evaluated on the relaxed parent: [{payload['multipoles']}]"
            f"({payload['multipoles']}).",
            "These are descriptor-only valence multipoles in v.u.; they supply "
            "no energy, forces, or crystal-field model.",
        ]

    lines += ["", "## Limitations", "", payload.get("limitations", ""), ""]
    return lines


def _write_multipole_descriptors(
    context: BVWorkflowContext, fitted: dict[str, Any], relaxed_source: Path | Atoms
) -> Path:
    easybondvalence = _require_easybondvalence()
    seed = load_seed_file(
        context.config.seed_file,
        species=context.parent_atoms.get_chemical_symbols(),
    )
    parameters = [
        easybondvalence.BondValenceParameter(
            identifier=identifier,
            selector=easybondvalence.ParameterSelector(*pair["selector"]),
            r0=float(pair["r0"]),
            b=float(pair["b"]),
            source=pair["source"],
        )
        for identifier, pair in fitted["pairs"].items()
    ]
    fitted_parameter_set = easybondvalence.ParameterSet(parameters)
    atoms = read(relaxed_source) if isinstance(relaxed_source, Path) else relaxed_source
    structure = _to_bv_structure(atoms)
    structural = easybondvalence.evaluate(
        fitted_parameter_set,
        structure,
        contacts=context.effective_contact_policy(seed),
        valences=seed.valences_for(atoms.get_chemical_symbols()),
    )
    expansion = easybondvalence.expand_multipoles(structural, max_rank=4)
    coefficients = {
        str(rank): {
            "real": np.asarray(values).real.tolist(),
            "imag": np.asarray(values).imag.tolist(),
        }
        for rank, values in enumerate(expansion.coefficients)
    }
    data = {
        "max_rank": 4,
        "dipoles": np.asarray(expansion.dipoles).tolist(),
        "quadrupoles": np.asarray(expansion.quadrupoles).tolist(),
        "coefficients": coefficients,
        "invariants": np.asarray(expansion.invariants).tolist(),
        "units": "valence units (v.u.); descriptor-only, no energy model",
    }
    multipoles_dir = context.output_dir / "multipoles"
    multipoles_dir.mkdir(parents=True, exist_ok=True)
    return _write_json(data, multipoles_dir / "relaxed_parent.json")


def _flatten_report_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(
                _flatten_report_values(item, f"{prefix}.{key}" if prefix else str(key))
            )
    else:
        rows.append((prefix, value))
    return rows


def _iter_output_links(value: Any) -> list[str]:
    links: list[str] = []
    if isinstance(value, str) and _looks_like_artifact(value) and "/" not in value:
        links.append(value)
    elif isinstance(value, str) and _looks_like_artifact(value):
        links.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            links.extend(_iter_output_links(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            links.extend(_iter_output_links(item))
    return links


def _run_stage(
    context: BVWorkflowContext, name: str, func: StageFunction, force: bool = False
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


def _hydrate_relax_context(context: BVWorkflowContext, metadata: StageMetadata) -> None:
    """Restore in-memory state from a reused relax checkpoint."""
    easybondvalence = _require_easybondvalence()
    relaxed_path = context.path_from_manifest(metadata.outputs.get("relaxed"))
    if relaxed_path is not None and Path(relaxed_path).exists():
        context.relaxed_atoms = read(relaxed_path)
    cutoff = metadata.outputs.get("cutoff")
    if cutoff is not None and context.contact_policy is None:
        context.contact_policy = easybondvalence.ContactPolicy(cutoff=float(cutoff))
    suggestion = metadata.outputs.get("cutoff_suggestion")
    if suggestion is not None and context.cutoff_suggestion is None:
        context.cutoff_suggestion = suggestion


def _coerce_config(
    config: BVWorkflowConfig | None, kwargs: dict[str, Any]
) -> BVWorkflowConfig:
    if config is not None and kwargs:
        raise ValueError("Pass either BVWorkflowConfig or keyword arguments, not both")
    if config is not None:
        return config
    if "structure" not in kwargs:
        raise ValueError("Workflow requires a parent structure")
    if "seed_file" not in kwargs:
        raise ValueError("Workflow requires a seed parameter file")
    return BVWorkflowConfig(**kwargs)


def _load_parent_atoms(structure: str | Path | Atoms) -> Atoms:
    return structure.copy() if isinstance(structure, Atoms) else read(str(structure))


def _result_from_context(
    context: BVWorkflowContext, passed: bool | None | str = "auto"
) -> BVWorkflowResult:
    validation_metrics: dict[str, Any] = {}
    fit_outputs = context.manifest.get("stages", {}).get("fit", {}).get("outputs", {})
    validate_outputs = (
        context.manifest.get("stages", {}).get("validate", {}).get("outputs", {})
    )
    metrics_path = context.path_from_manifest(validate_outputs.get("metrics"))
    if metrics_path is not None and Path(metrics_path).exists():
        validation_metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    fitted = context.path_from_manifest(fit_outputs.get("fitted_parameters"))
    if passed == "auto":
        passed = (
            bool(validation_metrics["passed"])
            if "passed" in validation_metrics
            else None
        )
    return BVWorkflowResult(
        output_dir=context.output_dir,
        manifest=context.manifest,
        report_md=context.output_dir / "report.md",
        report_yaml=context.output_dir / "report.yaml",
        fitted_parameters=fitted,
        validation_metrics=validation_metrics,
        passed=passed,
    )


def _stage_signature(config: BVWorkflowConfig, name: str) -> str:
    payload = {
        "stage": name,
        "stage_version": STAGE_SIGNATURE_VERSIONS.get(name, 1),
        "config": _stage_config_slice(config, name),
    }
    text = json.dumps(_sanitize(payload, redact=False), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed_uses_auto_cutoff(config: BVWorkflowConfig) -> bool:
    try:
        data = yaml.safe_load(Path(config.seed_file).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    contacts = data.get("contacts") if isinstance(data, dict) else None
    cutoff = contacts.get("cutoff") if isinstance(contacts, dict) else None
    return isinstance(cutoff, str) and cutoff.strip().lower() == "auto"


def _stage_config_slice(config: BVWorkflowConfig, name: str) -> dict[str, Any]:
    """Config fields each stage actually depends on (ADR: seed edits refit only).

    Slices are transitive: a stage's slice contains every upstream stage's
    slice so a re-executed upstream stage always invalidates its consumers,
    while fit/validation-only changes (seed file, thresholds, split options)
    leave the expensive MACE stages reusable.
    """
    full = _config_to_dict(config, redact=False)
    structure_digest = (
        _atoms_digest(config.structure)
        if isinstance(config.structure, Atoms)
        else _file_digest(config.structure)
    )
    relax_slice = {
        "structure_digest": structure_digest,
        "model": full["model"],
        "model_path": full["model_path"],
        "relax": full["relax"],
    }
    if _seed_uses_auto_cutoff(config):
        # Auto cutoff resolution consumes the seed chemistry and the rattle/
        # strain margins, so relax outputs depend on them.
        relax_slice = {
            **relax_slice,
            "seed_file_digest": _file_digest(config.seed_file),
            "rattle_amplitudes": full["sampling"]["rattle_amplitudes"],
            "strain_range": full["sampling"]["strain_range"],
        }
    sampling_no_window = {
        key: value for key, value in full["sampling"].items() if key != "energy_window"
    }
    sample_slice = {**relax_slice, "seed": full["seed"], "sampling": sampling_no_window}
    evaluate_slice = {**sample_slice, "sampling": full["sampling"]}
    fit_slice = {
        **evaluate_slice,
        "seed_file_digest": _file_digest(config.seed_file),
        "fit": full["fit"],
    }
    validate_slice = {**fit_slice, "validation": full["validation"]}
    prepare_slice = {
        "structure_digest": structure_digest,
        "seed_file_digest": _file_digest(config.seed_file),
        "output_dir": full["output_dir"],
    }
    slices = {
        "prepare": prepare_slice,
        "relax": relax_slice,
        "sample": sample_slice,
        "evaluate": evaluate_slice,
        "fit": fit_slice,
        "validate": validate_slice,
        "report": validate_slice,
    }
    return slices[name]


def _json_safe_number(value: Any) -> Any:
    """Map non-finite numbers to None so json.dumps(allow_nan=False) survives."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _input_digests(config: BVWorkflowConfig) -> dict[str, str]:
    """Content digests of workflow inputs so in-place edits invalidate signatures."""
    digests: dict[str, str] = {}
    if isinstance(config.structure, Atoms):
        digests["structure"] = _atoms_digest(config.structure)
    else:
        digests["structure_file"] = _file_digest(config.structure)
    digests["seed_file"] = _file_digest(config.seed_file)
    return digests


def _atoms_digest(atoms: Atoms) -> str:
    data = {
        "numbers": [int(number) for number in atoms.numbers],
        "positions": np.asarray(atoms.positions, dtype=float)
        .round(decimals=12)
        .tolist(),
        "cell": np.asarray(atoms.cell, dtype=float).round(decimals=12).tolist(),
        "pbc": [bool(flag) for flag in atoms.pbc],
    }
    text = json.dumps(data, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_digest(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _config_to_dict(config: BVWorkflowConfig, *, redact: bool = True) -> dict[str, Any]:
    data = asdict(config)
    structure = data.get("structure")
    data["structure"] = (
        "<ase.Atoms>" if isinstance(structure, Atoms) else str(structure)
    )
    data["seed_file"] = str(data["seed_file"])
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


def _relativize_manifest_paths(
    context: BVWorkflowContext, value: Any, *, _key: str = ""
) -> Any:
    if _key == "config":
        return value
    if isinstance(value, str) and _looks_like_artifact(value):
        try:
            return context.relative_path(value)
        except Exception:
            return value
    if isinstance(value, dict):
        return {
            key: _relativize_manifest_paths(context, item, _key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_relativize_manifest_paths(context, item) for item in value]
    return value


def _looks_like_artifact(value: str) -> bool:
    suffixes = (
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".traj",
        ".vasp",
        ".poscar",
        ".xml",
        ".png",
        ".csv",
        ".nc",
    )
    return "/" in value or value.lower().endswith(suffixes)


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


def _write_manifest(context: BVWorkflowContext) -> Path:
    return _write_yaml(context.manifest, context.output_dir / "manifest.yaml")


def _default_stage_functions() -> dict[str, StageFunction]:
    return {
        "prepare": run_prepare_stage,
        "relax": run_relax_stage,
        "sample": run_sample_stage,
        "evaluate": run_evaluate_stage,
        "fit": run_fit_stage,
        "validate": run_validate_stage,
        "report": run_report_stage,
    }


def mlbvmodel_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate a bond-valence parameter model workflow."
    )
    parser.add_argument("structure", nargs="?", help="Parent structure file")
    parser.add_argument("--config", help="YAML workflow configuration")
    parser.add_argument(
        "--seed-file", default=None, help="Bond-valence seed parameter YAML (required)"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--relax-cell", action="store_true", default=None)
    parser.add_argument("--rattle-amplitudes", nargs="+", type=float, default=None)
    parser.add_argument("--frames-per-amplitude", type=int, default=None)
    parser.add_argument("--strain-range", nargs=2, type=float, default=None)
    parser.add_argument("--strain-points", type=int, default=None)
    parser.add_argument("--energy-window", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--gii-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-stage", action="append", default=[])
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default=None)
    args = parser.parse_args()
    try:
        config = _bv_config_from_cli(args)
        result = run_bondvalence_workflow(
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


def _bv_config_from_cli(args) -> BVWorkflowConfig:
    data: dict[str, Any] = {}
    if args.config:
        data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.structure:
        data["structure"] = args.structure
    flat = {
        "seed_file": args.seed_file,
        "model": args.model,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "seed": args.seed,
    }
    for key, value in flat.items():
        if value is not None:
            data[key] = value
    sections = {
        "relax": {
            "fmax": args.fmax,
            "relax_cell": args.relax_cell,
        },
        "sampling": {
            "rattle_amplitudes": args.rattle_amplitudes,
            "frames_per_amplitude": args.frames_per_amplitude,
            "strain_range": args.strain_range,
            "strain_points": args.strain_points,
            "energy_window": args.energy_window,
        },
        "fit": {
            "validation_fraction": args.validation_fraction,
        },
        "validation": {
            "gii_threshold": args.gii_threshold,
        },
    }
    for section, overrides in sections.items():
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if overrides:
            merged = dict(data.get(section) or {})
            merged.update(overrides)
            data[section] = merged
    if "seed_file" not in data or not data["seed_file"]:
        raise ValueError("--seed-file (or config seed_file) is required")
    if "structure" not in data or not data["structure"]:
        raise ValueError("a parent structure file is required")
    section_types = {
        "relax": RelaxStageConfig,
        "sampling": BVSamplingStageConfig,
        "fit": FitStageConfig,
        "validation": BVValidationStageConfig,
    }
    for section, cls in section_types.items():
        if section in data:
            data[section] = cls(**data[section])
    return BVWorkflowConfig(**data)


# ---------------------------------------------------------------------------
# Seed file loading (story-002)
# ---------------------------------------------------------------------------


def _require_easybondvalence():
    """Import easybondvalence lazily with an actionable error (ADR-003)."""
    try:
        import easybondvalence
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "easybondvalence is required for the bond-valence workflow; install "
            "it with `pip install atomchain[bondvalence]` or "
            "`uv pip install easybondvalence`"
        ) from exc
    return easybondvalence


@dataclass(frozen=True)
class BondValenceSeed:
    """Validated chemistry loaded from one seed YAML file (ADR-002)."""

    parameter_set: Any
    fit_parameters: tuple[Any, ...]
    contact_policy: Any
    oxidation_states: dict[str, float]
    provenance: str
    literature: dict[str, dict[str, float]]
    raw: dict[str, Any]
    warnings: tuple[str, ...] = ()
    cutoff_auto: bool = False
    cutoff_mode: str = "robust"

    def valences_for(self, species: Any) -> Any:
        easybondvalence = _require_easybondvalence()
        return easybondvalence.FormalValenceAssignment(
            [self.oxidation_states[str(item)] for item in species],
            source=self.provenance,
        )


def load_seed_file(path: str | Path, species: Any | None = None) -> BondValenceSeed:
    """Load and validate a seed YAML file, reporting all problems at once."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Seed file must be a YAML mapping: {path}")
    errors: list[str] = []
    warnings: list[str] = []

    def to_float(value: Any, label: str) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{label} must be a number")
            return None
        if not np.isfinite(result):
            errors.append(f"{label} must be finite")
            return None
        return result

    provenance = data.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        errors.append("provenance must be a non-empty string")

    oxidation_states: dict[str, float] = {}
    raw_states = data.get("oxidation_states")
    if not isinstance(raw_states, dict) or not raw_states:
        errors.append("oxidation_states must be a non-empty species -> number mapping")
    else:
        for element, value in raw_states.items():
            converted = to_float(value, f"oxidation state for {element}")
            if converted is not None:
                oxidation_states[str(element)] = converted
    if species is not None:
        for element in dict.fromkeys(str(item) for item in species):
            if element not in oxidation_states:
                errors.append(f"oxidation state missing for species {element}")

    contacts = data.get("contacts")
    cutoff = None
    cutoff_auto = False
    cutoff_mode = "robust"
    if not isinstance(contacts, dict):
        errors.append("contacts must be a mapping with a cutoff entry")
    else:
        raw_mode = contacts.get("cutoff_mode")
        if raw_mode is not None and raw_mode not in ("minimal", "robust"):
            errors.append("contacts.cutoff_mode must be 'minimal' or 'robust'")
        elif raw_mode is not None:
            cutoff_mode = str(raw_mode)
        raw_cutoff = contacts.get("cutoff")
        if isinstance(raw_cutoff, str) and raw_cutoff.strip().lower() == "auto":
            cutoff_auto = True
        else:
            cutoff = to_float(raw_cutoff, "contacts.cutoff")
            if cutoff is not None and cutoff <= 0.0:
                errors.append("contacts.cutoff must be finite and positive")
                cutoff = None

    raw_pairs = data.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        errors.append("pairs must be a non-empty list")
        raw_pairs = []

    easybondvalence = _require_easybondvalence()
    parameters: list[Any] = []
    plan_parameters: list[Any] = []
    pair_ids: list[str] = []
    selector_keys: set[frozenset] = set()
    for index, raw_pair in enumerate(raw_pairs):
        label = f"pairs[{index}]"
        if not isinstance(raw_pair, dict):
            errors.append(f"{label} must be a mapping")
            continue
        selector = raw_pair.get("selector")
        if (
            not isinstance(selector, (list, tuple))
            or len(selector) != 2
            or not all(isinstance(item, str) and item.strip() for item in selector)
        ):
            errors.append(f"{label}.selector must be [cation, anion] element symbols")
            continue
        center, neighbor = (item.strip() for item in selector)
        identifier = f"{center}-{neighbor}"
        selector_key = frozenset((center, neighbor))
        if selector_key in selector_keys:
            errors.append(
                f"{label}: duplicate selector {identifier} (a reversed pair already "
                "exists; selectors are orientation-insensitive)"
            )
            continue
        selector_keys.add(selector_key)
        pair_ids.append(identifier)
        specs: dict[str, dict[str, Any]] = {}
        for field_name in ("r0", "b"):
            spec = raw_pair.get(field_name)
            if not isinstance(spec, dict):
                errors.append(f"{label}.{field_name} must be a mapping with 'initial'")
                continue
            specs[field_name] = dict(spec)
        if len(specs) != 2:
            continue
        initial_r0 = to_float(specs["r0"].get("initial"), f"{identifier}:r0.initial")
        initial_b = to_float(specs["b"].get("initial"), f"{identifier}:b.initial")
        if initial_b is not None and initial_b <= 0.0:
            errors.append(f"{identifier}:b.initial must be strictly positive")
            initial_b = None
        if initial_r0 is not None and initial_b is not None:
            try:
                parameters.append(
                    easybondvalence.BondValenceParameter(
                        identifier=identifier,
                        selector=easybondvalence.ParameterSelector(center, neighbor),
                        r0=initial_r0,
                        b=initial_b,
                        source=(
                            provenance if isinstance(provenance, str) else "unvalidated"
                        ),
                    )
                )
            except Exception as exc:
                errors.append(f"{identifier}: {exc}")
        for field_name in ("r0", "b"):
            spec = specs[field_name]
            initial = {"r0": initial_r0, "b": initial_b}[field_name]
            tie = spec.get("tie")
            if spec.get("fixed") and tie is not None:
                errors.append(
                    f"{identifier}:{field_name} cannot be both fixed and tied"
                )
                continue
            if initial is None:
                continue
            transform = spec.get("transform")
            if transform is None:
                transform = "identity"
            elif transform not in ("identity", "positive"):
                errors.append(
                    f"{identifier}:{field_name}.transform must be 'identity' or "
                    f"'positive', got {transform!r}"
                )
                continue
            try:
                plan_parameters.append(
                    easybondvalence.FitParameter(
                        identifier=identifier,
                        field=field_name,
                        initial=initial,
                        lower=(
                            None
                            if spec.get("lower") is None
                            else to_float(
                                spec["lower"], f"{identifier}:{field_name}.lower"
                            )
                        ),
                        upper=(
                            None
                            if spec.get("upper") is None
                            else to_float(
                                spec["upper"], f"{identifier}:{field_name}.upper"
                            )
                        ),
                        transform=transform,
                        tie_to=(str(tie) if tie is not None else None),
                        status=(
                            "tied"
                            if tie is not None
                            else ("fixed" if spec.get("fixed") else "free")
                        ),
                    )
                )
            except Exception as exc:
                errors.append(f"{identifier}:{field_name}: {exc}")

    by_key = {item.key: item for item in plan_parameters}
    for item in plan_parameters:
        if item.status != "tied":
            continue
        target = by_key.get(item.tie_to or "")
        if target is None:
            errors.append(f"tie target {item.tie_to!r} for {item.key} does not exist")
            continue
        if target.status != "free":
            errors.append(
                f"tie target {item.tie_to!r} for {item.key} must be a free parameter"
            )
    if plan_parameters and not any(item.status == "free" for item in plan_parameters):
        errors.append("seed must declare at least one free parameter")

    if species is not None:
        present = set(str(item) for item in species)
        cations = sorted(
            element for element in present if oxidation_states.get(element, 0.0) > 0.0
        )
        anions = sorted(
            element for element in present if oxidation_states.get(element, 0.0) < 0.0
        )
        for cation in cations:
            for anion in anions:
                if not any(
                    {cation, anion} <= selector_key for selector_key in selector_keys
                ):
                    warnings.append(
                        f"structure contains {cation}-{anion} contacts but the seed "
                        "has no parameter for that pair"
                    )
        for selector_key in selector_keys:
            for element in selector_key:
                if element not in present:
                    pair_id = next(
                        pid for pid in pair_ids if set(pid.split("-")) >= {element}
                    )
                    warnings.append(
                        f"seed pair {pair_id} references {element}, which is not "
                        "present in the structure"
                    )

    literature: dict[str, dict[str, float]] = {}
    raw_literature = data.get("literature")
    if raw_literature is not None:
        if not isinstance(raw_literature, dict):
            errors.append("literature must be a pair -> {r0, b} mapping")
        else:
            for pair_id, values in raw_literature.items():
                if not isinstance(values, dict):
                    errors.append(f"literature[{pair_id}] must be a mapping")
                    continue
                converted: dict[str, float] = {}
                for field_name in ("r0", "b"):
                    if values.get(field_name) is None:
                        continue
                    number = to_float(
                        values[field_name], f"literature[{pair_id}].{field_name}"
                    )
                    if number is not None:
                        converted[field_name] = number
                if converted:
                    literature[str(pair_id)] = converted

    if errors:
        raise ValueError("Seed file validation failed:\n- " + "\n- ".join(errors))

    return BondValenceSeed(
        parameter_set=easybondvalence.ParameterSet(parameters),
        fit_parameters=tuple(plan_parameters),
        contact_policy=(
            easybondvalence.ContactPolicy(cutoff=cutoff) if not cutoff_auto else None
        ),
        oxidation_states=oxidation_states,
        provenance=provenance.strip(),
        literature=literature,
        raw=data,
        warnings=tuple(warnings),
        cutoff_auto=cutoff_auto,
        cutoff_mode=cutoff_mode,
    )


def _to_bv_structure(atoms: Atoms) -> Atoms:
    """Pass ASE Atoms through; easybondvalence consumes Atoms directly
    (and normalizes them internally)."""
    return atoms


def audit_contacts(
    seed: BondValenceSeed, atoms: Atoms, policy: Any = None
) -> list[str]:
    """Warn on zero-contact or non-integer-coordination pairs at this geometry.

    Selector matching is orientation-insensitive (mirroring
    ``ParameterSet.resolve``): a contact counts for a pair when its sites
    carry the pair's two elements in either orientation.
    """
    easybondvalence = _require_easybondvalence()
    effective = policy if policy is not None else seed.contact_policy
    if effective is None:
        raise ValueError("auto cutoff has not been resolved; pass a policy")
    structure = _to_bv_structure(atoms)
    contacts = easybondvalence.build_contacts(structure, effective)
    symbols = list(atoms.get_chemical_symbols())
    warnings: list[str] = []
    for parameter in seed.parameter_set.parameters:
        center = parameter.selector.center_element
        neighbor = parameter.selector.neighbor_element
        as_center = [
            contact
            for contact in contacts
            if symbols[contact.i] == center and symbols[contact.j] == neighbor
        ]
        as_neighbor = [
            contact
            for contact in contacts
            if symbols[contact.i] == neighbor and symbols[contact.j] == center
        ]
        total = len(as_center) + len(as_neighbor)
        if total == 0:
            warnings.append(
                f"pair {parameter.identifier} has zero contacts within cutoff "
                f"{effective.cutoff} at this geometry"
            )
            continue
        center_sites = {contact.i for contact in as_center} | {
            contact.j for contact in as_neighbor
        }
        mean = total / len(center_sites)
        if abs(mean - round(mean)) > 0.05:
            warnings.append(
                f"pair {parameter.identifier} has non-integer mean coordination "
                f"{mean:.2f} within the cutoff"
            )
    return warnings


# ---------------------------------------------------------------------------
# Smart contact-cutoff suggestion (story-011, research 2026-08-15)
# ---------------------------------------------------------------------------

SHELL_CLUSTER_TOLERANCE = 0.02
CUTOFF_THETA = 1e-3


@dataclass(frozen=True)
class ShellRecord:
    """One clustered contact shell with its seed-parameterized valence weight."""

    family: str
    distance: float
    multiplicity: int
    weight: float | None  # None when the family has no parameter


@dataclass(frozen=True)
class CutoffInterval:
    """A cutoff range clearing the flicker margin around every live shell."""

    lower: float
    upper: float
    representative: float
    families_inside: tuple[str, ...]
    missing_families: tuple[str, ...]

    @property
    def clearance(self) -> float:
        return 0.5 * (self.upper - self.lower)


@dataclass(frozen=True)
class CutoffSuggestion:
    sigma_max: float
    eps_max: float
    theta: float
    shells: tuple[ShellRecord, ...]
    live_shells: tuple[ShellRecord, ...]
    intervals: tuple[CutoffInterval, ...]
    minimal: CutoffInterval | None
    robust: CutoffInterval | None
    warnings: tuple[str, ...]


def suggest_cutoffs(
    atoms: Atoms,
    seed: BondValenceSeed,
    *,
    sigma_max: float,
    eps_max: float,
    theta: float = CUTOFF_THETA,
    r_max: float | None = None,
) -> CutoffSuggestion:
    """Suggest safe contact cutoffs from the shell structure (Proposal A).

    Enumerates contact shells on the relaxed structure, weights each by its
    total seed valence, keeps shells with weight >= theta as "live", and
    returns the cutoff intervals that clear the flicker margin
    m = 3*sqrt(2)*sigma_max + eps_max*d around every live shell on both
    sides. Each interval reports the parameter families it would sweep in
    and which of those the seed is missing.
    """
    easybondvalence = _require_easybondvalence()
    if r_max is None:
        if any(atoms.pbc):
            r_max = min(3.0 * float(np.max(atoms.cell.lengths())), 20.0)
        else:
            r_max = 12.0
    structure = _to_bv_structure(atoms)
    contacts = easybondvalence.build_contacts(
        structure, easybondvalence.ContactPolicy(cutoff=float(r_max))
    )
    symbols = list(atoms.get_chemical_symbols())
    seed_parameters: dict[frozenset, Any] = {}
    for parameter in seed.parameter_set.parameters:
        key = frozenset(
            (parameter.selector.center_element, parameter.selector.neighbor_element)
        )
        seed_parameters[key] = parameter

    family_distances: dict[frozenset, list[float]] = {}
    for contact in contacts:
        family = frozenset((symbols[contact.i], symbols[contact.j]))
        family_distances.setdefault(family, []).append(float(contact.distance))

    shells: list[ShellRecord] = []
    for family, distances in family_distances.items():
        distances.sort()
        clusters: list[list[float]] = []
        for distance in distances:
            if clusters and distance - clusters[-1][0] <= SHELL_CLUSTER_TOLERANCE:
                clusters[-1].append(distance)
            else:
                clusters.append([distance])
        parameter = seed_parameters.get(family)
        if parameter is not None:
            family_id = parameter.identifier
        elif len(family) == 1:
            family_id = f"{next(iter(family))}-{next(iter(family))}"
        else:
            family_id = "-".join(sorted(family))
        for cluster in clusters:
            representative = float(np.mean(cluster))
            weight = None
            if parameter is not None:
                weight = len(cluster) * float(
                    np.exp((parameter.r0 - representative) / parameter.b)
                )
            shells.append(
                ShellRecord(
                    family=family_id,
                    distance=representative,
                    multiplicity=len(cluster),
                    weight=weight,
                )
            )
    shells.sort(key=lambda shell: shell.distance)
    live = [
        shell for shell in shells if shell.weight is not None and shell.weight >= theta
    ]

    def margin(distance: float) -> float:
        return 3.0 * np.sqrt(2.0) * float(sigma_max) + float(eps_max) * distance

    warnings: list[str] = []
    intervals: list[CutoffInterval] = []
    if live:
        # The only region that includes every non-negligible shell with a
        # flicker-clearing margin is beyond the last live shell. Gaps between
        # live shells would truncate a non-negligible contribution and are
        # not offered.
        lower = live[-1].distance + margin(live[-1].distance)
        if lower < float(r_max):
            base = _make_interval(shells, lower, float(r_max), seed_parameters)
            intervals.append(base)
            # Minimal mode sits just inside the safe region rather than at
            # the midpoint; families are recomputed at its own representative.
            minimal_interval = _make_interval(
                shells,
                lower,
                lower + 0.2 * (float(r_max) - lower),
                seed_parameters,
            )
            intervals.insert(
                0,
                CutoffInterval(
                    lower=base.lower,
                    upper=base.upper,
                    representative=lower + 0.1 * (float(r_max) - lower),
                    families_inside=minimal_interval.families_inside,
                    missing_families=minimal_interval.missing_families,
                ),
            )
    if not intervals:
        warnings.append(
            f"no safe cutoff interval found within r_max={r_max:.2f} A: the "
            f"flicker margins (sigma_max={sigma_max}, eps_max={eps_max}) "
            "swallow every gap between live shells; reduce rattle amplitudes "
            "or the strain range"
        )

    minimal = (
        min(intervals, key=lambda interval: interval.representative)
        if intervals
        else None
    )
    robust = (
        max(
            intervals,
            key=lambda interval: (interval.clearance, interval.representative),
        )
        if intervals
        else None
    )
    return CutoffSuggestion(
        sigma_max=float(sigma_max),
        eps_max=float(eps_max),
        theta=float(theta),
        shells=tuple(shells),
        live_shells=tuple(live),
        intervals=tuple(intervals),
        minimal=minimal,
        robust=robust,
        warnings=tuple(warnings),
    )


def _make_interval(
    shells: list[ShellRecord],
    lower: float,
    upper: float,
    seed_parameters: dict[frozenset, Any],
) -> CutoffInterval:
    representative = 0.5 * (lower + upper)
    inside = sorted(
        {shell.family for shell in shells if shell.distance <= representative}
    )
    seed_families = {
        family_id
        for family_key, parameter in seed_parameters.items()
        for family_id in [parameter.identifier]
    }
    missing = tuple(family for family in inside if family not in seed_families)
    return CutoffInterval(
        lower=float(lower),
        upper=float(upper),
        representative=float(representative),
        families_inside=tuple(inside),
        missing_families=missing,
    )


def _resolve_contact_policy(
    context: BVWorkflowContext, seed: BondValenceSeed, atoms: Atoms
) -> Any:
    easybondvalence = _require_easybondvalence()
    sampling = context.config.sampling
    sigma_max = max((float(a) for a in sampling.rattle_amplitudes), default=0.0)
    lo, hi = (float(v) for v in sampling.strain_range)
    eps_max = max(abs(lo), abs(hi))
    suggestion = suggest_cutoffs(atoms, seed, sigma_max=sigma_max, eps_max=eps_max)
    context.cutoff_suggestion = suggestion
    interval = (
        suggestion.minimal
        if seed.cutoff_mode == "minimal"
        else (suggestion.robust or suggestion.minimal)
    )
    if interval is None:
        raise RuntimeError(
            "cutoff: auto found no safe interval ("
            + "; ".join(suggestion.warnings)
            + ")"
        )
    if interval.missing_families:
        raise RuntimeError(
            "cutoff: auto would include contact families without parameters: "
            + ", ".join(interval.missing_families)
            + "; add fixed near-zero pad parameters for them"
        )
    context.contact_policy = easybondvalence.ContactPolicy(
        cutoff=interval.representative
    )
    return context.contact_policy


# ---------------------------------------------------------------------------
# Minimal fit-and-reuse API (story-012)
# ---------------------------------------------------------------------------

BVMODEL_FORMAT = "atomchain-bv-model-v1"


@dataclass(frozen=True)
class SavedBondValenceModel:
    """A self-contained fitted bond-valence model loaded from disk."""

    parameter_set: Any
    contact_policy: Any
    oxidation_states: dict[str, float]
    provenance: str
    dataset_fingerprint: str | None = None

    def valences_for(self, species) -> Any:
        easybondvalence = _require_easybondvalence()
        missing = sorted({str(item) for item in species} - set(self.oxidation_states))
        if missing:
            raise ValueError(
                "the saved model has no oxidation states for: "
                + ", ".join(missing)
                + "; refit with a seed whose oxidation_states cover these "
                "species"
            )
        return easybondvalence.FormalValenceAssignment(
            [self.oxidation_states[str(item)] for item in species],
            source=self.provenance,
        )


def _write_fitted_parameters(result: Any, provenance: str) -> dict[str, Any]:
    """Self-contained fitted-model document (story-012 schema)."""
    return {
        "format": BVMODEL_FORMAT,
        "pairs": {
            parameter.identifier: {
                "selector": [
                    parameter.selector.center_element,
                    parameter.selector.neighbor_element,
                ],
                "r0": parameter.r0,
                "b": parameter.b,
                "source": parameter.source,
            }
            for parameter in result.parameter_set.parameters
        },
        "contacts": {"cutoff": None},  # filled by the caller (resolved policy)
        "oxidation_states": None,  # filled by the caller (seed map)
        "dataset_fingerprint": result.dataset_fingerprint,
        "provenance": provenance,
    }


def load_bond_valence_model(path: str | Path) -> SavedBondValenceModel:
    """Load a fitted bond-valence model saved by the workflow."""
    easybondvalence = _require_easybondvalence()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or data.get("format") != BVMODEL_FORMAT:
        raise ValueError(f"not an {BVMODEL_FORMAT} model file: {path}")
    pairs = data.get("pairs")
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("model file has no pairs section")
    contacts = data.get("contacts")
    cutoff = contacts.get("cutoff") if isinstance(contacts, dict) else None
    if cutoff is None:
        raise ValueError("model file is missing contacts.cutoff")
    raw_states = data.get("oxidation_states")
    if not isinstance(raw_states, dict) or not raw_states:
        raise ValueError("model file is missing oxidation_states")
    oxidation_states = {
        str(element): float(value) for element, value in raw_states.items()
    }
    parameters = []
    for identifier, pair in pairs.items():
        if not isinstance(pair, dict):
            raise ValueError(f"model file pair {identifier!r} must be a mapping")
        selector = pair.get("selector")
        if (
            not isinstance(selector, (list, tuple))
            or isinstance(selector, str)
            or len(selector) != 2
        ):
            raise ValueError(
                f"model file pair {identifier!r} needs a two-element "
                f"[center, neighbor] selector"
            )
        try:
            r0 = float(pair["r0"])
            b = float(pair["b"])
        except KeyError as exc:
            raise ValueError(
                f"model file pair {identifier!r} is missing field {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model file pair {identifier!r} has non-numeric r0/b: {exc}"
            ) from exc
        parameters.append(
            easybondvalence.BondValenceParameter(
                identifier=identifier,
                selector=easybondvalence.ParameterSelector(*selector),
                r0=r0,
                b=b,
                source=str(pair.get("source") or "unspecified"),
            )
        )
    return SavedBondValenceModel(
        parameter_set=easybondvalence.ParameterSet(parameters),
        contact_policy=easybondvalence.ContactPolicy(cutoff=float(cutoff)),
        oxidation_states=oxidation_states,
        provenance=str(data.get("provenance") or "unspecified"),
        dataset_fingerprint=data.get("dataset_fingerprint"),
    )


def evaluate_bond_valence(
    model: SavedBondValenceModel, atoms: Atoms, *, max_rank: int = 4
) -> dict[str, Any]:
    """Evaluate a saved model on a structure: site sums, GII, multipoles."""
    easybondvalence = _require_easybondvalence()
    species = atoms.get_chemical_symbols()
    structure = _to_bv_structure(atoms)
    valences = model.valences_for(species)
    structural = easybondvalence.evaluate(
        model.parameter_set,
        structure,
        contacts=model.contact_policy,
        valences=valences,
    )
    mismatches = np.asarray(structural.mismatches, dtype=float)
    per_site = [
        {
            "index": site,
            "element": species[site],
            "bond_valence_sum": float(structural.site_sums[site]),
            "target": float(valences.target_values[site]),
            "mismatch": float(mismatches[site]),
        }
        for site in range(len(species))
    ]
    expansion = easybondvalence.expand_multipoles(structural, max_rank=max_rank)
    return {
        "gii": float(structural.gii),
        "site_sums": np.asarray(structural.site_sums).tolist(),
        "mismatches": mismatches.tolist(),
        "per_site": per_site,
        "dipoles": np.asarray(expansion.dipoles).tolist(),
        "quadrupoles": np.asarray(expansion.quadrupoles).tolist(),
        "invariants": np.asarray(expansion.invariants).tolist(),
        "max_rank": max_rank,
        "units": (
            "bond valences, site sums, mismatches, GII, dipoles, and "
            "quadrupoles in valence units (v.u.); normalized coefficients "
            "and invariants are dimensionless; descriptors only, no energy "
            "model"
        ),
    }
