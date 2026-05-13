"""Validation helpers for fitted MULTIBINIT/ASE-compatible models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from atomchain.init_model import init_calc
from atomchain.io.hist import read_abinit_hist
from atomchain.training.evaluate import evaluate_training_frames
from atomchain.training.samplers import sample_metastable_frames


@dataclass
class ArrayMetric:
    """Aggregate error metrics for one predicted quantity."""

    count: int
    mae: float
    rmse: float
    max_abs: float

    def to_dict(self):
        return asdict(self)


@dataclass
class FittedModelValidationResult:
    """Validation summary for a fitted model on a test set."""

    nframes: int
    metrics: dict[str, ArrayMetric]
    frame_errors: list[dict]
    evaluated_trajectory: str | None = None

    def to_dict(self):
        data = asdict(self)
        data["metrics"] = {key: value.to_dict() for key, value in self.metrics.items()}
        return data


@dataclass
class MetastableEnergyValidationResult:
    """Validation summary for metastable energy differences."""

    nstates: int
    reference_energy: float
    model_reference_energy: float
    metric: ArrayMetric
    states: list[dict]

    def to_dict(self):
        data = asdict(self)
        data["metric"] = self.metric.to_dict()
        return data


def validate_fitted_model(
    test_set,
    calculator="multibinit",
    model_path=None,
    output=None,
    evaluated_trajectory=None,
    include_stress=True,
    strict_stress=False,
):
    """Evaluate a fitted model on a test set and compare energy/forces/stress.

    ``test_set`` may be an ASE trajectory path, an ABINIT ``HIST.nc`` path, a
    single ASE ``Atoms`` object, or an iterable of ASE ``Atoms`` objects whose
    calculators already contain reference energy/force/stress data.
    """
    reference_frames = load_test_frames(test_set)
    model_frames = evaluate_training_frames(
        reference_frames,
        calculator=_coerce_calculator(calculator, model_path=model_path),
        output=evaluated_trajectory,
        strict_stress=strict_stress,
        provenance={"validation_role": "fitted_model_prediction"},
    )
    result = compare_evaluated_frames(
        reference_frames, model_frames, include_stress=include_stress
    )
    if evaluated_trajectory is not None:
        result.evaluated_trajectory = str(evaluated_trajectory)
    if output is not None:
        _write_json(result.to_dict(), output)
    return result


def compare_evaluated_frames(reference_frames, model_frames, include_stress=True):
    """Compare two evaluated ASE frame lists."""
    reference_frames = list(reference_frames)
    model_frames = list(model_frames)
    if len(reference_frames) != len(model_frames):
        raise ValueError("reference and model frame counts differ")

    energy_errors = []
    force_errors = []
    stress_errors = []
    frame_errors = []
    for iframe, (ref, model) in enumerate(zip(reference_frames, model_frames)):
        _ensure_compatible(ref, model, iframe)
        ref_energy = float(ref.get_potential_energy())
        model_energy = float(model.get_potential_energy())
        ref_forces = np.asarray(ref.get_forces(), dtype=float)
        model_forces = np.asarray(model.get_forces(), dtype=float)
        energy_error = model_energy - ref_energy
        forces_error = model_forces - ref_forces
        energy_errors.append(energy_error)
        force_errors.append(forces_error.reshape(-1))
        item = {
            "frame_index": iframe,
            "energy_error_ev": float(energy_error),
            "max_force_error_ev_ang": float(np.max(np.abs(forces_error)))
            if forces_error.size
            else 0.0,
        }
        if include_stress:
            try:
                ref_stress = np.asarray(ref.get_stress(voigt=True), dtype=float)
                model_stress = np.asarray(model.get_stress(voigt=True), dtype=float)
            except Exception:
                ref_stress = None
                model_stress = None
            if ref_stress is not None and model_stress is not None:
                stress_error = model_stress - ref_stress
                stress_errors.append(stress_error.reshape(-1))
                item["max_stress_error_ev_ang3"] = float(np.max(np.abs(stress_error)))
        frame_errors.append(item)

    metrics = {
        "energy_ev": _metric(np.asarray(energy_errors, dtype=float)),
        "forces_ev_ang": _metric(
            np.concatenate(force_errors) if force_errors else np.array([], dtype=float)
        ),
    }
    if include_stress and stress_errors:
        metrics["stress_ev_ang3"] = _metric(np.concatenate(stress_errors))
    return FittedModelValidationResult(
        nframes=len(reference_frames), metrics=metrics, frame_errors=frame_errors
    )


def validate_metastable_energy_differences(
    reference_state,
    metastable_states,
    calculator="multibinit",
    model_path=None,
    reference_calculator=None,
    output=None,
):
    """Compare metastable energy differences against original/reference data.

    The comparison is performed for ``E(state) - E(reference_state)``. If
    ``reference_calculator`` is supplied, it is used to compute the original
    reference energies; otherwise existing single-point data on the input atoms
    are used.
    """
    ref = load_test_frames(reference_state)[0]
    states = _load_metastable_states(metastable_states)
    if reference_calculator is not None:
        reference_evaluated = evaluate_training_frames(
            [ref, *states], calculator=reference_calculator, strict_stress=False
        )
        ref = reference_evaluated[0]
        states = reference_evaluated[1:]

    model_frames = evaluate_training_frames(
        [ref, *states],
        calculator=_coerce_calculator(calculator, model_path=model_path),
        strict_stress=False,
        provenance={"validation_role": "metastable_model_prediction"},
    )
    ref_energy = float(ref.get_potential_energy())
    model_ref_energy = float(model_frames[0].get_potential_energy())
    rows = []
    delta_errors = []
    for index, (state, model_state) in enumerate(zip(states, model_frames[1:])):
        reference_delta = float(state.get_potential_energy()) - ref_energy
        model_delta = float(model_state.get_potential_energy()) - model_ref_energy
        error = model_delta - reference_delta
        delta_errors.append(error)
        rows.append(
            {
                "state_index": index,
                "metastable_id": state.info.get("metastable_id"),
                "combined_label": state.info.get("combined_label"),
                "reference_delta_ev": reference_delta,
                "model_delta_ev": model_delta,
                "delta_error_ev": error,
            }
        )
    result = MetastableEnergyValidationResult(
        nstates=len(states),
        reference_energy=ref_energy,
        model_reference_energy=model_ref_energy,
        metric=_metric(np.asarray(delta_errors, dtype=float)),
        states=rows,
    )
    if output is not None:
        _write_json(result.to_dict(), output)
    return result


def load_test_frames(test_set):
    """Load ASE/HIST/Atoms test frames into a list."""
    if isinstance(test_set, Atoms):
        return [test_set]
    if isinstance(test_set, (str, Path)):
        path = Path(test_set)
        if path.suffix == ".nc" or "HIST" in path.name.upper():
            return read_abinit_hist(path)
        loaded = read(str(path), ":")
        return [loaded] if isinstance(loaded, Atoms) else list(loaded)
    return list(test_set)


def _load_metastable_states(value):
    if isinstance(value, (str, Path, Atoms)):
        return load_test_frames(value)
    if isinstance(value, dict):
        return sample_metastable_frames(value)
    value = list(value)
    if value and isinstance(value[0], dict):
        return sample_metastable_frames(value)
    return value


def _coerce_calculator(calculator, model_path=None):
    if isinstance(calculator, str):
        return init_calc(model_type=calculator, model_path=model_path)
    return calculator


def _metric(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return ArrayMetric(count=0, mae=0.0, rmse=0.0, max_abs=0.0)
    return ArrayMetric(
        count=int(values.size),
        mae=float(np.mean(np.abs(values))),
        rmse=float(np.sqrt(np.mean(values**2))),
        max_abs=float(np.max(np.abs(values))),
    )


def _ensure_compatible(reference, model, iframe):
    if len(reference) != len(model):
        raise ValueError(f"Frame {iframe} has different atom count")
    if reference.get_chemical_symbols() != model.get_chemical_symbols():
        raise ValueError(f"Frame {iframe} has different atom ordering or symbols")


def _write_json(data, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
