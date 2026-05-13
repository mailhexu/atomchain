"""Evaluate training frames and attach provenance metadata."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from atomchain.init_model import init_calc


def evaluate_training_frames(
    frames,
    calculator="chgnet",
    model_path=None,
    output=None,
    strict_stress=True,
    provenance=None,
):
    """Evaluate frames with a calculator and attach energy, forces, stress."""
    frames = list(frames)
    if isinstance(calculator, str):
        calc = init_calc(model_type=calculator, model_path=model_path)
        calculator_name = calculator
    else:
        calc = calculator
        calculator_name = type(calculator).__name__
    evaluated = []
    for iframe, frame in enumerate(frames):
        atoms = frame.copy()
        atoms.info.update(dict(provenance or {}))
        atoms.info.setdefault("frame_index", iframe)
        atoms.info.setdefault("calculator", calculator_name)
        atoms.calc = calc
        try:
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            stress = atoms.get_stress(voigt=True)
        except Exception as exc:
            if strict_stress:
                raise RuntimeError(
                    f"Could not evaluate energy/forces/stress for frame {iframe}"
                ) from exc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            stress = None
        kwargs = {"energy": energy, "forces": forces}
        if stress is not None:
            kwargs["stress"] = stress
        atoms.calc = SinglePointCalculator(atoms, **kwargs)
        evaluated.append(atoms)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write(str(output), evaluated)
    return evaluated
