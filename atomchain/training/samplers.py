"""Structure samplers for MULTIBINIT training trajectories."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from atomchain.training.evaluate import evaluate_training_frames


def generate_training_trajectory(
    parent,
    model="chgnet",
    sources=None,
    output=None,
    evaluate=True,
    calculator=None,
    **kwargs,
):
    """Generate a training trajectory from one or more source strategies."""
    atoms = _read_atoms(parent)
    sources = sources or ["phonon_modes"]
    frames = []
    for source in sources:
        if source == "md":
            frames.extend(sample_md_frames(atoms, **kwargs.get("md", {})))
        elif source == "phonon_modes":
            frames.extend(
                sample_phonon_mode_frames(atoms, **kwargs.get("phonon_modes", {}))
            )
        elif source == "metastable":
            frames.extend(
                sample_metastable_frames(
                    kwargs.get("metastable_results", []), parent_atoms=atoms
                )
            )
        elif source in {
            "metastable_linear_combinations",
            "metastable_linear_combination",
        }:
            frames.extend(
                sample_metastable_linear_combinations(
                    atoms,
                    kwargs.get("metastable_results", []),
                    **kwargs.get("metastable_linear_combinations", {}),
                )
            )
        else:
            raise ValueError(f"Unknown training trajectory source: {source}")
    if evaluate:
        frames = evaluate_training_frames(
            frames, calculator=calculator or model, output=output
        )
    elif output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write(str(output), frames)
    return frames


def sample_md_frames(
    atoms,
    steps=10,
    displacement_stdev=0.01,
    seed=None,
    interval=1,
    calc=None,
    model_path=None,
    temperature=300.0,
    timestep=1.0,
    **_kwargs,
):
    """Sample MD frames, using atomchain MD when a calculator is provided.

    If ``calc`` is omitted, a deterministic perturbation fallback is used for
    lightweight tests and dry-run dataset design.
    """
    if steps < 0 or interval <= 0:
        raise ValueError("steps must be non-negative and interval must be positive")
    if calc is not None and steps > 0:
        from atomchain.md import md_nvt_langevin

        with tempfile.TemporaryDirectory() as tmpdir:
            traj = Path(tmpdir) / "md.traj"
            md_nvt_langevin(
                atoms,
                calc=calc,
                model_path=model_path,
                timestep=timestep,
                steps=steps,
                temperature=temperature,
                trajectory=str(traj),
                logfile=None,
                loginterval=interval,
            )
            frames = read(str(traj), ":") if traj.exists() else []
        for iframe, frame in enumerate(frames):
            frame.info.update(
                {
                    "source": "md",
                    "md_step": iframe * interval,
                    "temperature": temperature,
                    "timestep": timestep,
                    "random_seed": seed,
                }
            )
        return frames
    rng = np.random.default_rng(seed)
    frames = []
    for step in range(0, steps + 1, interval):
        frame = atoms.copy()
        if step:
            frame.positions += rng.normal(
                scale=displacement_stdev, size=frame.positions.shape
            )
        frame.info.update(
            {
                "source": "md",
                "md_step": step,
                "displacement_stdev": displacement_stdev,
                "random_seed": seed,
            }
        )
        frames.append(frame)
    return frames


def sample_phonon_mode_frames(
    atoms,
    amplitudes=(-0.05, 0.05),
    atom_indices=None,
    directions=(0, 1, 2),
    mode_label="cartesian",
):
    """Generate simple phonon-mode-like Cartesian displacement frames."""
    frames = []
    atom_indices = list(atom_indices) if atom_indices is not None else [0]
    for atom_index in atom_indices:
        for direction in directions:
            for amplitude in amplitudes:
                frame = atoms.copy()
                frame.positions[atom_index, int(direction)] += float(amplitude)
                frame.info.update(
                    {
                        "source": "phonon_modes",
                        "mode_label": mode_label,
                        "atom_index": int(atom_index),
                        "direction": int(direction),
                        "amplitude": float(amplitude),
                    }
                )
                frames.append(frame)
    return frames


def sample_metastable_frames(
    metastable_results, parent_atoms=None, include_relaxed=True
):
    """Convert metastable exploration results to training frames."""
    frames = []
    results = (
        metastable_results.get("results", metastable_results)
        if isinstance(metastable_results, dict)
        else metastable_results
    )
    for result in results or []:
        atoms = result.get("atoms") if isinstance(result, dict) else None
        if atoms is None and parent_atoms is not None:
            atoms = parent_atoms.copy()
        if atoms is None:
            continue
        frame = atoms.copy()
        frame.info.update(
            {
                "source": "metastable",
                "metastable_id": result.get("id") if isinstance(result, dict) else None,
                "combined_label": result.get("combined_label")
                if isinstance(result, dict)
                else None,
                "include_relaxed": include_relaxed,
            }
        )
        frames.append(frame)
    return frames


def sample_metastable_linear_combinations(
    parent_atoms, metastable_results, weights=(0.25, 0.5, 0.75)
):
    """Interpolate parent positions toward metastable structures with same mapping."""
    frames = []
    parent = parent_atoms.copy()
    for meta in sample_metastable_frames(metastable_results):
        _ensure_compatible(parent, meta)
        displacement = meta.positions - parent.positions
        for weight in weights:
            frame = parent.copy()
            frame.positions = parent.positions + float(weight) * displacement
            frame.info.update(
                {
                    "source": "metastable_linear_combination",
                    "metastable_id": meta.info.get("metastable_id"),
                    "combined_label": meta.info.get("combined_label"),
                    "weight": float(weight),
                }
            )
            frames.append(frame)
    return frames


def _read_atoms(value):
    if isinstance(value, Atoms):
        return value
    return read(str(value))


def _ensure_compatible(a, b):
    if len(a) != len(b) or a.get_chemical_symbols() != b.get_chemical_symbols():
        raise ValueError("Metastable structures must have matching atoms and ordering")
