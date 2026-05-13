"""DDB + HIST artifact bundle generation."""

from __future__ import annotations

from pathlib import Path

import yaml
from ase import Atoms
from ase.io import read, write

from atomchain.ddb import write_ddb_from_phonopy
from atomchain.io.hist import write_abinit_hist
from atomchain.training.evaluate import evaluate_training_frames


def generate_multibinit_training_artifacts(
    structure,
    trajectory,
    model="mace-r2scan",
    ddb="out.ddb",
    hist="out_HIST.nc",
    phonopy_yaml=None,
    output_dir=None,
    calculator=None,
    metadata="training_artifacts.yaml",
    evaluate=False,
):
    """Generate a reproducible DDB + HIST training artifact bundle."""
    outdir = Path(output_dir) if output_dir is not None else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)
    ddb_path = outdir / ddb
    hist_path = outdir / hist
    metadata_path = outdir / metadata
    atoms = (
        read(str(structure)) if not hasattr(structure, "get_positions") else structure
    )
    frames = _load_frames(trajectory)
    if evaluate:
        evaluated = evaluate_training_frames(
            frames, calculator=calculator or model, output=outdir / "training.traj"
        )
    else:
        evaluated = list(frames)
        write(str(outdir / "training.traj"), evaluated)
    write_abinit_hist(evaluated, hist_path, metadata=f"{hist_path}.yaml")
    ddb_written = False
    if phonopy_yaml is not None:
        write_ddb_from_phonopy(
            atoms=atoms, phonopy_yaml=phonopy_yaml, filename=ddb_path
        )
        ddb_written = True
    data = {
        "format": "atomchain-multibinit-training-bundle-v1",
        "structure": str(structure),
        "trajectory": str(trajectory),
        "model": model,
        "ddb": str(ddb_path) if ddb_written else None,
        "ddb_requested_path": str(ddb_path),
        "ddb_written": ddb_written,
        "hist": str(hist_path),
        "evaluated_trajectory": str(outdir / "training.traj"),
        "phonopy_yaml": str(phonopy_yaml) if phonopy_yaml is not None else None,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)
    return data


def _load_frames(trajectory):
    if isinstance(trajectory, (str, Path)):
        loaded = read(str(trajectory), ":")
        return [loaded] if isinstance(loaded, Atoms) else list(loaded)
    if isinstance(trajectory, Atoms):
        return [trajectory]
    return list(trajectory)
