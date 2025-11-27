#!/usr/bin/env python
"""
Rattle dataset generation for ML potential training.

This module provides functionality to generate training datasets by creating
multiple structures with random atomic displacements and optional cell deformations.
Structures are saved to trajectory files for later property calculations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from ase import Atoms
from ase.io import Trajectory, read

from atomchain.supercell import make_supercell_structure


def generate_rattle_dataset(
    atoms: Union[Atoms, str],
    n_struct: int = 100,
    stdev: float = 0.05,
    supercell: Optional[Union[int, List[int]]] = None,
    cell_stdev: Optional[float] = None,
    seed: Optional[int] = None,
    output: str = "rattle_dataset.traj",
    verbose: bool = True,
) -> str:
    """
    Generate training dataset with random atomic displacements and cell deformations.

    This function creates multiple structures by applying random displacements to
    atomic positions (rattle) and optionally random strain to cell vectors.
    Structures are saved to trajectory file without property calculations.

    Args:
        atoms: ASE Atoms object or path to structure file (POSCAR, CIF, XYZ, etc.)
        n_struct: Number of structures to generate (default: 100)
        stdev: Standard deviation for atomic displacements in Ångström (default: 0.05)
        supercell: Supercell specification applied before rattling. Can be:
                   - int: Isotropic scaling (e.g., 2 → 2×2×2)
                   - List[int]: Diagonal scaling (e.g., [2,2,3] → 2×2×3)
                   - None: No supercell generation (default)
        cell_stdev: Standard deviation for cell strain (default: None, no cell deformation)
                    Recommended range: 0.01-0.05 for typical materials
        seed: Random seed for reproducibility (default: None)
        output: Output trajectory file path (default: 'rattle_dataset.traj')
        verbose: Print progress messages (default: True)

    Returns:
        str: Path to output trajectory file

    Raises:
        FileNotFoundError: If structure file path does not exist
        ValueError: If n_struct < 1 or stdev/cell_stdev are negative

    Example:
        >>> # Basic dataset generation
        >>> from ase.build import bulk
        >>> atoms = bulk('Al', 'fcc', a=4.05)
        >>> traj_path = generate_rattle_dataset(atoms, n_struct=50, stdev=0.05)

        >>> # With supercell and cell strain
        >>> traj_path = generate_rattle_dataset(
        ...     'POSCAR',
        ...     n_struct=100,
        ...     supercell=[2,2,2],
        ...     cell_stdev=0.02,
        ...     seed=42
        ... )

        >>> # Read generated dataset
        >>> from ase.io import read
        >>> structures = read(traj_path, ':')
        >>> print(f"Generated {len(structures)} structures")
    """
    # Validation
    if n_struct < 1:
        raise ValueError(f"n_struct must be >= 1, got {n_struct}")
    if stdev < 0:
        raise ValueError(f"stdev must be non-negative, got {stdev}")
    if cell_stdev is not None and cell_stdev < 0:
        raise ValueError(f"cell_stdev must be non-negative, got {cell_stdev}")

    # Load structure from file if string is provided
    base_atoms: Atoms
    if isinstance(atoms, str):
        loaded = read(atoms)
        if isinstance(loaded, list):
            base_atoms = loaded[0]
        else:
            base_atoms = loaded
    else:
        base_atoms = atoms

    # Create supercell if requested
    if supercell is not None:
        if verbose:
            print(f"[Rattle] Creating supercell: {supercell}")
        base_atoms = make_supercell_structure(base_atoms, supercell)

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Initialize trajectory file
    if verbose:
        print(f"[Rattle] Generating {n_struct} structures")
        print(f"[Rattle] Atomic displacement stdev: {stdev} Å")
        if cell_stdev is not None:
            print(f"[Rattle] Cell strain stdev: {cell_stdev}")
        print(f"[Rattle] Output: {output}")

    traj = Trajectory(output, "w")

    # Progress reporting setup
    progress_interval = max(1, n_struct // 10)  # Report every 10%

    # Generate structures
    for i in range(n_struct):
        # Progress reporting
        if verbose and (i + 1) % progress_interval == 0:
            percent = 100 * (i + 1) / n_struct
            print(f"[Rattle] Progress: {i + 1}/{n_struct} ({percent:.0f}%)")

        # Create copy and apply random displacements
        rattled_atoms = base_atoms.copy()

        # Apply cell strain if requested
        if cell_stdev is not None:
            rattled_atoms = _apply_random_strain(rattled_atoms, cell_stdev)

        # Apply atomic displacements using ASE rattle method
        # Note: rattle modifies atoms in-place
        # Use rng=np.random to avoid ASE's default seed=42 when seed=None
        rattled_atoms.rattle(stdev=stdev, rng=np.random if seed is None else None)

        # Write to trajectory
        traj.write(rattled_atoms)

    # Close trajectory file
    traj.close()

    # Final summary
    if verbose:
        print("\n[Rattle] Dataset generation complete!")
        print(f"[Rattle] Successfully generated: {n_struct} structures")
        print(f"[Rattle] Output saved to: {output}")

    return output


def _apply_random_strain(atoms: Atoms, cell_stdev: float) -> Atoms:
    """
    Apply random symmetric strain to cell vectors.

    Generates a random symmetric 3×3 strain tensor and applies it to the cell
    while scaling atomic positions accordingly.

    Args:
        atoms: Structure to strain (modified in-place)
        cell_stdev: Standard deviation of strain components

    Returns:
        Strained Atoms object (same as input, modified in-place)
    """
    # Generate random symmetric strain tensor
    # Start with random 3×3 matrix
    random_matrix = np.random.normal(0, cell_stdev, (3, 3))

    # Make it symmetric: ε = (A + A^T) / 2
    strain = (random_matrix + random_matrix.T) / 2

    # Apply strain: new_cell = (I + ε) @ old_cell
    identity = np.eye(3)
    deformation = identity + strain

    old_cell = atoms.get_cell()
    new_cell = deformation @ old_cell

    # Set new cell and scale positions
    atoms.set_cell(new_cell, scale_atoms=True)

    return atoms


def mlrattle_cli():
    """
    Command-line interface for rattle dataset generation.

    This CLI tool allows users to generate training datasets with random
    atomic displacements and optional cell deformations from the command line.

    Example:
        $ mlrattle structure.cif --nstruct 100 --stdev 0.05 -o dataset.traj
        $ mlrattle POSCAR --nstruct 50 --supercell 2,2,2
        $ mlrattle input.cif --nstruct 100 --stdev 0.05 --cell-stdev 0.02 --seed 42
    """
    parser = argparse.ArgumentParser(
        description="Generate training datasets with random atomic displacements."
    )
    parser.add_argument("fname", help="Input structure file (POSCAR, CIF, XYZ, etc.)")
    parser.add_argument(
        "--nstruct",
        "-n",
        type=int,
        default=100,
        help="Number of structures to generate (default: 100)",
    )
    parser.add_argument(
        "--stdev",
        "-s",
        type=float,
        default=0.05,
        help="Standard deviation for atomic displacements in Å (default: 0.05)",
    )
    parser.add_argument(
        "--supercell",
        "-sc",
        type=str,
        default=None,
        help="Supercell specification: single int (2) or diagonal (2,2,3)",
    )
    parser.add_argument(
        "--cell-stdev",
        "-cs",
        type=float,
        default=None,
        help="Standard deviation for cell strain (default: None, no cell deformation)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="rattle_dataset.traj",
        help="Output trajectory file (default: rattle_dataset.traj)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Parse supercell specification
    supercell: Optional[Union[int, List[int]]] = None
    if args.supercell is not None:
        try:
            # Try parsing as single integer
            if "," not in args.supercell:
                supercell = int(args.supercell)
            else:
                # Parse as diagonal list
                supercell = [int(x.strip()) for x in args.supercell.split(",")]
                if len(supercell) != 3:
                    raise ValueError("Diagonal supercell must have exactly 3 values")
        except ValueError as e:
            parser.error(f"Invalid supercell format: {e}")

    # Print header
    if not args.quiet:
        print("=" * 60)
        print("Rattle Dataset Generation")
        print("=" * 60)
        print(f"Input structure: {args.fname}")
        print(f"Structures:      {args.nstruct}")
        print(f"Atom stdev:      {args.stdev} Å")
        if args.cell_stdev is not None:
            print(f"Cell stdev:      {args.cell_stdev}")
        if supercell is not None:
            print(f"Supercell:       {supercell}")
        if args.seed is not None:
            print(f"Random seed:     {args.seed}")
        print("=" * 60)
        print()

    # Generate dataset
    try:
        output_path = generate_rattle_dataset(
            atoms=args.fname,
            n_struct=args.nstruct,
            stdev=args.stdev,
            supercell=supercell,
            cell_stdev=args.cell_stdev,
            seed=args.seed,
            output=args.output,
            verbose=not args.quiet,
        )

        # Print final summary
        if not args.quiet:
            structures = read(output_path, ":")
            if not isinstance(structures, list):
                structures = [structures]

            print("\n" + "=" * 60)
            print("Dataset Summary")
            print("=" * 60)
            print(f"Output file:     {output_path}")
            print(f"File size:       {Path(output_path).stat().st_size / 1024:.1f} KB")
            print(f"Structures:      {len(structures)}")
            print(f"Atoms per struct: {len(structures[0])}")
            print("=" * 60)

    except Exception as e:
        parser.error(f"Dataset generation failed: {e}")


if __name__ == "__main__":
    mlrattle_cli()
