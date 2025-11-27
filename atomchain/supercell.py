#!/usr/bin/env python
"""
Supercell generation using ML potentials and ASE.

This module provides functionality to generate supercells from primitive
structures with support for diagonal, isotropic, and general matrix transformations.
"""

from __future__ import annotations

import argparse
from typing import List, Union

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import read, write


def make_supercell_structure(
    atoms: Union[Atoms, str],
    P: Union[int, List[int], List[List[int]], np.ndarray],
    wrap: bool = True,
    order: str = "cell-major",
) -> Atoms:
    """
    Generate a supercell from a primitive structure.

    This function wraps ASE's make_supercell with convenient input handling
    for common use cases. It supports diagonal supercells, isotropic scaling,
    and general transformation matrices.

    Args:
        atoms: ASE Atoms object or path to structure file (POSCAR, CIF, XYZ, etc.)
        P: Transformation specification, can be:
           - int: Isotropic scaling (e.g., 2 → 2×2×2 supercell)
           - List[int] of length 3: Diagonal scaling (e.g., [2,2,3] → 2×2×3)
           - 3×3 matrix: General transformation matrix
        wrap: If True, wrap atomic positions into the supercell (default: True)
        order: Atom ordering in supercell. Options:
               - "cell-major": [atom1_cell1, atom2_cell1, ..., atom1_cell2, ...]
               - "atom-major": [atom1_cell1, atom1_cell2, ..., atom2_cell1, ...]
               (default: "cell-major")

    Returns:
        Atoms: Supercell structure with N × det(P) atoms

    Raises:
        ValueError: If matrix is not integer, or has non-positive determinant
        FileNotFoundError: If structure file path does not exist

    Example:
        >>> # Isotropic 2×2×2 supercell
        >>> from ase.build import bulk
        >>> atoms = bulk('Al', 'fcc', a=4.05)
        >>> supercell = make_supercell_structure(atoms, 2)
        >>> len(supercell) == 8  # 8 times the original
        True

        >>> # Diagonal 2×2×3 supercell
        >>> supercell = make_supercell_structure(atoms, [2, 2, 3])
        >>> len(supercell) == 12  # 12 times the original
        True

        >>> # From file with general matrix
        >>> P = [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
        >>> supercell = make_supercell_structure('POSCAR', P)

        >>> # VASP-compatible atom ordering
        >>> supercell = make_supercell_structure(atoms, [2, 2, 2], order='atom-major')
    """
    # Load structure from file if string is provided
    if isinstance(atoms, str):
        loaded = read(atoms)
        # Handle case where read() returns a list of Atoms
        if isinstance(loaded, list):
            atoms = loaded[0]
        else:
            atoms = loaded

    # Make a copy to avoid modifying the original
    atoms = atoms.copy()

    # Convert input to 3×3 numpy array
    P_matrix = _parse_supercell_matrix(P)

    # Validate the matrix
    _validate_supercell_matrix(P_matrix)

    # Generate supercell using ASE
    supercell = make_supercell(atoms, P_matrix, wrap=wrap, order=order)

    return supercell


def _parse_supercell_matrix(
    P: Union[int, List[int], List[List[int]], np.ndarray],
) -> np.ndarray:
    """
    Convert various input formats to 3×3 numpy array.

    Args:
        P: Supercell specification (scalar, diagonal, or matrix)

    Returns:
        3×3 numpy array representing the transformation matrix

    Raises:
        ValueError: If input format is invalid
    """
    # Scalar → diagonal matrix
    if isinstance(P, int):
        return np.diag([P, P, P])

    # Convert to numpy array
    P_array = np.array(P)

    # Diagonal [a, b, c] → diagonal matrix
    if P_array.shape == (3,):
        return np.diag(P_array)

    # 3×3 matrix
    if P_array.shape == (3, 3):
        return P_array

    # Invalid shape
    raise ValueError(
        f"Invalid supercell matrix shape {P_array.shape}. "
        f"Expected scalar, (3,) array, or (3,3) matrix."
    )


def _validate_supercell_matrix(P: np.ndarray) -> None:
    """
    Validate supercell transformation matrix.

    Args:
        P: 3×3 transformation matrix

    Raises:
        ValueError: If matrix contains non-integers or has non-positive determinant
    """
    # Check for integer values
    if not np.allclose(P, P.astype(int)):
        raise ValueError(
            f"Supercell matrix must contain only integers.\n"
            f"Got matrix with non-integer values:\n{P}"
        )

    # Convert to integer for determinant calculation
    P_int = P.astype(int)

    # Check determinant
    det = int(np.round(np.linalg.det(P_int)))
    if det <= 0:
        raise ValueError(
            f"Supercell matrix determinant must be positive, got det(P) = {det}.\n"
            f"Matrix:\n{P_int}"
        )


def mlsupercell_cli():
    """
    Command-line interface for supercell generation.

    This CLI tool allows users to generate supercells from structure files
    with various transformation options.

    Example:
        $ mlsupercell structure.cif --diagonal 2,2,3 -o supercell.vasp
        $ mlsupercell POSCAR --size 2 -o SPOSCAR
        $ mlsupercell input.cif --matrix 2,0,0,0,2,0,0,0,3 -o output.cif
        $ mlsupercell POSCAR --diagonal 2,2,2 --order atom-major -o SPOSCAR
    """
    parser = argparse.ArgumentParser(
        description="Generate supercells from atomic structures."
    )
    parser.add_argument("fname", help="Input structure file (POSCAR, CIF, XYZ, etc.)")

    # Supercell specification (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--size",
        "-s",
        type=int,
        help="Isotropic supercell size (e.g., 2 for 2×2×2)",
    )
    group.add_argument(
        "--diagonal",
        "-d",
        type=str,
        help="Diagonal supercell (e.g., '2,2,3' for 2×2×3)",
    )
    group.add_argument(
        "--matrix",
        "-m",
        type=str,
        help="3×3 transformation matrix as 9 comma-separated values (row-major)",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: <input>_supercell.<ext>)",
        default=None,
    )
    parser.add_argument(
        "--format",
        "-f",
        help="Output format (default: same as input, e.g., vasp, cif, xyz)",
        default=None,
    )
    parser.add_argument(
        "--order",
        choices=["cell-major", "atom-major"],
        default="cell-major",
        help="Atom ordering: cell-major (default) or atom-major (VASP-friendly)",
    )
    parser.add_argument(
        "--no-wrap",
        action="store_true",
        help="Do not wrap atomic positions into supercell",
    )

    args = parser.parse_args()

    # Parse supercell specification
    P: Union[int, List[int], np.ndarray]
    desc: str

    if args.size is not None:
        P = args.size
        desc = f"{args.size}×{args.size}×{args.size}"
    elif args.diagonal is not None:
        try:
            diagonal = [int(x.strip()) for x in args.diagonal.split(",")]
            if len(diagonal) != 3:
                raise ValueError("Diagonal must have exactly 3 values")
            P = diagonal
            desc = f"{diagonal[0]}×{diagonal[1]}×{diagonal[2]}"
        except (ValueError, IndexError) as e:
            parser.error(f"Invalid diagonal format: {e}")
            return  # Make type checker happy (parser.error exits)
    elif args.matrix is not None:
        try:
            values = [int(x.strip()) for x in args.matrix.split(",")]
            if len(values) != 9:
                raise ValueError("Matrix must have exactly 9 values")
            P = np.array(values).reshape(3, 3)
            desc = "custom matrix"
        except (ValueError, IndexError) as e:
            parser.error(f"Invalid matrix format: {e}")
            return  # Make type checker happy (parser.error exits)
    else:
        parser.error("One of --size, --diagonal, or --matrix must be specified")
        return  # Make type checker happy (parser.error exits)

    # Determine output filename
    if args.output is None:
        from pathlib import Path

        input_path = Path(args.fname)
        args.output = str(
            input_path.parent / f"{input_path.stem}_supercell{input_path.suffix}"
        )

    # Generate supercell
    print(f"[Supercell] Loading structure from {args.fname}")
    print(f"[Supercell] Generating {desc} supercell")
    print(f"[Supercell] Atom ordering: {args.order}")

    try:
        supercell = make_supercell_structure(
            atoms=args.fname, P=P, wrap=not args.no_wrap, order=args.order
        )
    except Exception as e:
        parser.error(f"Supercell generation failed: {e}")

    # Write output
    write(args.output, supercell, format=args.format)
    print(f"[Supercell] Output written to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("Supercell Generation Summary")
    print("=" * 60)
    print(f"Input atoms:     {len(read(args.fname))}")
    print(f"Output atoms:    {len(supercell)}")
    print(f"Multiplication:  {len(supercell) // len(read(args.fname))}×")
    print(f"Formula:         {supercell.get_chemical_formula()}")
    print(f"Atom ordering:   {args.order}")
    print("=" * 60)


if __name__ == "__main__":
    mlsupercell_cli()
