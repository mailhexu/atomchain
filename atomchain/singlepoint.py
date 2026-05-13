#!/usr/bin/env python
"""
Single point energy, forces, and stress calculations using ML potentials.

This module provides functionality to perform single point calculations on
atomic structures without optimization, with structured output to YAML files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import numpy as np
import yaml
from ase import Atoms
from ase.io import read

from atomchain.init_model import init_calc


@dataclass
class SinglePointResult:
    """
    Container for single point calculation results.

    Attributes:
        atoms: ASE Atoms object containing the structure
        energy: Potential energy in eV
        forces: Forces on each atom in eV/Å, shape (natoms, 3)
        stress: Stress tensor in Voigt notation (6,) in eV/Å³
                Order: [xx, yy, zz, yz, xz, xy]
        calculator_name: Name of the calculator used
        timestamp: ISO 8601 format timestamp of calculation
    """

    atoms: Atoms
    energy: float
    forces: np.ndarray
    stress: np.ndarray
    calculator_name: str
    timestamp: str

    def to_yaml(self, filename: str) -> None:
        """
        Save calculation results to a YAML file.

        Args:
            filename: Path to output YAML file

        Example:
            >>> result = calculate_single_point(atoms, calc='chgnet')
            >>> result.to_yaml('results.yaml')
        """
        # Convert Atoms object to dictionary
        structure_dict = {
            "cell": np.array(self.atoms.get_cell()).tolist(),
            "positions": self.atoms.get_positions().tolist(),
            "symbols": self.atoms.get_chemical_symbols(),
            "pbc": self.atoms.get_pbc().tolist(),
        }

        # Create output dictionary
        data = {
            "calculator": self.calculator_name,
            "timestamp": self.timestamp,
            "energy": float(self.energy),
            "forces": self.forces.tolist(),
            "stress": self.stress.tolist(),
            "structure": structure_dict,
        }

        # Write to YAML file with comments
        with open(filename, "w") as f:
            f.write("# Single Point Calculation Results\n")
            f.write("# Units: energy (eV), forces (eV/Å), stress (eV/Å³)\n")
            f.write("# Stress in Voigt notation: [xx, yy, zz, yz, xz, xy]\n")
            f.write("\n")
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, filename: str) -> SinglePointResult:
        """
        Load calculation results from a YAML file.

        Args:
            filename: Path to input YAML file

        Returns:
            SinglePointResult object with loaded data

        Example:
            >>> result = SinglePointResult.from_yaml('results.yaml')
            >>> print(f"Energy: {result.energy} eV")
        """
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        # Reconstruct Atoms object
        atoms = Atoms(
            symbols=data["structure"]["symbols"],
            positions=data["structure"]["positions"],
            cell=data["structure"]["cell"],
            pbc=data["structure"]["pbc"],
        )

        # Convert lists back to numpy arrays
        forces = np.array(data["forces"])
        stress = np.array(data["stress"])

        return cls(
            atoms=atoms,
            energy=data["energy"],
            forces=forces,
            stress=stress,
            calculator_name=data["calculator"],
            timestamp=data["timestamp"],
        )


def calculate_single_point(
    atoms: Union[Atoms, str],
    calc: Union[str, object] = "chgnet",
    model_path: Optional[str] = None,
) -> SinglePointResult:
    """
    Perform single point energy, forces, and stress calculation.

    This function calculates energy, forces, and stress for a given atomic
    structure without performing any optimization. It supports multiple ML
    potential calculators and can accept either an ASE Atoms object or a
    file path to a structure file.

    Args:
        atoms: ASE Atoms object or path to structure file (POSCAR, CIF, XYZ, etc.)
        calc: Calculator to use. Can be:
              - String: 'chgnet', 'm3gnet', 'mace', 'deepmd', etc.
              - ASE Calculator object: pre-initialized calculator
              Default: 'chgnet'
        model_path: Path to custom model file (required for some calculators like deepmd)

    Returns:
        SinglePointResult: Object containing energy, forces, stress, and structure

    Raises:
        FileNotFoundError: If structure file path does not exist
        ValueError: If calculator initialization fails

    Example:
        >>> # From Atoms object
        >>> from ase.build import bulk
        >>> atoms = bulk('Al', 'fcc', a=4.05)
        >>> result = calculate_single_point(atoms, calc='chgnet')
        >>> print(f"Energy: {result.energy} eV")

        >>> # From file
        >>> result = calculate_single_point('POSCAR', calc='mace')
        >>> result.to_yaml('results.yaml')

        >>> # With pre-initialized calculator
        >>> from atomchain.init_model import init_calc
        >>> my_calc = init_calc(model_type='chgnet')
        >>> result = calculate_single_point(atoms, calc=my_calc)
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

    # Initialize calculator if string name is provided
    if isinstance(calc, str):
        calculator = init_calc(model_type=calc, model_path=model_path)
        calculator_name = calc
    else:
        # Use provided calculator object
        calculator = calc
        calculator_name = type(calc).__name__

    # Attach calculator to atoms
    atoms.calc = calculator

    # Perform calculations
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)

    # Get current timestamp
    timestamp = datetime.now().isoformat()

    # Create and return result object
    return SinglePointResult(
        atoms=atoms,
        energy=energy,
        forces=forces,
        stress=stress,
        calculator_name=calculator_name,
        timestamp=timestamp,
    )


def mlsinglepoint_cli():
    """
    Command-line interface for single point calculations.

    This CLI tool allows users to perform single point energy, forces, and
    stress calculations from the command line with various calculator options.

    Example:
        $ mlsinglepoint structure.cif -o results.yaml
        $ mlsinglepoint POSCAR --model mace -o results.yaml
        $ mlsinglepoint structure.cif -m deepmd -p model.pb -o results.yaml
    """
    parser = argparse.ArgumentParser(
        description="Perform single point energy, forces, and stress calculation with ML potentials."
    )
    parser.add_argument("fname", help="Input structure file (POSCAR, CIF, XYZ, etc.)")
    parser.add_argument(
        "--model",
        "-m",
        help="ML potential model: chgnet|m3gnet|mace|deepmd. Default is chgnet",
        default="chgnet",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help="Output YAML file for results. Default is singlepoint_result.yaml",
        default="singlepoint_result.yaml",
    )
    parser.add_argument(
        "--model_path",
        "-p",
        help="Path to custom model file (for deepmd, etc.)",
        default=None,
    )

    args = parser.parse_args()

    # Perform calculation
    print(f"[SinglePoint] Loading structure from {args.fname}")
    print(f"[SinglePoint] Using calculator: {args.model}")

    result = calculate_single_point(
        atoms=args.fname, calc=args.model, model_path=args.model_path
    )

    # Save to YAML
    result.to_yaml(args.output_file)
    print(f"[SinglePoint] Results saved to {args.output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Single Point Calculation Summary")
    print("=" * 60)
    print(f"Calculator: {result.calculator_name}")
    print(f"Timestamp:  {result.timestamp}")
    print(f"Energy:     {result.energy:.6f} eV")
    print(f"Max force:  {np.max(np.abs(result.forces)):.6f} eV/Å")
    print(f"Max stress: {np.max(np.abs(result.stress)):.6f} eV/Å³")
    print("=" * 60)


if __name__ == "__main__":
    mlsinglepoint_cli()
