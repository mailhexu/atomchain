#!/usr/bin/env python
"""
Batch trajectory processing for computing energy, forces, and stress using ML potentials.

This module provides functionality to process multiple structures in a trajectory file,
computing properties for each structure and saving results to a new trajectory file
with calculator results attached to Atoms objects.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Union

from ase import Atoms
from ase.io import read, write

from atomchain.init_model import init_calc


def calculate_trajectory_batch(
    trajectory: Union[str, Path, List[Atoms]],
    calculator: Union[str, object] = "chgnet",
    model_path: Optional[str] = None,
    output: Union[str, Path] = "batch_output.traj",
    verbose: bool = False,
    quiet: bool = False,
) -> List[Atoms]:
    """
    Process all structures in a trajectory file with ML potential calculator.

    This function reads structures from a trajectory file, computes energy, forces,
    and stress for each structure using the specified ML potential, and saves the
    results to a new trajectory file with calculator results attached to Atoms objects.

    Args:
        trajectory: Path to input trajectory file (.traj, .xyz, etc.) or list of Atoms
        calculator: Calculator to use. Can be:
                   - String: 'chgnet', 'm3gnet', 'mace', 'deepmd', etc.
                   - ASE Calculator object: pre-initialized calculator
                   Default: 'chgnet'
        model_path: Path to custom model file (required for some calculators like deepmd)
        output: Path to output trajectory file. Default: 'batch_output.traj'
        verbose: If True, print progress information. Default: False
        quiet: If True, suppress all output. Default: False

    Returns:
        List of Atoms objects with calculated properties attached

    Raises:
        FileNotFoundError: If trajectory file does not exist
        ValueError: If trajectory is empty or calculator initialization fails

    Example:
        >>> # Process trajectory with CHGNet
        >>> atoms_list = calculate_trajectory_batch('input.traj', 'chgnet',
        ...                                          output='output.traj')

        >>> # With verbose output
        >>> atoms_list = calculate_trajectory_batch('structures.traj',
        ...                                          calculator='mace',
        ...                                          output='calculated.traj',
        ...                                          verbose=True)

        >>> # With pre-initialized calculator
        >>> from atomchain.init_model import init_calc
        >>> my_calc = init_calc(model_type='chgnet')
        >>> atoms_list = calculate_trajectory_batch('input.traj',
        ...                                          calculator=my_calc)
    """
    # Suppress output if quiet mode
    print_fn = (lambda *args, **kwargs: None) if quiet else print

    # Load trajectory
    if isinstance(trajectory, (str, Path)):
        trajectory_path = Path(trajectory)
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory}")

        print_fn(f"[Batch] Loading trajectory from {trajectory}...")
        try:
            structures = read(str(trajectory), ":")
        except Exception as e:
            # Handle empty files or unreadable files
            if "Empty file" in str(e) or "empty" in str(e).lower():
                structures = []
            else:
                raise

        # Handle case where read returns a single Atoms object
        if isinstance(structures, Atoms):
            structures = [structures]
    else:
        # trajectory is already a list of Atoms
        structures = list(trajectory)

    # Validate trajectory
    n_structures = len(structures)
    if n_structures == 0:
        if not quiet:
            print("[Batch] Warning: Empty trajectory. Nothing to process.")
        return []

    print_fn(f"[Batch] Found {n_structures} structures in trajectory")

    # Initialize calculator once and reuse for all structures
    if isinstance(calculator, str):
        print_fn(f"[Batch] Initializing calculator: {calculator}")
        calc = init_calc(model_type=calculator, model_path=model_path)
        calculator_name = calculator
    else:
        # Use provided calculator object
        calc = calculator
        calculator_name = type(calculator).__name__
        print_fn(f"[Batch] Using provided calculator: {calculator_name}")

    # Process structures
    processed_structures = []
    failed_indices = []
    start_time = time.time()

    for i, atoms in enumerate(structures):
        structure_start = time.time()

        try:
            # Make a copy to avoid modifying original
            atoms_copy = atoms.copy()

            # Attach calculator
            atoms_copy.calc = calc

            # Compute properties (this triggers the calculation)
            energy = atoms_copy.get_potential_energy()
            _ = atoms_copy.get_forces()
            _ = atoms_copy.get_stress(voigt=True)

            processed_structures.append(atoms_copy)

            # Print progress
            if not quiet:
                if verbose:
                    elapsed = time.time() - structure_start
                    total_elapsed = time.time() - start_time
                    avg_time = total_elapsed / (i + 1)
                    eta = avg_time * (n_structures - i - 1)
                    print(
                        f"[Batch] Processed structure {i + 1}/{n_structures} "
                        f"(E={energy:.6f} eV, {elapsed:.2f}s, ETA: {eta:.1f}s)"
                    )
                else:
                    # In default mode, show simpler progress with energy
                    print(
                        f"[Batch] Processed structure {i + 1}/{n_structures} "
                        f"(E={energy:.6f} eV)"
                    )

        except Exception as e:
            # Log error and continue with next structure
            failed_indices.append(i)
            if not quiet:
                print(f"[Batch] Error processing structure {i + 1}: {str(e)}")
            continue

    # Save to output trajectory
    if processed_structures:
        output_path = Path(output)
        print_fn(
            f"[Batch] Saving {len(processed_structures)} structures to {output}..."
        )

        # Write trajectory using ASE write function
        # For .traj files, this will create a trajectory with all structures
        write(str(output_path), processed_structures)

        print_fn(f"[Batch] Successfully saved to {output}")
    else:
        if not quiet:
            print("[Batch] No structures were successfully processed.")

    # Print summary
    if verbose and not quiet:
        total_time = time.time() - start_time
        avg_time = total_time / n_structures if n_structures > 0 else 0
        print("\n" + "=" * 60)
        print("Batch Processing Summary")
        print("=" * 60)
        print(f"Calculator:       {calculator_name}")
        print(f"Total structures: {n_structures}")
        print(f"Processed:        {len(processed_structures)}")
        print(f"Failed:           {len(failed_indices)}")
        print(f"Total time:       {total_time:.2f}s")
        print(f"Avg time/struct:  {avg_time:.2f}s")
        if failed_indices:
            print(f"Failed indices:   {failed_indices}")
        print("=" * 60)

    return processed_structures


def mlbatch_cli():
    """
    Command-line interface for batch trajectory processing.

    This CLI tool allows users to process trajectory files containing multiple
    structures, computing energy, forces, and stress for each structure using
    ML potentials and saving results to a new trajectory file.

    Example:
        $ mlbatch input.traj -o output.traj
        $ mlbatch structures.traj --calculator mace -o results.traj --verbose
        $ mlbatch trajectory.traj -m deepmd -p model.pb -o calculated.traj
    """
    parser = argparse.ArgumentParser(
        description="Batch process trajectory files with ML potential calculations.",
        epilog="""
Examples:
  mlbatch input.traj -o output.traj
  mlbatch structures.traj --calculator mace --verbose
  mlbatch trajectory.traj -m deepmd -p model.pb -o results.traj
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("trajectory", help="Input trajectory file (.traj, .xyz, etc.)")

    parser.add_argument(
        "--calculator",
        "--model",
        "-m",
        dest="calculator",
        help="ML potential calculator: chgnet|m3gnet|mace|deepmd. Default: chgnet",
        default="chgnet",
    )

    parser.add_argument(
        "--model-path",
        "-p",
        dest="model_path",
        help="Path to custom model file (for deepmd, etc.)",
        default=None,
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output trajectory file. Default: batch_output.traj",
        default="batch_output.traj",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    args = parser.parse_args()

    # Process trajectory
    try:
        calculate_trajectory_batch(
            trajectory=args.trajectory,
            calculator=args.calculator,
            model_path=args.model_path,
            output=args.output,
            verbose=args.verbose,
            quiet=args.quiet,
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(mlbatch_cli())
