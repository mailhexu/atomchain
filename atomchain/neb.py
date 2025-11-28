#!/usr/bin/env python
"""
NEB (Nudged Elastic Band) calculations for finding transition state paths and energy barriers.

This module provides functionality to perform Climbing Image NEB calculations using ML potentials
to find minimum energy paths between initial and final atomic configurations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read, write
try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB
from ase.optimize import BFGS, FIRE, LBFGS

from atomchain.init_model import init_calc


def _interpolate_path(
    initial: Atoms,
    final: Atoms,
    nimages: int = 7,
) -> List[Atoms]:
    """
    Interpolate intermediate images between initial and final states using IDPP.

    The Image Dependent Pair Potential (IDPP) method provides better initial guesses
    than linear interpolation by solving a spring potential in interatomic distance space.

    Args:
        initial: Initial atomic structure
        final: Final atomic structure
        nimages: Total number of images including initial and final (default: 7)

    Returns:
        List of interpolated Atoms objects (including initial and final)

    Raises:
        ValueError: If nimages < 3 or structures are incompatible
    """
    if nimages < 3:
        raise ValueError(f"nimages must be at least 3, got {nimages}")

    # Validate structures
    if len(initial) != len(final):
        raise ValueError(
            f"Initial and final structures have different number of atoms: "
            f"{len(initial)} vs {len(final)}"
        )

    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("Initial and final structures have different atom types")

    # Create copies to avoid modifying inputs
    initial_copy = initial.copy()
    final_copy = final.copy()

    # Create intermediate images
    images = [initial_copy]
    for i in range(nimages - 2):
        images.append(initial_copy.copy())
    images.append(final_copy)

    # Use IDPP for better interpolation
    neb_idpp = NEB(images)
    neb_idpp.interpolate(method='idpp')

    return images


def _calculate_barriers(energies: np.ndarray) -> Dict[str, float]:
    """
    Calculate forward and reverse energy barriers from NEB energy profile.

    Args:
        energies: Array of energies along the NEB path

    Returns:
        Dictionary with keys:
            - 'barrier_forward': Energy barrier from initial to transition state (eV)
            - 'barrier_reverse': Energy barrier from final to transition state (eV)
            - 'ts_index': Index of transition state image
            - 'ts_energy': Energy of transition state (eV)
            - 'reaction_energy': Energy difference final - initial (eV)
    """
    e_initial = energies[0]
    e_final = energies[-1]
    e_max = np.max(energies)
    ts_index = int(np.argmax(energies))

    barrier_forward = e_max - e_initial
    barrier_reverse = e_max - e_final
    reaction_energy = e_final - e_initial

    return {
        "barrier_forward": barrier_forward,
        "barrier_reverse": barrier_reverse,
        "ts_index": ts_index,
        "ts_energy": e_max,
        "reaction_energy": reaction_energy,
    }


def _create_energy_profile_plot(
    energies: np.ndarray,
    barriers: Dict[str, float],
    output: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Create energy profile plot for NEB path with transition state marked.

    Args:
        energies: Array of energies along the NEB path (eV)
        barriers: Dictionary from _calculate_barriers() with barrier info
        output: Output file path (optional)
        show: If True, display plot interactively

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Reaction coordinate (normalized)
    x = np.linspace(0, 1, len(energies))

    # Plot energy profile
    ax.plot(x, energies, 'o-', linewidth=2, markersize=8, label='NEB Path')

    # Mark transition state
    ts_idx = int(barriers['ts_index'])
    ax.plot(x[ts_idx], energies[ts_idx], 'r*', markersize=20,
            label=f"TS (Image {ts_idx})", zorder=5)

    # Mark initial and final states
    ax.plot(x[0], energies[0], 'go', markersize=12, label='Initial', zorder=5)
    ax.plot(x[-1], energies[-1], 'bo', markersize=12, label='Final', zorder=5)

    # Add barrier annotations
    barrier_fwd = barriers['barrier_forward']
    barrier_rev = barriers['barrier_reverse']
    reaction_e = barriers['reaction_energy']

    # Forward barrier arrow
    ax.annotate('', xy=(x[ts_idx], energies[ts_idx]),
                xytext=(x[0], energies[0]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(x[ts_idx] / 2, (energies[0] + energies[ts_idx]) / 2,
            f'E$_a^{{fwd}}$ = {barrier_fwd:.3f} eV',
            fontsize=11, color='green', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Reverse barrier arrow
    ax.annotate('', xy=(x[ts_idx], energies[ts_idx]),
                xytext=(x[-1], energies[-1]),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text((x[ts_idx] + x[-1]) / 2, (energies[-1] + energies[ts_idx]) / 2,
            f'E$_a^{{rev}}$ = {barrier_rev:.3f} eV',
            fontsize=11, color='blue', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels and formatting
    ax.set_xlabel('Reaction Coordinate', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('NEB Energy Profile', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Add info text box
    info_text = (
        f"Transition State: Image {ts_idx}\n"
        f"Forward Barrier: {barrier_fwd:.3f} eV\n"
        f"Reverse Barrier: {barrier_rev:.3f} eV\n"
        f"Reaction Energy: {reaction_e:.3f} eV"
    )
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    # Save to file
    if output:
        output_path = Path(output)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[NEB] Saved energy profile plot to {output}")

    # Show interactively
    if show:
        plt.show()

    return fig


def calculate_neb(
    initial: Union[Atoms, str, Path],
    final: Union[Atoms, str, Path],
    calculator: Union[str, object] = 'chgnet',
    nimages: int = 7,
    fmax: float = 0.05,
    optimizer: str = 'FIRE',
    max_steps: int = 500,
    output: Optional[Union[str, Path]] = None,
    plot: Optional[Union[str, Path]] = None,
    show: bool = False,
    quiet: bool = False,
    model_path: Optional[str] = None,
    climb: bool = True,
) -> Dict:
    """
    Perform Climbing Image NEB calculation to find transition state path.

    This function interpolates a path between initial and final structures, then
    optimizes it using the Nudged Elastic Band method with ML potentials to find
    the minimum energy path and transition state.

    Args:
        initial: Initial atomic structure (Atoms object or file path)
        final: Final atomic structure (Atoms object or file path)
        calculator: Calculator type ('chgnet', 'mace', 'm3gnet') or calculator object
                   Default: 'chgnet'
        nimages: Total number of images including initial and final (default: 7)
        fmax: Maximum force convergence criterion (eV/Å, default: 0.05)
        optimizer: Optimizer to use ('FIRE', 'BFGS', 'LBFGS', default: 'FIRE')
        max_steps: Maximum optimization steps (default: 500)
        output: Output trajectory file for converged path (optional)
        plot: Output file for energy profile plot (optional)
        show: If True, display plot interactively (default: False)
        quiet: If True, suppress progress output (default: False)
        model_path: Path to model file for calculators that support it
        climb: Use climbing image method (default: True)

    Returns:
        Dictionary with keys:
            - 'images': List of optimized Atoms objects along the path
            - 'energies': Array of energies along the path (eV)
            - 'barrier_forward': Forward activation energy (eV)
            - 'barrier_reverse': Reverse activation energy (eV)
            - 'ts_index': Index of transition state image
            - 'ts_energy': Transition state energy (eV)
            - 'reaction_energy': Reaction energy (eV)
            - 'converged': Whether optimization converged

    Raises:
        ValueError: If inputs are invalid or structures incompatible
        FileNotFoundError: If input files don't exist

    Example:
        >>> # Basic NEB calculation
        >>> results = calculate_neb('initial.vasp', 'final.vasp',
        ...                        calculator='chgnet',
        ...                        output='neb_path.traj',
        ...                        plot='barrier.png')
        >>> print(f"Barrier: {results['barrier_forward']:.3f} eV")

        >>> # With custom settings
        >>> results = calculate_neb(initial_atoms, final_atoms,
        ...                        calculator='mace',
        ...                        nimages=9,
        ...                        fmax=0.03,
        ...                        optimizer='BFGS',
        ...                        show=True)
    """
    print_fn = (lambda *args, **kwargs: None) if quiet else print

    # Load structures if file paths provided
    initial_atoms: Atoms
    final_atoms: Atoms
    
    if isinstance(initial, (str, Path)):
        if not Path(initial).exists():
            raise FileNotFoundError(f"Initial structure file not found: {initial}")
        print_fn(f"[NEB] Loading initial structure from {initial}...")
        loaded = read(str(initial))
        if isinstance(loaded, list):
            initial_atoms = loaded[0]
        else:
            initial_atoms = loaded
    else:
        initial_atoms = initial

    if isinstance(final, (str, Path)):
        if not Path(final).exists():
            raise FileNotFoundError(f"Final structure file not found: {final}")
        print_fn(f"[NEB] Loading final structure from {final}...")
        loaded = read(str(final))
        if isinstance(loaded, list):
            final_atoms = loaded[0]
        else:
            final_atoms = loaded
    else:
        final_atoms = final

    # Validate nimages
    if nimages < 3:
        raise ValueError(f"nimages must be at least 3, got {nimages}")

    # Initialize calculator
    if isinstance(calculator, str):
        print_fn(f"[NEB] Initializing {calculator} calculator...")
        calc = init_calc(model_type=calculator, model_path=model_path)
    else:
        calc = calculator

    # Interpolate path
    print_fn(f"[NEB] Interpolating {nimages} images using IDPP...")
    images = _interpolate_path(initial_atoms, final_atoms, nimages)

    # Attach calculator to intermediate images (not endpoints)
    for img in images[1:-1]:
        img.calc = calc

    # Create NEB object
    print_fn(f"[NEB] Setting up NEB with climb={climb}...")
    neb = NEB(images, climb=climb, allow_shared_calculator=True)

    # Select optimizer
    optimizer_map = {
        'FIRE': FIRE,
        'BFGS': BFGS,
        'LBFGS': LBFGS,
    }

    if optimizer.upper() not in optimizer_map:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. "
            f"Choose from: {', '.join(optimizer_map.keys())}"
        )

    opt_class = optimizer_map[optimizer.upper()]
    print_fn(f"[NEB] Optimizing with {optimizer} (fmax={fmax}, max_steps={max_steps})...")

    # Run optimization
    opt = opt_class(neb)
    converged = opt.run(fmax=fmax, steps=max_steps)

    if converged:
        print_fn(f"[NEB] Optimization converged after {opt.get_number_of_steps()} steps")
    else:
        print_fn(f"[NEB] Warning: Optimization did not converge in {max_steps} steps")

    # Extract energies from converged path
    energies = np.array([img.get_potential_energy() for img in images])

    # Calculate barriers
    barriers = _calculate_barriers(energies)

    print_fn(f"[NEB] Forward barrier:  {barriers['barrier_forward']:.4f} eV")
    print_fn(f"[NEB] Reverse barrier:  {barriers['barrier_reverse']:.4f} eV")
    print_fn(f"[NEB] Reaction energy:  {barriers['reaction_energy']:.4f} eV")
    print_fn(f"[NEB] Transition state: Image {barriers['ts_index']}")

    # Save trajectory
    if output:
        print_fn(f"[NEB] Saving converged path to {output}...")
        write(str(output), images)

    # Create plot
    if plot or show:
        print_fn("[NEB] Generating energy profile plot...")
        _create_energy_profile_plot(energies, barriers, output=plot, show=show)

    # Return results
    return {
        'images': images,
        'energies': energies,
        'barrier_forward': barriers['barrier_forward'],
        'barrier_reverse': barriers['barrier_reverse'],
        'ts_index': barriers['ts_index'],
        'ts_energy': barriers['ts_energy'],
        'reaction_energy': barriers['reaction_energy'],
        'converged': converged,
    }


def mlneb_cli():
    """
    Command-line interface for NEB calculations.

    This CLI tool performs Climbing Image NEB calculations to find transition states
    and energy barriers between initial and final atomic configurations.

    Examples:
        mlneb initial.vasp final.vasp --calc chgnet -o neb_path.traj
        mlneb initial.cif final.cif --calc mace --nimages 9 --plot barrier.png
        mlneb init.xyz final.xyz --fmax 0.03 --optimizer BFGS --show
    """
    parser = argparse.ArgumentParser(
        description="Perform NEB calculation to find transition state path and energy barriers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlneb initial.vasp final.vasp --calc chgnet -o neb_path.traj
  mlneb initial.cif final.cif --calc mace --nimages 9 --plot barrier.png
  mlneb init.xyz final.xyz --fmax 0.03 --optimizer BFGS --show
        """,
    )

    parser.add_argument("initial", help="Initial structure file")
    parser.add_argument("final", help="Final structure file")

    parser.add_argument(
        "--calc",
        "-c",
        default="chgnet",
        help="Calculator type: chgnet|mace|m3gnet. Default: chgnet",
    )
    parser.add_argument(
        "--nimages",
        "-n",
        type=int,
        default=7,
        help="Total number of images including endpoints. Default: 7",
    )
    parser.add_argument(
        "--fmax",
        "-f",
        type=float,
        default=0.05,
        help="Force convergence criterion (eV/Å). Default: 0.05",
    )
    parser.add_argument(
        "--optimizer",
        "-opt",
        default="FIRE",
        choices=["FIRE", "BFGS", "LBFGS"],
        help="Optimizer to use. Default: FIRE",
    )
    parser.add_argument(
        "--max-steps",
        "-m",
        type=int,
        default=500,
        help="Maximum optimization steps. Default: 500",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output trajectory file for converged path",
    )
    parser.add_argument(
        "--plot",
        "-p",
        help="Output file for energy profile plot (e.g., barrier.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display energy profile plot interactively",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--model-path",
        help="Path to model file (for calculators that support it)",
    )
    parser.add_argument(
        "--no-climb",
        action="store_true",
        help="Disable climbing image (not recommended)",
    )

    args = parser.parse_args()

    # Run NEB calculation
    try:
        results = calculate_neb(
            initial=args.initial,
            final=args.final,
            calculator=args.calc,
            nimages=args.nimages,
            fmax=args.fmax,
            optimizer=args.optimizer,
            max_steps=args.max_steps,
            output=args.output,
            plot=args.plot,
            show=args.show,
            quiet=args.quiet,
            model_path=args.model_path,
            climb=not args.no_climb,
        )

        # Print summary
        if not args.quiet:
            print("\n" + "=" * 60)
            print("NEB Calculation Summary")
            print("=" * 60)
            print(f"Converged: {results['converged']}")
            print(f"Forward barrier:  {results['barrier_forward']:.4f} eV")
            print(f"Reverse barrier:  {results['barrier_reverse']:.4f} eV")
            print(f"Reaction energy:  {results['reaction_energy']:.4f} eV")
            print(f"Transition state: Image {results['ts_index']}")
            print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(mlneb_cli())
