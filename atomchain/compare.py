#!/usr/bin/env python
"""
Trajectory comparison tool for validating ML potentials against reference data.

This module provides functionality to compare calculated properties (energy, forces,
stress) between two trajectory files, generating scatter plots and statistical metrics
to assess agreement between different calculators or methods.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read


def _extract_properties(
    structures: List[Atoms],
) -> Dict[str, np.ndarray]:
    """
    Extract energy, forces, and stress from a list of Atoms objects.

    Args:
        structures: List of ASE Atoms objects with calculated properties

    Returns:
        Dictionary with keys:
            - 'energy': Array of energies (eV)
            - 'forces': List of force arrays per structure
            - 'stress': List of stress tensors per structure (Voigt notation)

    Raises:
        ValueError: If properties are missing from structures
    """
    energies = []
    forces_list = []
    stress_list = []

    for i, atoms in enumerate(structures):
        try:
            energies.append(atoms.get_potential_energy())
            forces_list.append(atoms.get_forces())
            stress_list.append(atoms.get_stress(voigt=True))
        except Exception as e:
            raise ValueError(
                f"Structure {i} missing calculated properties. "
                f"Ensure trajectories have energy, forces, and stress attached. "
                f"Error: {e}"
            )

    return {
        "energy": np.array(energies),
        "forces": forces_list,
        "stress": stress_list,
    }


def _calculate_rms_forces(forces_list: List[np.ndarray]) -> np.ndarray:
    """
    Calculate RMS forces per structure.

    Args:
        forces_list: List of force arrays, each with shape (n_atoms, 3)

    Returns:
        Array of RMS forces (eV/Å) per structure
    """
    rms_forces = []
    for forces in forces_list:
        # RMS = sqrt(mean(F_x^2 + F_y^2 + F_z^2))
        rms = np.sqrt(np.mean(np.sum(forces**2, axis=1)))
        rms_forces.append(rms)
    return np.array(rms_forces)


def _calculate_hydrostatic_stress(stress_list: List[np.ndarray]) -> np.ndarray:
    """
    Calculate hydrostatic stress (pressure) from stress tensors.

    Args:
        stress_list: List of stress tensors in Voigt notation [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]

    Returns:
        Array of hydrostatic stress values: P = -(σ_xx + σ_yy + σ_zz) / 3
        Units: Same as input (typically eV/Å³, convert to GPa for display)
    """
    hydro = []
    for stress in stress_list:
        # Hydrostatic stress (pressure) = -mean of diagonal components
        p = -(stress[0] + stress[1] + stress[2]) / 3.0
        hydro.append(p)
    return np.array(hydro)


def _calculate_shear_stress(stress_list: List[np.ndarray]) -> np.ndarray:
    """
    Calculate max shear stress from stress tensors.

    Args:
        stress_list: List of stress tensors in Voigt notation

    Returns:
        Array of max shear stress values: τ_max = (σ_max - σ_min) / 2
    """
    shear = []
    for stress in stress_list:
        # Get principal stresses (eigenvalues of stress tensor)
        # Stress tensor from Voigt notation
        stress_tensor = np.array(
            [
                [stress[0], stress[5], stress[4]],
                [stress[5], stress[1], stress[3]],
                [stress[4], stress[3], stress[2]],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(stress_tensor)
        # Max shear stress
        tau_max = (eigenvalues.max() - eigenvalues.min()) / 2.0
        shear.append(tau_max)
    return np.array(shear)


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate correlation and error metrics.

    Args:
        y_true: Reference values
        y_pred: Predicted values

    Returns:
        Dictionary with keys: 'r2', 'rmse', 'mae'
    """
    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {"r2": r2, "rmse": rmse, "mae": mae}


def _normalize_energy(energies: np.ndarray) -> np.ndarray:
    """
    Normalize energies relative to minimum energy.

    Args:
        energies: Array of energies

    Returns:
        Normalized energies: E_rel = E - E_min
    """
    return energies - np.min(energies)


def _print_comparison_statistics(
    labels: Tuple[str, str],
    n_structures: int,
    energy_metrics: Dict[str, float],
    forces_metrics: Dict[str, float],
    hydro_metrics: Dict[str, float],
    shear_metrics: Dict[str, float],
    energy_range_1: Tuple[float, float],
    energy_range_2: Tuple[float, float],
    forces_range_1: Tuple[float, float],
    forces_range_2: Tuple[float, float],
    quiet: bool = False,
) -> None:
    """
    Print formatted comparison statistics to console.

    Args:
        labels: Tuple of (trajectory1_name, trajectory2_name)
        n_structures: Number of structures compared
        energy_metrics: Metrics dict for energy
        forces_metrics: Metrics dict for forces
        hydro_metrics: Metrics dict for hydrostatic stress
        shear_metrics: Metrics dict for shear stress
        energy_range_1: (min, max) energy for trajectory 1
        energy_range_2: (min, max) energy for trajectory 2
        forces_range_1: (min, max) RMS forces for trajectory 1
        forces_range_2: (min, max) RMS forces for trajectory 2
        quiet: If True, suppress output
    """
    if quiet:
        return

    print("\n" + "=" * 60)
    print("Trajectory Comparison Statistics")
    print("=" * 60)
    print(f"Trajectory 1: {labels[0]}")
    print(f"Trajectory 2: {labels[1]}")
    print(f"Structures compared: {n_structures}")
    print()
    print(f"{'Property':<20} {'R²':>8} {'RMSE':>12} {'MAE':>12}")
    print("-" * 60)
    print(
        f"{'Energy (eV)':<20} {energy_metrics['r2']:>8.4f} "
        f"{energy_metrics['rmse']:>12.6f} {energy_metrics['mae']:>12.6f}"
    )
    print(
        f"{'RMS Forces (eV/Å)':<20} {forces_metrics['r2']:>8.4f} "
        f"{forces_metrics['rmse']:>12.6f} {forces_metrics['mae']:>12.6f}"
    )
    print(
        f"{'Hydro Stress (GPa)':<20} {hydro_metrics['r2']:>8.4f} "
        f"{hydro_metrics['rmse']:>12.6f} {hydro_metrics['mae']:>12.6f}"
    )
    print(
        f"{'Shear Stress (GPa)':<20} {shear_metrics['r2']:>8.4f} "
        f"{shear_metrics['rmse']:>12.6f} {shear_metrics['mae']:>12.6f}"
    )
    print()
    print("Value Ranges:")
    print(
        f"  Energy: [{energy_range_1[0]:.2f}, {energy_range_1[1]:.2f}] ({labels[0]}), "
        f"[{energy_range_2[0]:.2f}, {energy_range_2[1]:.2f}] ({labels[1]})"
    )
    print(
        f"  Forces: [{forces_range_1[0]:.4f}, {forces_range_1[1]:.4f}] ({labels[0]}), "
        f"[{forces_range_2[0]:.4f}, {forces_range_2[1]:.4f}] ({labels[1]})"
    )
    print("=" * 60 + "\n")


def _create_scatter_plot(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    metrics: Dict[str, float],
    unit: str = "",
) -> None:
    """
    Create a scatter plot with perfect agreement line and metrics.

    Args:
        ax: Matplotlib axes object
        x: X values (trajectory 1)
        y: Y values (trajectory 2)
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        metrics: Dictionary with 'r2', 'rmse', 'mae'
        unit: Unit string for display in metrics box
    """
    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=30, edgecolors="k", linewidths=0.5)

    # Perfect agreement line (y=x)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Perfect agreement")

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Metrics text box
    metrics_text = (
        f"R² = {metrics['r2']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}{unit}\n"
        f"MAE = {metrics['mae']:.4f}{unit}"
    )
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )


def _create_comparison_figure(
    energy_1: np.ndarray,
    energy_2: np.ndarray,
    forces_1: np.ndarray,
    forces_2: np.ndarray,
    hydro_1: np.ndarray,
    hydro_2: np.ndarray,
    shear_1: np.ndarray,
    shear_2: np.ndarray,
    labels: Tuple[str, str],
    output: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create 2x2 comparison figure with all property plots.

    Args:
        energy_1: Energy values from trajectory 1
        energy_2: Energy values from trajectory 2
        forces_1: RMS forces from trajectory 1
        forces_2: RMS forces from trajectory 2
        hydro_1: Hydrostatic stress from trajectory 1
        hydro_2: Hydrostatic stress from trajectory 2
        shear_1: Shear stress from trajectory 1
        shear_2: Shear stress from trajectory 2
        labels: Tuple of (trajectory1_name, trajectory2_name)
        output: Output file path (optional)
        show: If True, display plot interactively

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Trajectory Comparison", fontsize=14, fontweight="bold")

    # Calculate metrics for each property
    energy_metrics = _calculate_metrics(energy_1, energy_2)
    forces_metrics = _calculate_metrics(forces_1, forces_2)

    # Convert stress to GPa (1 eV/Å³ = 160.2176 GPa)
    eV_A3_to_GPa = 160.2176
    hydro_1_gpa = hydro_1 * eV_A3_to_GPa
    hydro_2_gpa = hydro_2 * eV_A3_to_GPa
    shear_1_gpa = shear_1 * eV_A3_to_GPa
    shear_2_gpa = shear_2 * eV_A3_to_GPa

    # Recalculate metrics in GPa
    hydro_metrics_gpa = _calculate_metrics(hydro_1_gpa, hydro_2_gpa)
    shear_metrics_gpa = _calculate_metrics(shear_1_gpa, shear_2_gpa)

    # Energy plot
    _create_scatter_plot(
        axes[0, 0],
        energy_1,
        energy_2,
        f"Energy ({labels[0]}) [eV]",
        f"Energy ({labels[1]}) [eV]",
        "Energy Comparison",
        energy_metrics,
        " eV",
    )

    # Forces plot
    _create_scatter_plot(
        axes[0, 1],
        forces_1,
        forces_2,
        f"RMS Forces ({labels[0]}) [eV/Å]",
        f"RMS Forces ({labels[1]}) [eV/Å]",
        "RMS Forces Comparison",
        forces_metrics,
        " eV/Å",
    )

    # Hydrostatic stress plot
    _create_scatter_plot(
        axes[1, 0],
        hydro_1_gpa,
        hydro_2_gpa,
        f"Hydrostatic Stress ({labels[0]}) [GPa]",
        f"Hydrostatic Stress ({labels[1]}) [GPa]",
        "Hydrostatic Stress Comparison",
        hydro_metrics_gpa,
        " GPa",
    )

    # Shear stress plot
    _create_scatter_plot(
        axes[1, 1],
        shear_1_gpa,
        shear_2_gpa,
        f"Shear Stress ({labels[0]}) [GPa]",
        f"Shear Stress ({labels[1]}) [GPa]",
        "Shear Stress Comparison",
        shear_metrics_gpa,
        " GPa",
    )

    plt.tight_layout()

    # Save to file
    if output:
        output_path = Path(output)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[Compare] Saved comparison plot to {output}")

    # Show interactively
    if show:
        plt.show()

    return fig


def compare_trajectories(
    trajectory1: Union[str, Path],
    trajectory2: Union[str, Path],
    labels: Optional[Tuple[str, str]] = None,
    normalize_energy: bool = False,
    output: Optional[Union[str, Path]] = "trajectory_comparison.png",
    show: bool = False,
    quiet: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compare calculated properties between two trajectory files.

    This function loads two trajectory files with calculated properties (energy, forces,
    stress), computes comparison metrics, generates scatter plots, and prints statistics.

    Args:
        trajectory1: Path to first trajectory file
        trajectory2: Path to second trajectory file
        labels: Optional tuple of (name1, name2) for axis labels. Default: ('Trajectory 1', 'Trajectory 2')
        normalize_energy: If True, normalize energies relative to minimum. Default: False
        output: Output file path for plot. Default: 'trajectory_comparison.png'
                Set to None to skip saving.
        show: If True, display plot interactively. Default: False
        quiet: If True, suppress console output. Default: False

    Returns:
        Dictionary with comparison results containing:
            - 'energy_1', 'energy_2': Energy arrays
            - 'forces_1', 'forces_2': RMS forces arrays
            - 'hydro_1', 'hydro_2': Hydrostatic stress arrays
            - 'shear_1', 'shear_2': Shear stress arrays
            - 'metrics': Dict of metrics for each property

    Raises:
        FileNotFoundError: If trajectory files don't exist
        ValueError: If trajectories are empty or missing properties

    Example:
        >>> # Compare two trajectory files
        >>> results = compare_trajectories('dft.traj', 'ml.traj',
        ...                                labels=('DFT', 'CHGNet'),
        ...                                output='comparison.png',
        ...                                show=True)

        >>> # With energy normalization
        >>> results = compare_trajectories('ref.traj', 'pred.traj',
        ...                                normalize_energy=True,
        ...                                quiet=True)
    """
    print_fn = (lambda *args, **kwargs: None) if quiet else print

    # Load trajectories
    print_fn(f"[Compare] Loading trajectory 1 from {trajectory1}...")
    if not Path(trajectory1).exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory1}")

    structures1 = read(str(trajectory1), ":")
    if isinstance(structures1, Atoms):
        structures1 = [structures1]

    print_fn(f"[Compare] Loading trajectory 2 from {trajectory2}...")
    if not Path(trajectory2).exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory2}")

    structures2 = read(str(trajectory2), ":")
    if isinstance(structures2, Atoms):
        structures2 = [structures2]

    # Validate trajectories
    if len(structures1) == 0 or len(structures2) == 0:
        raise ValueError("One or both trajectories are empty")

    # Handle mismatched lengths
    n1, n2 = len(structures1), len(structures2)
    if n1 != n2:
        n_compare = min(n1, n2)
        print_fn(
            f"[Compare] Warning: Trajectory lengths differ ({n1} vs {n2}). "
            f"Comparing first {n_compare} structures."
        )
        structures1 = structures1[:n_compare]
        structures2 = structures2[:n_compare]
    else:
        n_compare = n1

    print_fn(f"[Compare] Comparing {n_compare} structures...")

    # Extract properties
    props1 = _extract_properties(structures1)
    props2 = _extract_properties(structures2)

    # Get energies
    energy_1 = props1["energy"]
    energy_2 = props2["energy"]

    # Normalize if requested
    if normalize_energy:
        print_fn("[Compare] Normalizing energies relative to minimum...")
        energy_1 = _normalize_energy(energy_1)
        energy_2 = _normalize_energy(energy_2)

    # Calculate RMS forces
    forces_1 = _calculate_rms_forces(props1["forces"])
    forces_2 = _calculate_rms_forces(props2["forces"])

    # Calculate stresses
    hydro_1 = _calculate_hydrostatic_stress(props1["stress"])
    hydro_2 = _calculate_hydrostatic_stress(props2["stress"])
    shear_1 = _calculate_shear_stress(props1["stress"])
    shear_2 = _calculate_shear_stress(props2["stress"])

    # Calculate metrics
    energy_metrics = _calculate_metrics(energy_1, energy_2)
    forces_metrics = _calculate_metrics(forces_1, forces_2)

    # Stress metrics in GPa
    eV_A3_to_GPa = 160.2176
    hydro_metrics = _calculate_metrics(hydro_1 * eV_A3_to_GPa, hydro_2 * eV_A3_to_GPa)
    shear_metrics = _calculate_metrics(shear_1 * eV_A3_to_GPa, shear_2 * eV_A3_to_GPa)

    # Set labels
    if labels is None:
        labels = ("Trajectory 1", "Trajectory 2")

    # Print statistics
    _print_comparison_statistics(
        labels,
        n_compare,
        energy_metrics,
        forces_metrics,
        hydro_metrics,
        shear_metrics,
        (energy_1.min(), energy_1.max()),
        (energy_2.min(), energy_2.max()),
        (forces_1.min(), forces_1.max()),
        (forces_2.min(), forces_2.max()),
        quiet,
    )

    # Create plots
    if output or show:
        print_fn("[Compare] Generating comparison plots...")
        _create_comparison_figure(
            energy_1,
            energy_2,
            forces_1,
            forces_2,
            hydro_1,
            hydro_2,
            shear_1,
            shear_2,
            labels,
            output,
            show,
        )

    # Return results
    return {
        "energy_1": energy_1,
        "energy_2": energy_2,
        "forces_1": forces_1,
        "forces_2": forces_2,
        "hydro_1": hydro_1,
        "hydro_2": hydro_2,
        "shear_1": shear_1,
        "shear_2": shear_2,
        "metrics": {
            "energy": energy_metrics,
            "forces": forces_metrics,
            "hydro": hydro_metrics,
            "shear": shear_metrics,
        },
    }


def mlcompare_cli():
    """
    Command-line interface for trajectory comparison.

    This CLI tool allows users to compare two trajectory files containing structures
    with calculated properties, generating comparison plots and statistics.

    Examples:
        mlcompare traj1.traj traj2.traj -o comparison.png
        mlcompare dft.traj ml.traj --labels "DFT" "CHGNet" --show
        mlcompare ref.traj pred.traj --normalize-energy --quiet
    """
    parser = argparse.ArgumentParser(
        description="Compare calculated properties between two trajectory files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlcompare traj1.traj traj2.traj -o comparison.png
  mlcompare dft.traj ml.traj --labels "DFT" "CHGNet" --show
  mlcompare ref.traj pred.traj --normalize-energy --format pdf
        """,
    )

    parser.add_argument("trajectory1", help="First trajectory file")
    parser.add_argument("trajectory2", help="Second trajectory file")
    parser.add_argument(
        "--output",
        "-o",
        default="trajectory_comparison.png",
        help="Output plot file. Default: trajectory_comparison.png",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )
    parser.add_argument(
        "--normalize-energy",
        action="store_true",
        help="Normalize energies relative to minimum",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        metavar=("LABEL1", "LABEL2"),
        help="Custom labels for trajectories (e.g., --labels 'DFT' 'CHGNet')",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress console output"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        help="Output format (overrides extension in --output)",
    )

    args = parser.parse_args()

    # Handle format option
    output = args.output
    if args.format:
        output_path = Path(output)
        output = str(output_path.with_suffix(f".{args.format}"))

    # Convert labels to tuple if provided
    labels = tuple(args.labels) if args.labels else None

    # Run comparison
    try:
        compare_trajectories(
            args.trajectory1,
            args.trajectory2,
            labels=labels,
            normalize_energy=args.normalize_energy,
            output=output,
            show=args.show,
            quiet=args.quiet,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(mlcompare_cli())
