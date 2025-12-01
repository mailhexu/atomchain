"""
Tests for trajectory comparison functionality.

Tests the compare_trajectories() function and mlcompare CLI for comparing
calculated properties between two trajectory files.

Purpose:
    Verify that trajectory comparison works correctly, calculates metrics
    accurately, generates comparison plots, and provides formatted statistics.

How to run:
    pytest tests/test_compare.py -v
    or
    pytest tests/test_compare.py::test_compare_basic -v  # run specific test
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write

from atomchain.compare import (
    _calculate_hydrostatic_stress,
    _calculate_metrics,
    _calculate_rms_forces,
    _calculate_shear_stress,
    _extract_properties,
    _normalize_energy,
    compare_trajectories,
)


@pytest.fixture
def al_structure():
    """Create Al fcc structure."""
    return bulk("Al", "fcc", a=4.05)


@pytest.fixture
def sample_trajectories(al_structure, temp_dir):
    """Create two sample trajectory files with calculated properties."""
    from ase.calculators.singlepoint import SinglePointCalculator

    structures_1 = []
    structures_2 = []

    calc = EMT()

    for i in range(5):
        # First trajectory
        atoms1 = al_structure.copy()
        atoms1.rattle(stdev=0.05 * (i + 1))
        atoms1.calc = calc
        energy1 = atoms1.get_potential_energy()
        forces1 = atoms1.get_forces()
        stress1 = atoms1.get_stress()
        # Attach as single point calculator to preserve through write/read
        atoms1.calc = SinglePointCalculator(atoms1, energy=energy1, forces=forces1, stress=stress1)
        structures_1.append(atoms1)

        # Second trajectory with slightly different perturbations
        atoms2 = al_structure.copy()
        atoms2.rattle(stdev=0.06 * (i + 1))
        atoms2.calc = calc
        energy2 = atoms2.get_potential_energy()
        forces2 = atoms2.get_forces()
        stress2 = atoms2.get_stress()
        # Attach as single point calculator to preserve through write/read
        atoms2.calc = SinglePointCalculator(atoms2, energy=energy2, forces=forces2, stress=stress2)
        structures_2.append(atoms2)

    traj1_path = temp_dir / "traj1.traj"
    traj2_path = temp_dir / "traj2.traj"
    write(str(traj1_path), structures_1)
    write(str(traj2_path), structures_2)

    return traj1_path, traj2_path


def test_extract_properties(sample_trajectories):
    """
    Test property extraction from trajectory.

    Verifies that energy, forces, and stress are correctly extracted
    from trajectory files.
    """
    traj1_path, _ = sample_trajectories
    structures = read(str(traj1_path), ":")

    props = _extract_properties(structures)

    assert "energy" in props
    assert "forces" in props
    assert "stress" in props

    assert len(props["energy"]) == 5
    assert len(props["forces"]) == 5
    assert len(props["stress"]) == 5

    # Check shapes
    assert props["forces"][0].shape == (1, 3)  # Al bulk primitive has 1 atom
    assert props["stress"][0].shape == (6,)  # Voigt notation


def test_calculate_rms_forces(sample_trajectories):
    """
    Test RMS forces calculation.

    Verifies that RMS forces per structure are calculated correctly.
    """
    traj1_path, _ = sample_trajectories
    structures = read(str(traj1_path), ":")
    props = _extract_properties(structures)

    rms_forces = _calculate_rms_forces(props["forces"])

    assert len(rms_forces) == 5
    assert all(isinstance(f, (float, np.floating)) for f in rms_forces)
    assert all(f >= 0 for f in rms_forces)


def test_calculate_hydrostatic_stress(sample_trajectories):
    """
    Test hydrostatic stress calculation.

    Verifies that hydrostatic stress (pressure) is calculated correctly.
    """
    traj1_path, _ = sample_trajectories
    structures = read(str(traj1_path), ":")
    props = _extract_properties(structures)

    hydro_stress = _calculate_hydrostatic_stress(props["stress"])

    assert len(hydro_stress) == 5
    assert all(isinstance(s, (float, np.floating)) for s in hydro_stress)


def test_calculate_shear_stress(sample_trajectories):
    """
    Test shear stress calculation.

    Verifies that maximum shear stress is calculated correctly.
    """
    traj1_path, _ = sample_trajectories
    structures = read(str(traj1_path), ":")
    props = _extract_properties(structures)

    shear_stress = _calculate_shear_stress(props["stress"])

    assert len(shear_stress) == 5
    assert all(isinstance(s, (float, np.floating)) for s in shear_stress)
    assert all(s >= 0 for s in shear_stress)


def test_calculate_metrics():
    """
    Test R², RMSE, and MAE calculation.

    Verifies that statistical metrics are calculated correctly.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    metrics = _calculate_metrics(y_true, y_pred)

    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics

    # R² should be between 0 and 1 for reasonable predictions
    assert 0 <= metrics["r2"] <= 1

    # RMSE should be positive
    assert metrics["rmse"] > 0

    # MAE should be positive and less than or equal to RMSE
    assert metrics["mae"] > 0
    assert metrics["mae"] <= metrics["rmse"]


def test_normalize_energy():
    """
    Test energy normalization.

    Verifies that energies are normalized relative to minimum value.
    """
    energy = np.array([10.0, 12.0, 11.0, 13.0, 9.0])
    normalized = _normalize_energy(energy)

    # Minimum should be zero
    assert np.min(normalized) == 0.0

    # All values should be non-negative
    assert np.all(normalized >= 0)

    # Relative differences should be preserved
    assert np.isclose(normalized[1] - normalized[0], energy[1] - energy[0])


def test_compare_basic(sample_trajectories, temp_dir):
    """
    Test basic trajectory comparison.

    Verifies that comparison completes and returns expected results.
    """
    traj1_path, traj2_path = sample_trajectories
    output_path = temp_dir / "comparison.png"

    results = compare_trajectories(
        str(traj1_path),
        str(traj2_path),
        output=str(output_path),
        quiet=True,
    )

    # Check that output file was created
    assert output_path.exists()

    # Check results structure
    assert "energy_1" in results
    assert "energy_2" in results
    assert "forces_1" in results
    assert "forces_2" in results
    assert "metrics" in results

    # Check metrics structure
    assert "energy" in results["metrics"]
    assert "forces" in results["metrics"]
    assert "hydro" in results["metrics"]
    assert "shear" in results["metrics"]

    # Each metric should have r2, rmse, mae
    for metric_name in ["energy", "forces", "hydro", "shear"]:
        metric = results["metrics"][metric_name]
        assert "r2" in metric
        assert "rmse" in metric
        assert "mae" in metric


def test_compare_with_normalization(sample_trajectories, temp_dir):
    """
    Test trajectory comparison with energy normalization.

    Verifies that energy normalization works correctly.
    """
    traj1_path, traj2_path = sample_trajectories
    output_path = temp_dir / "comparison_normalized.png"

    results = compare_trajectories(
        str(traj1_path),
        str(traj2_path),
        output=str(output_path),
        normalize_energy=True,
        quiet=True,
    )

    # Check that normalized energies have minimum at zero
    energy_1 = results["energy_1"]
    energy_2 = results["energy_2"]

    assert np.min(energy_1) == 0.0
    assert np.min(energy_2) == 0.0


def test_compare_with_labels(sample_trajectories, temp_dir):
    """
    Test trajectory comparison with custom labels.

    Verifies that custom labels are accepted.
    """
    traj1_path, traj2_path = sample_trajectories
    output_path = temp_dir / "comparison_labeled.png"

    results = compare_trajectories(
        str(traj1_path),
        str(traj2_path),
        output=str(output_path),
        labels=("DFT", "ML"),
        quiet=True,
    )

    assert output_path.exists()
    assert results is not None


def test_compare_different_output_formats(sample_trajectories, temp_dir):
    """
    Test trajectory comparison with different output formats.

    Verifies that different output formats (png, pdf, svg) work.
    """
    traj1_path, traj2_path = sample_trajectories

    for fmt in ["png", "pdf", "svg"]:
        output_path = temp_dir / f"comparison.{fmt}"

        compare_trajectories(
            str(traj1_path),
            str(traj2_path),
            output=str(output_path),
            quiet=True,
        )

        assert output_path.exists()


def test_cli_help():
    """
    Test mlcompare CLI help output.

    Verifies that CLI is accessible and shows help.
    """
    result = subprocess.run(
        ["mlcompare", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "trajectory1" in result.stdout
    assert "trajectory2" in result.stdout


def test_cli_basic(sample_trajectories, temp_dir):
    """
    Test mlcompare CLI basic usage.

    Verifies that CLI can be invoked and produces output.
    """
    traj1_path, traj2_path = sample_trajectories
    output_path = temp_dir / "cli_comparison.png"

    result = subprocess.run(
        [
            "mlcompare",
            str(traj1_path),
            str(traj2_path),
            "-o",
            str(output_path),
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_path.exists()


def test_cli_with_labels(sample_trajectories, temp_dir):
    """
    Test mlcompare CLI with custom labels.

    Verifies that CLI accepts custom labels.
    """
    traj1_path, traj2_path = sample_trajectories
    output_path = temp_dir / "cli_labeled.png"

    result = subprocess.run(
        [
            "mlcompare",
            str(traj1_path),
            str(traj2_path),
            "-o",
            str(output_path),
            "--labels",
            "DFT",
            "CHGNet",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_path.exists()


def test_cli_normalize(sample_trajectories, temp_dir):
    """
    Test mlcompare CLI with energy normalization.

    Verifies that CLI accepts --normalize-energy flag.
    """
    traj1_path, traj2_path = sample_trajectories
    output_path = temp_dir / "cli_normalized.png"

    result = subprocess.run(
        [
            "mlcompare",
            str(traj1_path),
            str(traj2_path),
            "-o",
            str(output_path),
            "--normalize-energy",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_path.exists()


def test_cli_format(sample_trajectories, temp_dir):
    """
    Test mlcompare CLI with different output format.

    Verifies that CLI accepts --format flag.
    """
    traj1_path, traj2_path = sample_trajectories
    output_path = temp_dir / "cli_comparison.pdf"

    result = subprocess.run(
        [
            "mlcompare",
            str(traj1_path),
            str(traj2_path),
            "-o",
            str(output_path),
            "--format",
            "pdf",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_path.exists()


def test_mismatched_lengths(al_structure, temp_dir):
    """
    Test handling of mismatched trajectory lengths.

    Verifies that mismatched lengths are handled by truncating to shorter length.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    structures_1 = [al_structure.copy() for _ in range(3)]
    structures_2 = [al_structure.copy() for _ in range(5)]

    calc = EMT()
    for atoms in structures_1 + structures_2:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)

    traj1_path = temp_dir / "traj1_short.traj"
    traj2_path = temp_dir / "traj2_long.traj"
    write(str(traj1_path), structures_1)
    write(str(traj2_path), structures_2)
    output_path = temp_dir / "mismatch_comparison.png"

    # Should work by truncating to shorter length
    results = compare_trajectories(
        str(traj1_path),
        str(traj2_path),
        output=str(output_path),
        quiet=True,
    )

    # Should only compare first 3 structures
    assert len(results["energy_1"]) == 3
    assert len(results["energy_2"]) == 3
    assert output_path.exists()
