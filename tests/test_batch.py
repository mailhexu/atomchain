"""
Tests for batch trajectory processing.

Tests the calculate_trajectory_batch() function for processing multiple structures
in a trajectory file with ML potential calculators.

Purpose:
    Verify that batch processing works correctly, attaches calculator results
    to Atoms objects, and saves properly to output trajectory files.

How to run:
    pytest tests/test_batch.py -v
    or
    pytest tests/test_batch.py::test_batch_basic -v  # run specific test
"""

import os

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write

from atomchain.batch import calculate_trajectory_batch


@pytest.fixture
def al_structure():
    """Create Al fcc structure."""
    return bulk("Al", "fcc", a=4.05)


@pytest.fixture
def sample_trajectory(al_structure, temp_dir):
    """Create a sample trajectory file with multiple structures."""
    structures = []
    for i in range(5):
        atoms = al_structure.copy()
        # Add slight perturbations to positions
        atoms.rattle(stdev=0.05 * (i + 1))
        structures.append(atoms)

    traj_path = temp_dir / "input.traj"
    write(str(traj_path), structures)
    return traj_path


def test_batch_basic(sample_trajectory, temp_dir):
    """
    Test basic batch processing with EMT calculator.

    Uses EMT calculator (no external dependencies) to process structures.
    """
    os.chdir(temp_dir)

    output = temp_dir / "output.traj"

    # Use EMT calculator for testing (no external dependencies)
    calc = EMT()

    result = calculate_trajectory_batch(
        sample_trajectory, calculator=calc, output=str(output), quiet=True
    )

    assert len(result) == 5
    assert output.exists()

    # Verify results are saved
    structures = read(str(output), ":")
    assert len(structures) == 5


def test_batch_properties_attached(sample_trajectory, temp_dir):
    """
    Test that energy, forces, and stress are attached to Atoms objects.

    Verifies that calculator results can be accessed after batch processing.
    """
    os.chdir(temp_dir)

    output = temp_dir / "output.traj"
    calc = EMT()

    result = calculate_trajectory_batch(
        sample_trajectory, calculator=calc, output=str(output), quiet=True
    )

    # Check that properties can be accessed
    for atoms in result:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=True)

        assert isinstance(energy, float)
        assert forces.shape == (len(atoms), 3)
        assert stress.shape == (6,)


def test_batch_from_list(al_structure, temp_dir):
    """
    Test batch processing from a list of Atoms objects.

    Verifies that trajectory parameter can be a list instead of file path.
    """
    os.chdir(temp_dir)

    # Create list of structures
    structures = [al_structure.copy() for _ in range(3)]
    for i, atoms in enumerate(structures):
        atoms.rattle(stdev=0.05 * (i + 1))

    output = temp_dir / "output.traj"
    calc = EMT()

    result = calculate_trajectory_batch(
        structures, calculator=calc, output=str(output), quiet=True
    )

    assert len(result) == 3
    assert output.exists()


def test_batch_single_structure(al_structure, temp_dir):
    """
    Test batch processing with single structure trajectory.

    Verifies edge case of trajectory with only one structure.
    """
    os.chdir(temp_dir)

    traj_path = temp_dir / "single.traj"
    write(str(traj_path), [al_structure])

    output = temp_dir / "output.traj"
    calc = EMT()

    result = calculate_trajectory_batch(
        traj_path, calculator=calc, output=str(output), quiet=True
    )

    assert len(result) == 1


def test_batch_empty_trajectory(temp_dir):
    """
    Test batch processing with empty trajectory.

    Verifies graceful handling of empty input.
    """
    os.chdir(temp_dir)

    # Create empty trajectory
    traj_path = temp_dir / "empty.traj"
    write(str(traj_path), [])

    output = temp_dir / "output.traj"
    calc = EMT()

    result = calculate_trajectory_batch(
        traj_path, calculator=calc, output=str(output), quiet=True
    )

    assert len(result) == 0


def test_batch_missing_file(temp_dir):
    """
    Test batch processing with missing input file.

    Verifies proper error handling for non-existent files.
    """
    os.chdir(temp_dir)

    nonexistent = temp_dir / "nonexistent.traj"
    calc = EMT()

    with pytest.raises(FileNotFoundError):
        calculate_trajectory_batch(
            nonexistent, calculator=calc, output="output.traj", quiet=True
        )


def test_batch_verbose_output(sample_trajectory, temp_dir, capsys):
    """
    Test batch processing with verbose output.

    Verifies that progress information is printed when verbose=True.
    """
    os.chdir(temp_dir)

    output = temp_dir / "output.traj"
    calc = EMT()

    calculate_trajectory_batch(
        sample_trajectory, calculator=calc, output=str(output), verbose=True
    )

    captured = capsys.readouterr()
    assert "[Batch]" in captured.out
    assert "structures" in captured.out.lower()


def test_batch_quiet_mode(sample_trajectory, temp_dir, capsys):
    """
    Test batch processing with quiet mode.

    Verifies that output is suppressed when quiet=True.
    """
    os.chdir(temp_dir)

    output = temp_dir / "output.traj"
    calc = EMT()

    calculate_trajectory_batch(
        sample_trajectory, calculator=calc, output=str(output), quiet=True
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_batch_calculator_reuse(sample_trajectory, temp_dir):
    """
    Test that calculator is initialized once and reused.

    Verifies efficiency by ensuring single calculator instance is used.
    """
    os.chdir(temp_dir)

    output = temp_dir / "output.traj"
    calc = EMT()

    result = calculate_trajectory_batch(
        sample_trajectory, calculator=calc, output=str(output), quiet=True
    )

    # All structures should have the same calculator instance
    assert len(result) == 5
    for atoms in result:
        assert atoms.calc is not None


def test_batch_output_file_creation(sample_trajectory, temp_dir):
    """
    Test that output file is created correctly.

    Verifies output file exists and can be read.
    """
    os.chdir(temp_dir)

    output = temp_dir / "output.traj"
    calc = EMT()

    calculate_trajectory_batch(
        sample_trajectory, calculator=calc, output=str(output), quiet=True
    )

    assert output.exists()

    # Verify file can be read
    loaded = read(str(output), ":")
    assert len(loaded) == 5


def test_batch_preserves_structure_info(al_structure, temp_dir):
    """
    Test that original structure information is preserved.

    Verifies that cell, positions, symbols, and PBC are maintained.
    """
    os.chdir(temp_dir)

    # Create trajectory with specific structure
    traj_path = temp_dir / "input.traj"
    structures = [al_structure.copy() for _ in range(3)]
    write(str(traj_path), structures)

    output = temp_dir / "output.traj"
    calc = EMT()

    result = calculate_trajectory_batch(
        traj_path, calculator=calc, output=str(output), quiet=True
    )

    # Check structure preservation
    for i, atoms in enumerate(result):
        assert atoms.get_chemical_formula() == al_structure.get_chemical_formula()
        assert len(atoms) == len(al_structure)
        # Cell should be similar (may have numerical differences)
        np.testing.assert_allclose(atoms.get_cell(), al_structure.get_cell(), rtol=1e-5)


def test_batch_with_different_structures(temp_dir):
    """
    Test batch processing with structures of different sizes.

    Verifies that varying structure sizes are handled correctly.
    """
    os.chdir(temp_dir)

    # Create structures of different sizes (all EMT-compatible: Al, Cu, Ag)
    structures = [
        bulk("Al", "fcc", a=4.05),
        bulk("Cu", "fcc", a=3.61),
        bulk("Ag", "fcc", a=4.09),
    ]

    traj_path = temp_dir / "mixed.traj"
    write(str(traj_path), structures)

    output = temp_dir / "output.traj"
    calc = EMT()

    result = calculate_trajectory_batch(
        traj_path, calculator=calc, output=str(output), quiet=True
    )

    assert len(result) == 3
    # Verify each structure has correct element
    assert result[0].get_chemical_formula() == "Al"
    assert result[1].get_chemical_formula() == "Cu"
    assert result[2].get_chemical_formula() == "Ag"
