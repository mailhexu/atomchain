"""
Tests for rattle dataset generation.

Tests the generate_rattle_dataset() function for creating training datasets
with random atomic displacements and optional cell deformations.

Purpose:
    Verify that rattle dataset generation works correctly with different parameters
    and that structures are saved/loaded from trajectory files properly.

How to run:
    pytest tests/test_rattle.py -v
    or
    pytest tests/test_rattle.py::test_rattle_basic -v  # run specific test
"""

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read

from atomchain.rattle import generate_rattle_dataset


@pytest.fixture
def al_structure():
    """Create Al fcc structure."""
    return bulk("Al", "fcc", a=4.05)


@pytest.fixture
def srtio3_structure(fixtures_dir):
    """Load SrTiO3 cubic perovskite structure."""
    return read(fixtures_dir / "SrTiO3.vasp")


def test_rattle_basic(al_structure, temp_dir):
    """
    Test basic rattle dataset generation.

    Generates a small dataset with default parameters.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    result = generate_rattle_dataset(
        al_structure, n_struct=10, stdev=0.05, output=str(output), verbose=False
    )

    assert result == str(output)
    assert output.exists()

    # Read and verify
    structures = read(str(output), ":")
    assert len(structures) == 10
    assert all(len(s) == len(al_structure) for s in structures)


def test_rattle_from_file(fixtures_dir, temp_dir):
    """
    Test rattle generation loading structure from file.

    Verifies that atoms can be provided as a file path.
    """
    os.chdir(temp_dir)

    structure_path = str(fixtures_dir / "SrTiO3.vasp")
    output = temp_dir / "dataset.traj"

    result = generate_rattle_dataset(
        structure_path, n_struct=5, stdev=0.05, output=str(output), verbose=False
    )

    assert output.exists()
    structures = read(str(output), ":")
    assert len(structures) == 5


def test_rattle_with_supercell(al_structure, temp_dir):
    """
    Test rattle generation with supercell expansion.

    Verifies that supercell is created before rattling.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    result = generate_rattle_dataset(
        al_structure,
        n_struct=5,
        stdev=0.05,
        supercell=2,  # 2×2×2
        output=str(output),
        verbose=False,
    )

    structures = read(str(output), ":")
    assert len(structures) == 5
    # 2×2×2 supercell should have 8 times the atoms
    assert all(len(s) == 8 * len(al_structure) for s in structures)


def test_rattle_with_diagonal_supercell(al_structure, temp_dir):
    """
    Test rattle generation with diagonal supercell.

    Uses non-uniform supercell expansion [2,2,3].
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    result = generate_rattle_dataset(
        al_structure,
        n_struct=5,
        stdev=0.05,
        supercell=[2, 2, 3],
        output=str(output),
        verbose=False,
    )

    structures = read(str(output), ":")
    assert len(structures) == 5
    # 2×2×3 supercell should have 12 times the atoms
    assert all(len(s) == 12 * len(al_structure) for s in structures)


def test_rattle_with_cell_strain(al_structure, temp_dir):
    """
    Test rattle generation with cell strain.

    Verifies that cell vectors are perturbed.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    result = generate_rattle_dataset(
        al_structure,
        n_struct=10,
        stdev=0.05,
        cell_stdev=0.02,
        output=str(output),
        verbose=False,
    )

    structures = read(str(output), ":")
    assert len(structures) == 10

    # Check that cell vectors are different from original
    original_cell = al_structure.get_cell()
    for s in structures:
        cell = s.get_cell()
        # Cells should be different (perturbed)
        assert not np.allclose(cell, original_cell, atol=1e-6)


def test_rattle_reproducible_with_seed(al_structure, temp_dir):
    """
    Test that rattle generation is reproducible with seed.

    Generates two datasets with same seed and verifies they are identical.
    """
    os.chdir(temp_dir)

    output1 = temp_dir / "dataset1.traj"
    output2 = temp_dir / "dataset2.traj"

    # Generate first dataset
    generate_rattle_dataset(
        al_structure,
        n_struct=5,
        stdev=0.05,
        seed=42,
        output=str(output1),
        verbose=False,
    )

    # Generate second dataset with same seed
    generate_rattle_dataset(
        al_structure,
        n_struct=5,
        stdev=0.05,
        seed=42,
        output=str(output2),
        verbose=False,
    )

    # Read both datasets
    structures1 = read(str(output1), ":")
    structures2 = read(str(output2), ":")

    # Verify they are identical
    assert len(structures1) == len(structures2)
    for s1, s2 in zip(structures1, structures2):
        assert np.allclose(s1.get_positions(), s2.get_positions())
        assert np.allclose(s1.get_cell(), s2.get_cell())


def test_rattle_different_without_seed(al_structure, temp_dir):
    """
    Test that rattle generation produces different structures without seed.

    Generates two datasets without seed and verifies they are different.
    """
    os.chdir(temp_dir)

    output1 = temp_dir / "dataset1.traj"
    output2 = temp_dir / "dataset2.traj"

    # Generate first dataset
    generate_rattle_dataset(
        al_structure, n_struct=5, stdev=0.05, output=str(output1), verbose=False
    )

    # Generate second dataset
    generate_rattle_dataset(
        al_structure, n_struct=5, stdev=0.05, output=str(output2), verbose=False
    )

    # Read both datasets
    structures1 = read(str(output1), ":")
    structures2 = read(str(output2), ":")

    # Verify they are different
    different = False
    for s1, s2 in zip(structures1, structures2):
        if not np.allclose(s1.get_positions(), s2.get_positions(), atol=1e-6):
            different = True
            break

    assert different, "Structures should be different without seed"


def test_rattle_zero_stdev(al_structure, temp_dir):
    """
    Test rattle with zero stdev (no displacement).

    All structures should be identical to original.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    generate_rattle_dataset(
        al_structure, n_struct=5, stdev=0.0, output=str(output), verbose=False
    )

    structures = read(str(output), ":")
    original_pos = al_structure.get_positions()

    for s in structures:
        assert np.allclose(s.get_positions(), original_pos)


def test_rattle_validation_errors(al_structure, temp_dir):
    """
    Test that validation errors are raised for invalid inputs.
    """
    os.chdir(temp_dir)

    # Negative n_struct
    with pytest.raises(ValueError, match="n_struct must be >= 1"):
        generate_rattle_dataset(al_structure, n_struct=0, verbose=False)

    # Negative stdev
    with pytest.raises(ValueError, match="stdev must be non-negative"):
        generate_rattle_dataset(al_structure, n_struct=5, stdev=-0.1, verbose=False)

    # Negative cell_stdev
    with pytest.raises(ValueError, match="cell_stdev must be non-negative"):
        generate_rattle_dataset(
            al_structure, n_struct=5, cell_stdev=-0.1, verbose=False
        )


def test_rattle_positions_are_displaced(al_structure, temp_dir):
    """
    Test that atomic positions are actually displaced.

    Verifies that rattle actually changes positions with non-zero stdev.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    generate_rattle_dataset(
        al_structure,
        n_struct=10,
        stdev=0.1,  # Large stdev for clear displacement
        output=str(output),
        verbose=False,
    )

    structures = read(str(output), ":")
    original_pos = al_structure.get_positions()

    # All structures should have displaced positions
    for s in structures:
        positions = s.get_positions()
        # Check that positions are different
        assert not np.allclose(positions, original_pos, atol=1e-6)
        # Check that displacement magnitude is reasonable (within ~3σ)
        displacement = np.linalg.norm(positions - original_pos, axis=1)
        assert np.all(displacement < 0.5)  # Should be within reasonable bounds


def test_rattle_verbose_output(al_structure, temp_dir, capsys):
    """
    Test that verbose mode produces output.

    Verifies progress messages are printed.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    generate_rattle_dataset(
        al_structure, n_struct=25, stdev=0.05, output=str(output), verbose=True
    )

    captured = capsys.readouterr()
    assert "[Rattle] Generating" in captured.out
    assert "[Rattle] Progress:" in captured.out
    assert "[Rattle] Dataset generation complete!" in captured.out


def test_rattle_quiet_mode(al_structure, temp_dir, capsys):
    """
    Test that quiet mode suppresses output.

    Verifies no progress messages are printed with verbose=False.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    generate_rattle_dataset(
        al_structure, n_struct=10, stdev=0.05, output=str(output), verbose=False
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_rattle_combined_strain_and_displacement(al_structure, temp_dir):
    """
    Test rattle with both cell strain and atomic displacement.

    Verifies that both perturbations are applied correctly.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    generate_rattle_dataset(
        al_structure,
        n_struct=10,
        stdev=0.05,
        cell_stdev=0.02,
        output=str(output),
        verbose=False,
    )

    structures = read(str(output), ":")
    original_pos = al_structure.get_positions()
    original_cell = al_structure.get_cell()

    for s in structures:
        # Check both positions and cell are perturbed
        assert not np.allclose(s.get_positions(), original_pos, atol=1e-6)
        assert not np.allclose(s.get_cell(), original_cell, atol=1e-6)


def test_rattle_large_dataset(al_structure, temp_dir):
    """
    Test rattle generation with larger dataset.

    Verifies that generation works efficiently for many structures.
    """
    os.chdir(temp_dir)

    output = temp_dir / "large_dataset.traj"
    n_struct = 100

    result = generate_rattle_dataset(
        al_structure, n_struct=n_struct, stdev=0.05, output=str(output), verbose=False
    )

    assert output.exists()
    structures = read(str(output), ":")
    assert len(structures) == n_struct


def test_rattle_supercell_and_strain(al_structure, temp_dir):
    """
    Test rattle with supercell and cell strain combined.

    Verifies all features work together.
    """
    os.chdir(temp_dir)

    output = temp_dir / "dataset.traj"
    generate_rattle_dataset(
        al_structure,
        n_struct=10,
        stdev=0.05,
        supercell=[2, 2, 2],
        cell_stdev=0.02,
        seed=42,
        output=str(output),
        verbose=False,
    )

    structures = read(str(output), ":")
    assert len(structures) == 10
    # Check supercell size
    assert all(len(s) == 8 * len(al_structure) for s in structures)
    # Check that cells are strained
    original_cell = bulk("Al", "fcc", a=4.05).get_cell()
    supercell_cell = original_cell * 2  # Approximate supercell cell
    for s in structures:
        # Cells should be different due to strain
        cell = s.get_cell()
        # Just verify cell is not exactly the unperturbed supercell
        assert not np.allclose(cell, supercell_cell, atol=1e-6)
