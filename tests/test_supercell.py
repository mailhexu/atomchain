"""
Tests for supercell generation.

Tests the make_supercell_structure() function using various input formats
and validates supercell properties.

Purpose:
    Verify that supercell generation works correctly with different input types
    (scalar, diagonal, matrix) and that atom counts and cell vectors are correct.

How to run:
    pytest tests/test_supercell.py -v
    or
    pytest tests/test_supercell.py::test_supercell_scalar -v  # run specific test

Note:
    These tests use ASE's make_supercell which is well-tested. We mainly verify
    our wrapper logic and input parsing.
"""

import os

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read

from atomchain.supercell import make_supercell_structure


@pytest.fixture
def al_structure(fixtures_dir):
    """Load Al fcc structure."""
    return read(fixtures_dir / "Al_fcc.vasp")


@pytest.fixture
def srtio3_structure(fixtures_dir):
    """Load SrTiO3 cubic perovskite structure."""
    return read(fixtures_dir / "SrTiO3.vasp")


@pytest.fixture
def pbtio3_structure(fixtures_dir):
    """Load PbTiO3 tetragonal structure."""
    return read(fixtures_dir / "PbTiO3.vasp")


def test_supercell_scalar_input(al_structure):
    """
    Test isotropic supercell generation from scalar input.

    Scalar 2 should create 2×2×2 supercell.
    """
    supercell = make_supercell_structure(al_structure, 2)

    # Check atom count: should be 8× original
    assert len(supercell) == 8 * len(al_structure)

    # Check formula preserved
    assert supercell.get_chemical_formula() == "Al8"


def test_supercell_diagonal_input(al_structure):
    """
    Test diagonal supercell generation from list input.

    [2, 2, 3] should create 2×2×3 supercell.
    """
    supercell = make_supercell_structure(al_structure, [2, 2, 3])

    # Check atom count: should be 12× original (2*2*3)
    assert len(supercell) == 12 * len(al_structure)

    # Check formula
    assert supercell.get_chemical_formula() == "Al12"


def test_supercell_matrix_input(al_structure):
    """
    Test general matrix supercell generation.

    [[2,0,0],[0,2,0],[0,0,3]] should create 2×2×3 supercell.
    """
    P = [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
    supercell = make_supercell_structure(al_structure, P)

    # Check atom count: det(P) = 12
    assert len(supercell) == 12 * len(al_structure)


def test_supercell_numpy_array_input(al_structure):
    """
    Test supercell generation with numpy array input.
    """
    P = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    supercell = make_supercell_structure(al_structure, P)

    # Check atom count: det(P) = 8
    assert len(supercell) == 8 * len(al_structure)


def test_supercell_from_file(fixtures_dir, temp_dir):
    """
    Test supercell generation from file path.
    """
    os.chdir(temp_dir)
    structure_path = str(fixtures_dir / "Al_fcc.vasp")

    supercell = make_supercell_structure(structure_path, [2, 2, 2])

    assert len(supercell) == 8  # Original has 1 atom


def test_supercell_cell_vectors(al_structure):
    """
    Test that supercell cell vectors are correct.

    New cell = P @ old_cell
    """
    P = [[2, 0, 0], [0, 3, 0], [0, 0, 2]]
    supercell = make_supercell_structure(al_structure, P)

    # Calculate expected cell
    original_cell = al_structure.get_cell()
    expected_cell = np.dot(P, original_cell)

    # Check cell vectors match
    assert np.allclose(supercell.get_cell(), expected_cell)


def test_supercell_preserves_composition(srtio3_structure):
    """
    Test that chemical composition is preserved in supercell.

    SrTiO3 → Sr8Ti8O24 for 2×2×2 supercell
    """
    supercell = make_supercell_structure(srtio3_structure, 2)

    # Original: SrTiO3 (5 atoms)
    # Supercell: 8× multiplication
    assert len(supercell) == 8 * len(srtio3_structure)
    assert supercell.get_chemical_formula() == "O24Sr8Ti8"


def test_supercell_atom_ordering_cell_major(al_structure):
    """
    Test cell-major atom ordering (default).
    """
    supercell = make_supercell_structure(al_structure, [2, 1, 1], order="cell-major")

    # Should have 2 atoms
    assert len(supercell) == 2


def test_supercell_atom_ordering_atom_major(al_structure):
    """
    Test atom-major ordering (VASP-friendly).
    """
    supercell = make_supercell_structure(al_structure, [2, 1, 1], order="atom-major")

    # Should have 2 atoms
    assert len(supercell) == 2


def test_supercell_wrap_positions(al_structure):
    """
    Test that atomic positions are wrapped by default.
    """
    supercell = make_supercell_structure(al_structure, 2, wrap=True)

    # All scaled positions should be in [0, 1)
    scaled_positions = supercell.get_scaled_positions()
    assert np.all(scaled_positions >= 0)
    assert np.all(scaled_positions < 1)


def test_supercell_no_wrap(al_structure):
    """
    Test supercell generation without wrapping.
    """
    supercell = make_supercell_structure(al_structure, 2, wrap=False)

    # Should still create valid supercell
    assert len(supercell) == 8


def test_supercell_original_not_modified(al_structure):
    """
    Test that original structure is not modified.
    """
    original_positions = al_structure.get_positions().copy()
    original_cell = al_structure.get_cell().copy()
    original_natoms = len(al_structure)

    _ = make_supercell_structure(al_structure, [2, 2, 2])

    # Original should be unchanged
    assert len(al_structure) == original_natoms
    assert np.allclose(al_structure.get_positions(), original_positions)
    assert np.allclose(al_structure.get_cell(), original_cell)


def test_supercell_invalid_matrix_non_integer():
    """
    Test that non-integer matrix raises ValueError.
    """
    atoms = bulk("Al", "fcc", a=4.05)
    P = [[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]]

    with pytest.raises(ValueError, match="must contain only integers"):
        make_supercell_structure(atoms, P)


def test_supercell_invalid_matrix_zero_det():
    """
    Test that matrix with det=0 raises ValueError.
    """
    atoms = bulk("Al", "fcc", a=4.05)
    P = [[2, 0, 0], [0, 2, 0], [0, 0, 0]]  # det = 0

    with pytest.raises(ValueError, match="determinant must be positive"):
        make_supercell_structure(atoms, P)


def test_supercell_invalid_matrix_negative_det():
    """
    Test that matrix with negative det raises ValueError.
    """
    atoms = bulk("Al", "fcc", a=4.05)
    P = [[2, 0, 0], [0, 2, 0], [0, 0, -2]]  # det = -8

    with pytest.raises(ValueError, match="determinant must be positive"):
        make_supercell_structure(atoms, P)


def test_supercell_invalid_input_shape():
    """
    Test that invalid input shape raises ValueError.
    """
    atoms = bulk("Al", "fcc", a=4.05)
    P = [2, 2]  # Wrong length

    with pytest.raises(ValueError, match="Invalid supercell matrix shape"):
        make_supercell_structure(atoms, P)


def test_supercell_different_structures(pbtio3_structure):
    """
    Test supercell generation on tetragonal PbTiO3.
    """
    supercell = make_supercell_structure(pbtio3_structure, [2, 2, 1])

    # Should multiply by 4 (2*2*1)
    assert len(supercell) == 4 * len(pbtio3_structure)
    assert supercell.get_chemical_formula() == "O12Pb4Ti4"


def test_supercell_large_multiplication():
    """
    Test larger supercell multiplication.
    """
    atoms = bulk("Al", "fcc", a=4.05)
    supercell = make_supercell_structure(atoms, [3, 3, 3])

    # Should have 27 atoms
    assert len(supercell) == 27


def test_supercell_non_diagonal_matrix():
    """
    Test non-diagonal transformation matrix.
    """
    atoms = bulk("Al", "fcc", a=4.05)
    P = [[2, 1, 0], [0, 2, 0], [0, 0, 2]]  # det = 8

    supercell = make_supercell_structure(atoms, P)

    # det(P) = 8
    assert len(supercell) == 8
