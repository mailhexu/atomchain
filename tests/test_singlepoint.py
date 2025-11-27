"""
Tests for single point energy, forces, and stress calculations.

Tests the calculate_single_point() function and SinglePointResult class
using real structures and ML calculators.

Purpose:
    Verify that single point calculations work correctly with different structures
    and that results can be saved/loaded from YAML files.

How to run:
    pytest tests/test_singlepoint.py -v
    or
    pytest tests/test_singlepoint.py::test_singlepoint_al_basic -v  # run specific test

Note:
    These tests require ML potential packages (CHGNet, M3GNet, etc.) to be installed.
    Tests will be skipped if the required packages are not available.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from atomchain.singlepoint import SinglePointResult, calculate_single_point


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
    """Load PbTiO3 tetragonal structure with Ti displacement."""
    return read(fixtures_dir / "PbTiO3.vasp")


def test_singlepoint_al_basic(al_structure, temp_dir):
    """
    Test basic single point calculation of Al structure.

    Uses default CHGNet calculator.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    result = calculate_single_point(al_structure, calc="chgnet")

    assert result is not None
    assert isinstance(result, SinglePointResult)
    assert result.energy is not None
    assert isinstance(result.energy, float)
    assert result.forces.shape == (len(al_structure), 3)
    assert result.stress.shape == (6,)
    assert result.calculator_name == "chgnet"
    assert result.timestamp is not None


def test_singlepoint_from_file(fixtures_dir, temp_dir):
    """
    Test single point calculation loading structure from file.

    Verifies that atoms can be provided as a file path.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    structure_path = str(fixtures_dir / "Al_fcc.vasp")
    result = calculate_single_point(structure_path, calc="chgnet")

    assert result is not None
    assert isinstance(result, SinglePointResult)
    assert result.energy is not None


def test_singlepoint_yaml_save_load(al_structure, temp_dir):
    """
    Test saving and loading SinglePointResult to/from YAML.

    Verifies YAML serialization and deserialization works correctly.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    # Calculate and save
    result = calculate_single_point(al_structure, calc="chgnet")
    yaml_path = temp_dir / "result.yaml"
    result.to_yaml(str(yaml_path))

    # Check file was created
    assert yaml_path.exists()

    # Load and compare
    loaded_result = SinglePointResult.from_yaml(str(yaml_path))

    assert loaded_result is not None
    assert np.isclose(loaded_result.energy, result.energy)
    assert np.allclose(loaded_result.forces, result.forces)
    assert np.allclose(loaded_result.stress, result.stress)
    assert loaded_result.calculator_name == result.calculator_name
    assert len(loaded_result.atoms) == len(result.atoms)
    assert loaded_result.atoms.get_chemical_formula() == "Al"


def test_singlepoint_srtio3(srtio3_structure, temp_dir):
    """
    Test single point calculation of SrTiO3 cubic perovskite.

    Tests a multi-element structure with 5 atoms.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    result = calculate_single_point(srtio3_structure, calc="chgnet")

    assert result is not None
    assert len(result.atoms) == 5
    assert result.atoms.get_chemical_formula() == "O3SrTi"
    assert result.forces.shape == (5, 3)
    assert result.stress.shape == (6,)
    # Energy should be reasonable (negative for stable structure)
    assert result.energy < 0


def test_singlepoint_pbtio3(pbtio3_structure, temp_dir):
    """
    Test single point calculation of PbTiO3 with Ti displacement.

    Tests structure with broken symmetry.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    result = calculate_single_point(pbtio3_structure, calc="chgnet")

    assert result is not None
    assert len(result.atoms) == 5
    assert result.atoms.get_chemical_formula() == "O3PbTi"
    assert result.forces.shape == (5, 3)

    # For a structure with Ti displacement, forces should be non-zero
    max_force = np.max(np.abs(result.forces))
    print(f"Max force on atoms: {max_force:.4f} eV/Å")
    assert max_force > 0


def test_singlepoint_forces_magnitude(al_structure, temp_dir):
    """
    Test that calculated forces have reasonable magnitudes.

    For a nearly-equilibrium structure, forces should be small.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    result = calculate_single_point(al_structure, calc="chgnet")

    max_force = np.max(np.abs(result.forces))
    # Forces should be relatively small for a bulk crystal
    # (typically < 1 eV/Å for equilibrium structures)
    print(f"Max force: {max_force:.4f} eV/Å")
    assert max_force < 5.0  # Reasonable upper bound


def test_singlepoint_stress_voigt_format(al_structure, temp_dir):
    """
    Test that stress is returned in Voigt notation format.

    Stress should be a 1D array with 6 components: [xx, yy, zz, yz, xz, xy]
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    result = calculate_single_point(al_structure, calc="chgnet")

    assert result.stress.shape == (6,)
    assert isinstance(result.stress, np.ndarray)
    # All stress components should be finite
    assert np.all(np.isfinite(result.stress))


def test_singlepoint_atoms_not_modified(al_structure, temp_dir):
    """
    Test that original atoms object is not modified by calculation.

    The function should make a copy before attaching calculator.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    original_positions = al_structure.get_positions().copy()
    original_cell = al_structure.get_cell().copy()

    result = calculate_single_point(al_structure, calc="chgnet")

    # Original atoms should be unchanged
    assert np.allclose(al_structure.get_positions(), original_positions)
    assert np.allclose(al_structure.get_cell(), original_cell)
    assert al_structure.calc is None  # Calculator should not be attached


def test_singlepoint_with_default_calculator(al_structure, temp_dir):
    """
    Test single point calculation with default calculator.

    Verifies calc="chgnet" is used by default.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    # Default should use chgnet
    result = calculate_single_point(al_structure)

    assert result is not None
    assert result.calculator_name == "chgnet"


def test_singlepoint_yaml_structure_preservation(srtio3_structure, temp_dir):
    """
    Test that atomic structure is correctly preserved in YAML save/load.

    Verifies that positions, cell, symbols, and PBC are correctly stored.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    result = calculate_single_point(srtio3_structure, calc="chgnet")
    yaml_path = temp_dir / "srtio3_result.yaml"
    result.to_yaml(str(yaml_path))

    loaded_result = SinglePointResult.from_yaml(str(yaml_path))

    # Check structure properties
    assert np.allclose(
        loaded_result.atoms.get_positions(), srtio3_structure.get_positions()
    )
    assert np.allclose(loaded_result.atoms.get_cell(), srtio3_structure.get_cell())
    assert (
        loaded_result.atoms.get_chemical_symbols()
        == srtio3_structure.get_chemical_symbols()
    )
    assert np.all(loaded_result.atoms.get_pbc() == srtio3_structure.get_pbc())
