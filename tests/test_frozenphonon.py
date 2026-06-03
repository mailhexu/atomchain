"""
Tests for frozen phonon calculations.

Tests the calculate_phonon() and compute_phonon_at_qpoints() functions
using real structures and ML calculators.

Purpose:
    Verify that frozen phonon calculations work correctly with different structures
    and ML potential calculators (MACE, etc.)

How to run:
    pytest tests/test_frozenphonon.py -v
    or
    pytest tests/test_frozenphonon.py::test_calculate_phonon_al_basic -v  # run specific test

Note:
    These tests require ML potential packages (MACE, etc.) to be installed.
    Tests will be skipped if the required packages are not available.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from atomchain.phonon import calculate_phonon, compute_phonon_at_qpoints


@pytest.fixture
def al_structure(fixtures_dir):
    """Load Al fcc structure."""
    return read(fixtures_dir / "Al_fcc.vasp")


@pytest.fixture
def srtio3_structure(fixtures_dir):
    """Load SrTiO3 cubic perovskite structure."""
    return read(fixtures_dir / "SrTiO3.vasp")


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def mace_calculator():
    """Initialize MACE calculator for tests."""
    from atomchain.init_model import init_calc

    return init_calc(model_type="mace")


def test_calculate_phonon_al_basic(al_structure, mace_calculator, temp_dir):
    """
    Test basic frozen phonon calculation of Al structure.

    This test verifies that frozen phonon calculations can be performed
    successfully on a simple FCC metal structure.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Test with MACE calculator
        phonon = calculate_phonon(
            al_structure,
            calc=mace_calculator,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # 2x2x2 supercell
            distance=0.01,  # Smaller displacement for faster test
            parallel=False,  # Disable parallel for simpler testing
        )

        # Verify that phonon calculation completed
        assert phonon is not None

        # Check that phonon object has expected attributes
        assert hasattr(phonon, "force_constants")
        assert hasattr(phonon, "primitive")

        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()
        assert (phonon_dir / "FORCE_CONSTANTS").exists()

    finally:
        os.chdir(original_dir)


def test_calculate_phonon_srtio3_basic(srtio3_structure, mace_calculator, temp_dir):
    """
    Test frozen phonon calculation of SrTiO3 perovskite structure.

    This test verifies that frozen phonon calculations work on
    more complex oxide structures.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Test with MACE calculator
        phonon = calculate_phonon(
            srtio3_structure,
            calc=mace_calculator,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
            parallel=False,
        )

        # Verify that phonon calculation completed
        assert phonon is not None

        # Check that phonon object has expected attributes
        assert hasattr(phonon, "force_constants")
        assert hasattr(phonon, "primitive")

        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()
        assert (phonon_dir / "FORCE_CONSTANTS").exists()

    finally:
        os.chdir(original_dir)


def test_calculate_phonon_with_restart(al_structure, mace_calculator, temp_dir):
    """
    Test frozen phonon calculation with restart capability.

    This test verifies that the restart functionality works correctly.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # First calculation
        phonon1 = calculate_phonon(
            al_structure,
            calc=mace_calculator,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
            parallel=False,
            restart=True,
        )

        # Verify that first calculation completed
        assert phonon1 is not None

        # Second calculation with restart should use pickle file
        phonon2 = calculate_phonon(
            al_structure,
            calc=mace_calculator,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
            parallel=False,
            restart=True,
        )

        # Verify that second calculation completed
        assert phonon2 is not None

        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()
        assert (phonon_dir / "FORCE_CONSTANTS").exists()

    finally:
        os.chdir(original_dir)


def test_compute_phonon_at_qpoints(al_structure, mace_calculator, temp_dir):
    """
    Test computing phonon frequencies at specific q-points.

    This test verifies that the compute_phonon_at_qpoints function
    works correctly after a phonon calculation.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # First perform phonon calculation
        phonon = calculate_phonon(
            al_structure,
            calc=mace_calculator,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
            parallel=False,
        )

        # Verify that phonon calculation completed
        assert phonon is not None

        # Test computing phonon at Gamma point
        qpoints = [[0, 0, 0]]  # Gamma point
        result = compute_phonon_at_qpoints(qpoints)

        # Verify that computation completed
        assert result is not None
        assert "frequencies" in result
        assert "eigenvectors" in result
        assert "qpoints" in result
        assert "nbands" in result

        # Check result structure
        assert result["frequencies"].shape == (1, result["nbands"])
        assert result["eigenvectors"].shape == (
            1,
            result["nbands"],
            len(al_structure),
            3,
        )

        # Frequencies should be real numbers
        assert np.all(np.isreal(result["frequencies"]))

    finally:
        os.chdir(original_dir)


def test_compute_phonon_at_multiple_qpoints(al_structure, mace_calculator, temp_dir):
    """
    Test computing phonon frequencies at multiple q-points.

    This test verifies that the compute_phonon_at_qpoints function
    works correctly with multiple q-points.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # First perform phonon calculation
        phonon = calculate_phonon(
            al_structure,
            calc=mace_calculator,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
            parallel=False,
        )

        # Verify that phonon calculation completed
        assert phonon is not None

        # Test computing phonon at multiple q-points
        qpoints = [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]]  # Gamma, X, M points
        result = compute_phonon_at_qpoints(qpoints)

        # Verify that computation completed
        assert result is not None
        assert "frequencies" in result
        assert "eigenvectors" in result
        assert "qpoints" in result
        assert "nbands" in result

        # Check result structure
        assert result["frequencies"].shape == (3, result["nbands"])
        assert result["eigenvectors"].shape == (
            3,
            result["nbands"],
            len(al_structure),
            3,
        )
        assert result["qpoints"].shape == (3, 3)

        # Frequencies should be real numbers
        assert np.all(np.isreal(result["frequencies"]))

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
