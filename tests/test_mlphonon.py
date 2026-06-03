"""
Tests for ML phonon calculations.

Tests the phonon_with_ml() function using real structures and ML calculators.

Purpose:
    Verify that phonon calculations work correctly with different structures
    and ML potential calculators (MACE, M3GNet, etc.)

How to run:
    pytest tests/test_mlphonon.py -v
    or
    pytest tests/test_mlphonon.py::test_phonon_al_basic -v  # run specific test

Note:
    These tests require ML potential packages (MACE, M3GNet, etc.) to be installed.
    Tests will be skipped if the required packages are not available.
"""

import os
from pathlib import Path

import pytest
from ase.io import read

from atomchain.phonon import phonon_with_ml


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


def test_phonon_al_basic(al_structure, temp_dir):
    """
    Test basic phonon calculation of Al structure without relaxation.

    This test verifies that phonon calculations can be performed
    successfully on a simple FCC metal structure.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Test with MACE calculator
        result = phonon_with_ml(
            al_structure,
            calc="mace",
            relax=False,
            plot=False,  # Don't plot during tests
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # 2x2x2 supercell
            distance=0.01,  # Smaller displacement for faster test
        )

        # Verify that phonon calculation completed and returned relaxed atoms
        assert result is not None
        assert hasattr(result, "get_positions")  # Should be an ASE Atoms object

        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()

    finally:
        os.chdir(original_dir)


def test_phonon_al_with_relax(al_structure, temp_dir):
    """
    Test phonon calculation of Al structure with relaxation.

    This test verifies that the relaxation + phonon workflow
    works correctly.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Test with relaxation enabled
        result = phonon_with_ml(
            al_structure,
            calc="mace",
            relax=True,
            plot=False,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
        )

        # Verify that phonon calculation completed and returned relaxed atoms
        assert result is not None
        assert hasattr(result, "get_positions")  # Should be an ASE Atoms object

        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()

    finally:
        os.chdir(original_dir)


def test_phonon_srtio3_basic(srtio3_structure, temp_dir):
    """
    Test phonon calculation of SrTiO3 perovskite structure.

    This test verifies that phonon calculations work on
    more complex oxide structures.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Test with MACE calculator
        result = phonon_with_ml(
            srtio3_structure,
            calc="mace",
            relax=False,
            plot=False,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
        )

        # Verify that phonon calculation completed and returned relaxed atoms
        assert result is not None
        assert hasattr(result, "get_positions")  # Should be an ASE Atoms object

        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()

    finally:
        os.chdir(original_dir)


def test_phonon_with_kpath(al_structure, temp_dir):
    """
    Test phonon calculation with custom k-path.

    This test verifies that custom k-paths work correctly
    for band structure plotting.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Test with custom k-path
        result = phonon_with_ml(
            al_structure,
            calc="mace",
            relax=False,
            plot=False,
            knames="GXWG",  # Simple k-path for FCC
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
        )

        # Verify that phonon calculation completed and returned relaxed atoms
        assert result is not None
        assert hasattr(result, "get_positions")  # Should be an ASE Atoms object

        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()

    finally:
        os.chdir(original_dir)


def test_phonon_with_custom_calculator(al_structure, temp_dir):
    """
    Test phonon calculation with pre-initialized calculator.

    This test verifies that passing a calculator object
    (instead of string) works correctly.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Initialize calculator first
        from atomchain.init_model import init_calc

        calculator = init_calc(model_type="mace")

        # Test with pre-initialized calculator
        result = phonon_with_ml(
            al_structure,
            calc=calculator,
            relax=False,
            plot=False,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
        )

        # Verify that phonon calculation completed and returned relaxed atoms
        assert result is not None
        assert hasattr(result, "get_positions")  # Should be an ASE Atoms object

        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
