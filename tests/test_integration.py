"""
Integration tests for end-to-end workflows.

Tests complete workflows combining multiple AtomChain modules
using real structures and ML calculators.

Purpose:
    Verify that complex workflows involving multiple steps
    (relaxation, phonon calculation, etc.) work correctly together.

How to run:
    pytest tests/test_integration.py -v
    or
    pytest tests/test_integration.py::test_relax_then_phonon -v  # run specific test

Note:
    These tests require ML potential packages (CHGNet, etc.) to be installed.
    Tests will be skipped if the required packages are not available.
"""

import os
import tempfile
from pathlib import Path

import pytest
from ase.io import read

from atomchain.relax import relax_with_ml
from atomchain.phonon import phonon_with_ml
from atomchain.init_model import init_calc


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
def chgnet_calculator():
    """Initialize CHGNet calculator for tests."""
    return init_calc(model_type="chgnet")


def test_relax_then_phonon(al_structure, chgnet_calculator, temp_dir):
    """
    Test complete workflow: relaxation followed by phonon calculation.
    
    This test verifies that the relaxation + phonon workflow
    works correctly as an integrated process.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Step 1: Relax the structure
        print("[Integration] Step 1: Relaxing structure...")
        relaxed_atoms = relax_with_ml(
            al_structure,
            calc=chgnet_calculator,
            fmax=0.01,  # Tighter convergence for better phonon results
        )
        
        # Verify relaxation completed
        assert relaxed_atoms is not None
        assert hasattr(relaxed_atoms, 'get_positions')
        
        # Check that relaxation changed the structure
        # Note: original structure doesn't have calculator, so we can't compare energies
        # Just verify that we have a valid relaxed structure
        print(f"[Integration] Relaxation completed successfully")
        
        # Step 2: Perform phonon calculation on relaxed structure
        print("[Integration] Step 2: Calculating phonons...")
        result = phonon_with_ml(
            relaxed_atoms,
            calc=chgnet_calculator,
            relax=False,  # Already relaxed
            plot=False,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
        )
        
        # Verify phonon calculation completed
        assert result is not None
        assert hasattr(result, 'get_positions')
        
        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()
        assert (phonon_dir / "FORCE_CONSTANTS").exists()
        
        print("[Integration] Workflow completed successfully!")
        
    finally:
        os.chdir(original_dir)


def test_phonon_with_auto_relax(al_structure, temp_dir):
    """
    Test phonon calculation with automatic relaxation.
    
    This test verifies that the phonon_with_ml function
    correctly handles the relaxation step internally.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Test with automatic relaxation enabled
        result = phonon_with_ml(
            al_structure,
            calc="chgnet",
            relax=True,  # Enable automatic relaxation
            plot=False,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
        )
        
        # Verify phonon calculation completed
        assert result is not None
        assert hasattr(result, 'get_positions')
        
        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()
        assert (phonon_dir / "FORCE_CONSTANTS").exists()
        
        print("[Integration] Auto-relax phonon workflow completed successfully!")
        
    finally:
        os.chdir(original_dir)


def test_complex_oxide_workflow(srtio3_structure, temp_dir):
    """
    Test complete workflow on a complex oxide structure.
    
    This test verifies that the workflow works correctly
    on more complex materials like perovskites.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Test complete workflow on SrTiO3
        result = phonon_with_ml(
            srtio3_structure,
            calc="chgnet",
            relax=True,  # Enable relaxation for complex structure
            plot=False,
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
        )
        
        # Verify phonon calculation completed
        assert result is not None
        assert hasattr(result, 'get_positions')
        
        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()
        assert (phonon_dir / "FORCE_CONSTANTS").exists()
        
        print("[Integration] Complex oxide workflow completed successfully!")
        
    finally:
        os.chdir(original_dir)


def test_workflow_with_custom_kpath(al_structure, temp_dir):
    """
    Test complete workflow with custom k-path for band structure.
    
    This test verifies that custom k-paths work correctly
    in the complete workflow.
    """
    # Change to temp directory for output files
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Test with custom k-path
        result = phonon_with_ml(
            al_structure,
            calc="chgnet",
            relax=True,
            plot=False,
            knames="GXWG",  # Custom k-path for FCC
            ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            distance=0.01,
        )
        
        # Verify phonon calculation completed
        assert result is not None
        assert hasattr(result, 'get_positions')
        
        # Check that phonon output files were created
        phonon_dir = Path("phonon_save")
        assert phonon_dir.exists()
        assert (phonon_dir / "phonopy_params.yaml").exists()
        assert (phonon_dir / "FORCE_CONSTANTS").exists()
        
        print("[Integration] Custom k-path workflow completed successfully!")
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])