"""
Pytest fixtures for atomchain tests.

This module provides reusable test fixtures for loading test structures.

Structure Files:
    Place test structure files in tests/fixtures/ directory:
    - test_structure.vasp (or any ASE-readable format)
    - Small structures are preferred for fast tests

Usage:
    These fixtures are automatically discovered by pytest. Use them in tests by
    adding them as function parameters:
    
    def test_something(test_structure):
        # Use the fixture
        atoms = test_structure
        # Test with real ML calculators

Run tests with: pytest atomchain/tests/
"""

import os
from pathlib import Path

import pytest
from ase.io import read


@pytest.fixture
def fixtures_dir():
    """
    Provide path to fixtures directory.
    
    Returns the path to tests/fixtures/ where structure files are stored.
    """
    test_dir = Path(__file__).parent
    return test_dir / "fixtures"


@pytest.fixture
def test_structure(fixtures_dir):
    """
    Load a test structure from fixtures directory.
    
    Looks for structure files in tests/fixtures/:
    - Tries multiple formats: .vasp, .cif, .xyz, POSCAR
    - Returns the first structure file found
    - Skips test if no structure files are found
    
    To use: place a small structure file in tests/fixtures/
    """
    # Try common structure file patterns
    patterns = [
        "*.vasp",
        "*.cif", 
        "*.xyz",
        "POSCAR*",
        "test_structure.*",
    ]
    
    for pattern in patterns:
        files = list(fixtures_dir.glob(pattern))
        if files:
            structure_file = files[0]
            return read(str(structure_file))
    
    pytest.skip("No test structure files found in tests/fixtures/")


@pytest.fixture
def temp_dir(tmp_path):
    """
    Provide a temporary directory for test outputs.
    
    Automatically cleaned up after test completes.
    """
    return tmp_path
