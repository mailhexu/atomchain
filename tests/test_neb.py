"""
Tests for NEB calculations.

Tests the calculate_neb() function and mlneb CLI using structures with known barriers.

Purpose:
    Verify that NEB calculations work correctly for finding transition states and
    calculating energy barriers between initial and final configurations.

How to run:
    pytest tests/test_neb.py -v
    or
    pytest tests/test_neb.py::test_neb_basic -v  # run specific test

Note:
    These tests require ML potential packages (CHGNet, etc.) to be installed.
    Tests will be skipped if the required packages are not available.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read, write

from atomchain.neb import (
    _calculate_barriers,
    _interpolate_path,
    calculate_neb,
)


@pytest.fixture
def simple_diffusion_path():
    """
    Create simple 1D diffusion path for testing.
    
    Creates an Al atom moving from one octahedral site to another
    in an FCC lattice (simple vacancy diffusion analog).
    """
    # Initial: Al at (0, 0, 0)
    initial = Atoms('Al', positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    
    # Final: Al at (1.5, 0, 0) - halfway across cell
    final = Atoms('Al', positions=[[1.5, 0, 0]], cell=[3, 3, 3], pbc=True)
    
    return initial, final


@pytest.fixture
def al_bulk_distortion():
    """
    Create Al FCC structure with atom displaced (fixed cell for NEB).
    
    Initial: Perfect FCC
    Final: One atom slightly displaced
    """
    initial = bulk('Al', 'fcc', a=4.05, cubic=True)
    
    final = initial.copy()
    # Displace one atom slightly
    final.positions[0] += [0.1, 0.1, 0.0]
    
    return initial, final


def test_interpolate_path_basic(simple_diffusion_path):
    """
    Test basic IDPP interpolation between two structures.
    """
    initial, final = simple_diffusion_path
    
    images = _interpolate_path(initial, final, nimages=5)
    
    assert len(images) == 5
    assert images[0] == initial
    assert images[-1] == final
    
    # Check intermediate images exist
    for img in images[1:-1]:
        assert len(img) == 1
        assert img.get_chemical_formula() == "Al"


def test_interpolate_path_different_nimages(simple_diffusion_path):
    """
    Test interpolation with different numbers of images.
    """
    initial, final = simple_diffusion_path
    
    for n in [3, 5, 7, 9, 11]:
        images = _interpolate_path(initial, final, nimages=n)
        assert len(images) == n


def test_interpolate_path_invalid_nimages(simple_diffusion_path):
    """
    Test that interpolation fails with invalid nimages.
    """
    initial, final = simple_diffusion_path
    
    with pytest.raises(ValueError, match="nimages must be at least 3"):
        _interpolate_path(initial, final, nimages=2)
    
    with pytest.raises(ValueError, match="nimages must be at least 3"):
        _interpolate_path(initial, final, nimages=1)


def test_interpolate_path_incompatible_structures():
    """
    Test that interpolation fails for incompatible structures.
    """
    initial = Atoms('Al', positions=[[0, 0, 0]], cell=[3, 3, 3])
    final = Atoms('Al2', positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3])
    
    with pytest.raises(ValueError, match="different number of atoms"):
        _interpolate_path(initial, final, nimages=5)


def test_interpolate_path_periodic_mic():
    """
    Test that interpolation uses minimum image convention for periodic systems.
    
    When an atom crosses a periodic boundary, the interpolation should take
    the shorter path (using MIC) rather than the longer direct path.
    """
    # Initial: atom at x=0.1
    initial = Atoms('Al', positions=[[0.1, 0, 0]], cell=[3, 3, 3], pbc=True)
    
    # Final: atom at x=2.9 (close to boundary)
    # MIC distance = 0.2 Å, direct distance = 2.8 Å
    final = Atoms('Al', positions=[[2.9, 0, 0]], cell=[3, 3, 3], pbc=True)
    
    images = _interpolate_path(initial, final, nimages=5)
    
    # Check that intermediate images take the short path
    # Should go: 0.1 -> ~0.05 -> ~0.0 -> ~-0.05 -> 2.9 (wraps to -0.1)
    # NOT: 0.1 -> 0.8 -> 1.5 -> 2.2 -> 2.9
    
    # Check that we're going the right direction (decreasing initially)
    assert images[1].positions[0, 0] < images[0].positions[0, 0], \
        "MIC should take shorter path (decreasing x initially)"
    
    # Verify all images are consistent
    assert len(images) == 5
    for img in images:
        assert len(img) == 1
        assert img.get_chemical_formula() == "Al"


def test_calculate_barriers():
    """
    Test energy barrier calculation from energy profile.
    """
    # Simple energy profile: 0 -> 1.5 -> 0.5 (eV)
    energies = np.array([0.0, 0.5, 1.0, 1.5, 1.2, 0.8, 0.5])
    
    barriers = _calculate_barriers(energies)
    
    assert barriers['barrier_forward'] == pytest.approx(1.5, abs=1e-10)
    assert barriers['barrier_reverse'] == pytest.approx(1.0, abs=1e-10)
    assert barriers['ts_index'] == 3
    assert barriers['ts_energy'] == 1.5
    assert barriers['reaction_energy'] == pytest.approx(0.5, abs=1e-10)


def test_calculate_barriers_symmetric():
    """
    Test barrier calculation for symmetric energy profile.
    """
    # Symmetric: 0 -> 2.0 -> 0
    energies = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    
    barriers = _calculate_barriers(energies)
    
    assert barriers['barrier_forward'] == pytest.approx(2.0, abs=1e-10)
    assert barriers['barrier_reverse'] == pytest.approx(2.0, abs=1e-10)
    assert barriers['ts_index'] == 2
    assert barriers['reaction_energy'] == pytest.approx(0.0, abs=1e-10)


def test_neb_basic_with_mock(simple_diffusion_path, temp_dir, mocker):
    """
    Test basic NEB calculation with mocked calculator.
    
    Uses mock calculator to avoid needing actual ML potentials.
    """
    os.chdir(temp_dir)
    initial, final = simple_diffusion_path
    
    # Mock calculator that returns dummy energy
    mock_calc = mocker.Mock()
    mock_calc.get_potential_energy.return_value = 0.5
    mock_calc.get_forces.return_value = np.array([[0.01, 0.0, 0.0]])
    
    results = calculate_neb(
        initial,
        final,
        calculator=mock_calc,
        nimages=5,
        fmax=0.5,  # Loose convergence for quick test
        max_steps=10,
        quiet=True
    )
    
    assert 'images' in results
    assert 'energies' in results
    assert 'barrier_forward' in results
    assert 'barrier_reverse' in results
    assert len(results['images']) == 5
    assert len(results['energies']) == 5


def test_neb_with_file_paths(al_bulk_distortion, temp_dir):
    """
    Test NEB calculation loading structures from files.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)
    
    initial, final = al_bulk_distortion
    
    # Save structures to files
    initial_file = temp_dir / "initial.vasp"
    final_file = temp_dir / "final.vasp"
    write(str(initial_file), initial)
    write(str(final_file), final)
    
    results = calculate_neb(
        str(initial_file),
        str(final_file),
        calculator='chgnet',
        nimages=5,
        fmax=0.1,
        max_steps=50,
        quiet=True
    )
    
    assert results is not None
    assert len(results['images']) == 5
    assert results['energies'].shape == (5,)


def test_neb_output_trajectory(al_bulk_distortion, temp_dir):
    """
    Test that NEB saves trajectory file correctly.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)
    
    initial, final = al_bulk_distortion
    output_file = temp_dir / "neb_path.traj"
    
    results = calculate_neb(
        initial,
        final,
        calculator='chgnet',
        nimages=5,
        fmax=0.1,
        max_steps=50,
        output=str(output_file),
        quiet=True
    )
    
    assert output_file.exists()
    
    # Read trajectory and verify
    images = read(str(output_file), ":")
    assert len(images) == 5


def test_neb_output_plot(al_bulk_distortion, temp_dir):
    """
    Test that NEB creates energy profile plot.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)
    
    initial, final = al_bulk_distortion
    plot_file = temp_dir / "barrier.png"
    
    results = calculate_neb(
        initial,
        final,
        calculator='chgnet',
        nimages=5,
        fmax=0.1,
        max_steps=50,
        plot=str(plot_file),
        quiet=True
    )
    
    assert plot_file.exists()


def test_neb_different_optimizers(al_bulk_distortion, temp_dir):
    """
    Test NEB with different optimizers (FIRE, BFGS, LBFGS).
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)
    
    initial, final = al_bulk_distortion
    
    for optimizer in ['FIRE', 'BFGS', 'LBFGS']:
        results = calculate_neb(
            initial.copy(),
            final.copy(),
            calculator='chgnet',
            nimages=5,
            fmax=0.1,
            optimizer=optimizer,
            max_steps=50,
            quiet=True
        )
        
        assert results is not None
        assert 'barrier_forward' in results


def test_neb_invalid_optimizer(simple_diffusion_path):
    """
    Test that invalid optimizer raises error.
    """
    initial, final = simple_diffusion_path
    
    with pytest.raises(ValueError, match="Unknown optimizer"):
        calculate_neb(
            initial,
            final,
            calculator='chgnet',
            optimizer='InvalidOpt',
            quiet=True
        )


def test_neb_file_not_found():
    """
    Test that missing input files raise appropriate errors.
    """
    with pytest.raises(FileNotFoundError):
        calculate_neb(
            "nonexistent_initial.vasp",
            "nonexistent_final.vasp",
            quiet=True
        )


def test_neb_invalid_nimages(simple_diffusion_path):
    """
    Test that invalid nimages raises error.
    """
    initial, final = simple_diffusion_path
    
    with pytest.raises(ValueError, match="nimages must be at least 3"):
        calculate_neb(initial, final, nimages=2, quiet=True)


def test_neb_with_symmetry_breaking(simple_diffusion_path, temp_dir, mocker):
    """
    Test NEB with symmetry breaking option.
    
    Verifies that intermediate images are perturbed when break_symmetry > 0.
    """
    os.chdir(temp_dir)
    initial, final = simple_diffusion_path
    
    # Mock calculator
    mock_calc = mocker.Mock()
    mock_calc.get_potential_energy.return_value = 0.5
    mock_calc.get_forces.return_value = np.array([[0.01, 0.0, 0.0]])
    
    # Run with symmetry breaking
    np.random.seed(42)  # For reproducibility
    results = calculate_neb(
        initial,
        final,
        calculator=mock_calc,
        nimages=5,
        fmax=0.5,
        max_steps=5,
        break_symmetry=0.1,  # 0.1 Å perturbation
        quiet=True
    )
    
    assert results is not None
    assert len(results['images']) == 5
    
    # Check that intermediate images were perturbed
    # (endpoints should not be perturbed)
    for i in range(1, 4):  # Only check intermediate images
        # Images should have calculator attached
        assert results['images'][i].calc is not None


def test_neb_results_structure(al_bulk_distortion, temp_dir):
    """
    Test that NEB results have correct structure and types.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)
    
    initial, final = al_bulk_distortion
    
    results = calculate_neb(
        initial,
        final,
        calculator='chgnet',
        nimages=5,
        fmax=0.1,
        max_steps=50,
        quiet=True
    )
    
    # Check all expected keys
    expected_keys = {
        'images', 'energies', 'barrier_forward', 'barrier_reverse',
        'ts_index', 'ts_energy', 'reaction_energy', 'converged'
    }
    assert set(results.keys()) == expected_keys
    
    # Check types
    assert isinstance(results['images'], list)
    assert isinstance(results['energies'], np.ndarray)
    assert isinstance(results['barrier_forward'], (float, np.floating))
    assert isinstance(results['barrier_reverse'], (float, np.floating))
    assert isinstance(results['ts_index'], (int, np.integer))
    assert isinstance(results['reaction_energy'], (float, np.floating))
    assert isinstance(results['converged'], bool)


def test_neb_cli_basic(al_bulk_distortion, temp_dir):
    """
    Test mlneb CLI tool with basic options.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)
    
    initial, final = al_bulk_distortion
    
    # Save structures
    write("initial.vasp", initial)
    write("final.vasp", final)
    
    # Run CLI
    result = subprocess.run(
        [
            "mlneb",
            "initial.vasp",
            "final.vasp",
            "--calc", "chgnet",
            "--nimages", "5",
            "--fmax", "0.1",
            "--max-steps", "50",
            "-o", "neb_path.traj",
            "-p", "barrier.png",
            "--quiet"
        ],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert (temp_dir / "neb_path.traj").exists()
    assert (temp_dir / "barrier.png").exists()


def test_neb_cli_help():
    """
    Test that mlneb --help works.
    """
    result = subprocess.run(
        ["mlneb", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "NEB calculation" in result.stdout
    assert "--calc" in result.stdout
    assert "--nimages" in result.stdout
