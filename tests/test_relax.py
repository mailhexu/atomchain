"""
Tests for structure relaxation.

Tests the relax_with_ml() function using real structures and ML calculators.

Purpose:
    Verify that structure relaxation works correctly with different structures
    and relaxation options (with/without symmetry, cell relaxation, etc.)

How to run:
    pytest tests/test_relax.py -v
    or
    pytest tests/test_relax.py::test_relax_al_basic -v  # run specific test

Note:
    These tests require ML potential packages (CHGNet, M3GNet, etc.) to be installed.
    Tests will be skipped if the required packages are not available.
"""

import os
from pathlib import Path

import pytest
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

from atomchain.relax import relax_with_ml


class InfoEnergyCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": atoms.info.get("energy", 0.0),
            "forces": [[0.0, 0.0, 0.0] for _ in atoms],
            "stress": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }


def test_relax_default_cell_factor_is_100(monkeypatch, al_structure, temp_dir):
    """Cell relaxation should use a balanced stress-force factor by default."""
    os.chdir(temp_dir)
    captured = {}

    class MockUnitCellFilter:
        def __init__(self, atoms, cell_factor, **kwargs):
            captured["cell_factor"] = cell_factor
            self.atoms = atoms

    class MockFIRE:
        def __init__(self, atoms):
            self.atoms = atoms

        def run(self, fmax, steps=0):
            return None

    monkeypatch.setattr("atomchain.relax.UnitCellFilter", MockUnitCellFilter)
    monkeypatch.setattr("atomchain.relax.FIRE", MockFIRE)
    monkeypatch.setattr("atomchain.relax.BFGS", MockFIRE)

    relax_with_ml(al_structure, calc=InfoEnergyCalculator(), relax_cell=True, sym=False)

    assert captured["cell_factor"] == 100


def test_relax_cell_uses_bfgs_then_fire_fallback_when_energy_increases(
    monkeypatch, al_structure, temp_dir
):
    """Second-stage BFGS should fall back to FIRE from the first-stage result."""
    os.chdir(temp_dir)
    calls = []

    class MockUnitCellFilter:
        def __init__(self, atoms, cell_factor, **kwargs):
            self.atoms = atoms

    class MockFIRE:
        def __init__(self, subject):
            self.subject = subject

        def run(self, fmax, steps=0):
            atoms = self.subject.atoms
            calls.append(("FIRE", fmax, steps, atoms.positions[0, 0]))
            atoms.positions[0, 0] = 1.0
            atoms.info["energy"] = -1.0

    class MockBFGS:
        def __init__(self, subject):
            self.subject = subject

        def run(self, fmax, steps=0):
            atoms = self.subject.atoms
            calls.append(("BFGS", fmax, steps, atoms.positions[0, 0]))
            atoms.positions[0, 0] = 2.0
            atoms.info["energy"] = 0.0

    monkeypatch.setattr("atomchain.relax.UnitCellFilter", MockUnitCellFilter)
    monkeypatch.setattr("atomchain.relax.FIRE", MockFIRE)
    monkeypatch.setattr("atomchain.relax.BFGS", MockBFGS)

    relaxed = relax_with_ml(
        al_structure,
        calc=InfoEnergyCalculator(),
        relax_cell=True,
        sym=False,
        fmax=0.01,
    )

    assert calls == [
        ("FIRE", 0.1, 3500, al_structure.positions[0, 0]),
        ("BFGS", 0.01, 200, 1.0),
        ("FIRE", 0.01, 5000, 1.0),
    ]
    assert relaxed.positions[0, 0] == 1.0
    assert relaxed.info["energy"] == -1.0


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


def test_relax_al_basic(al_structure, temp_dir):
    """
    Test basic relaxation of Al structure without cell relaxation.

    Uses default CHGNet calculator.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    relaxed = relax_with_ml(
        al_structure,
        calc="chgnet",
        relax_cell=False,
        sym=False,
        fmax=0.05,
        traj_file="al_relax.traj",
    )

    assert relaxed is not None
    assert len(relaxed) == len(al_structure)
    assert relaxed.get_chemical_formula() == "Al"
    # Check that trajectory file was created
    assert (temp_dir / "al_relax.traj").exists() or Path("al_relax.traj").exists()


def test_relax_al_with_cell(al_structure, temp_dir):
    """
    Test relaxation of Al structure with cell optimization.

    Uses default CHGNet calculator and allows both positions and cell to relax.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    original_volume = al_structure.get_volume()

    relaxed = relax_with_ml(
        al_structure, calc="chgnet", relax_cell=True, sym=False, fmax=0.05
    )

    assert relaxed is not None
    assert len(relaxed) == len(al_structure)
    # Cell should be optimized (volume may change slightly)
    relaxed_volume = relaxed.get_volume()
    assert relaxed_volume > 0
    print(f"Volume change: {original_volume:.2f} → {relaxed_volume:.2f} Å³")


def test_relax_srtio3_with_symmetry(srtio3_structure, temp_dir):
    """
    Test relaxation of SrTiO3 with symmetry constraints.

    Cubic perovskite should maintain symmetry during relaxation.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    relaxed = relax_with_ml(
        srtio3_structure, calc="chgnet", relax_cell=False, sym=True, fmax=0.05
    )

    assert relaxed is not None
    assert len(relaxed) == 5
    assert relaxed.get_chemical_formula() == "O3SrTi"


def test_relax_srtio3_cell(srtio3_structure, temp_dir):
    """
    Test cell relaxation of SrTiO3 cubic perovskite.

    Tests full cell optimization without symmetry constraints.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    original_a = srtio3_structure.cell[0, 0]

    relaxed = relax_with_ml(
        srtio3_structure,
        calc="chgnet",
        relax_cell=True,
        sym=False,
        fmax=0.05,
        cell_factor=1000,
    )

    assert relaxed is not None
    assert len(relaxed) == 5
    relaxed_a = relaxed.cell[0, 0]
    print(f"Lattice parameter: {original_a:.3f} → {relaxed_a:.3f} Å")


def test_relax_pbtio3_basic(pbtio3_structure, temp_dir):
    """
    Test relaxation of PbTiO3 with Ti displacement.

    This structure has broken symmetry due to Ti displacement,
    testing relaxation of a distorted structure.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    # Get initial Ti position (should be displaced from center)
    ti_idx = 1  # Ti is the second atom
    original_ti_z = pbtio3_structure.positions[ti_idx, 2]

    relaxed = relax_with_ml(
        pbtio3_structure, calc="chgnet", relax_cell=False, sym=False, fmax=0.05
    )

    assert relaxed is not None
    assert len(relaxed) == 5
    assert relaxed.get_chemical_formula() == "O3PbTi"

    # Ti may move during relaxation
    relaxed_ti_z = relaxed.positions[ti_idx, 2]
    print(f"Ti z-position: {original_ti_z:.3f} → {relaxed_ti_z:.3f} Å")


def test_relax_pbtio3_cell(pbtio3_structure, temp_dir):
    """
    Test cell relaxation of tetragonal PbTiO3.

    Tests relaxation of both cell and atomic positions for
    a tetragonal structure (a ≠ c).
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    original_a = pbtio3_structure.cell[0, 0]
    original_c = pbtio3_structure.cell[2, 2]
    original_c_over_a = original_c / original_a

    relaxed = relax_with_ml(
        pbtio3_structure, calc="chgnet", relax_cell=True, sym=False, fmax=0.05
    )

    assert relaxed is not None
    assert len(relaxed) == 5

    relaxed_a = relaxed.cell[0, 0]
    relaxed_c = relaxed.cell[2, 2]
    relaxed_c_over_a = relaxed_c / relaxed_a

    print(f"Lattice a: {original_a:.3f} → {relaxed_a:.3f} Å")
    print(f"Lattice c: {original_c:.3f} → {relaxed_c:.3f} Å")
    print(f"c/a ratio: {original_c_over_a:.3f} → {relaxed_c_over_a:.3f}")


def test_relax_with_string_calculator(al_structure, temp_dir):
    """
    Test that calculator can be specified as a string.

    Verifies that calc="chgnet" works (not just passing calculator object).
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    relaxed = relax_with_ml(
        al_structure, calc="chgnet", relax_cell=False, sym=False, fmax=0.05
    )

    assert relaxed is not None
    assert len(relaxed) == 1


def test_relax_with_default_calculator(al_structure, temp_dir):
    """
    Test relaxation with default calculator (calc=None should use CHGNet).

    Verifies the default behavior when no calculator is specified.
    """
    pytest.importorskip("chgnet")
    os.chdir(temp_dir)

    relaxed = relax_with_ml(
        al_structure,
        calc=None,  # Should default to CHGNet
        relax_cell=False,
        sym=False,
        fmax=0.05,
    )

    assert relaxed is not None
    assert len(relaxed) == 1
