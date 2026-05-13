import os

import numpy as np
import pytest
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def _make_phonon_with_spring_forces(a=4.05, supercell_matrix=None, k=10.0):
    if supercell_matrix is None:
        supercell_matrix = np.diag([2, 2, 2])
    cell = PhonopyAtoms(
        symbols=["Al"],
        scaled_positions=[[0, 0, 0]],
        cell=np.diag([a, a, a]),
    )
    phonon = Phonopy(cell, supercell_matrix)
    phonon.generate_displacements(distance=0.01)
    forces = np.zeros((len(phonon.displacements), len(phonon.supercell), 3))
    for i, disp in enumerate(phonon.displacements):
        atom_idx = int(disp[0]) - 1
        displacement = np.array(disp[1:])
        forces[i, atom_idx] = -k * displacement
    phonon.produce_force_constants(forces=forces)
    return phonon


@pytest.fixture
def phonon_spring():
    return _make_phonon_with_spring_forces()


@pytest.fixture
def phonopy_yaml(tmp_path):
    phonon = _make_phonon_with_spring_forces()
    save_path = os.path.join(str(tmp_path), "phonopy_params.yaml")
    phonon.save(filename=save_path, settings={"force_constants": True})
    return save_path


def test_phonopy_atoms_to_ase():
    from atomchain.modulate import phonopy_atoms_to_ase

    pa = PhonopyAtoms(
        symbols=["Al", "Al"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4.05,
    )
    atoms = phonopy_atoms_to_ase(pa)
    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert atoms.get_chemical_symbols() == ["Al", "Al"]
    assert atoms.pbc.all()
    np.testing.assert_allclose(atoms.cell.array, np.eye(3) * 4.05, atol=1e-10)


def test_phonopy_atoms_to_ase_positions():
    from atomchain.modulate import phonopy_atoms_to_ase

    pos = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]
    pa = PhonopyAtoms(
        symbols=["Cu", "Cu", "Cu"],
        scaled_positions=pos,
        cell=np.diag([3.6, 3.6, 3.6]),
    )
    atoms = phonopy_atoms_to_ase(pa)
    assert len(atoms) == 3
    np.testing.assert_allclose(atoms.get_scaled_positions(), pos, atol=1e-10)


def test_get_high_symmetry_modulations_from_object(phonon_spring):
    from atomchain.modulate import get_high_symmetry_modulations

    results = get_high_symmetry_modulations(
        phonon_spring,
        qpoint=[0, 0, 0],
        band_index=0,
        supercell_matrix=np.diag([2, 2, 2]),
        amplitude=0.1,
    )
    assert isinstance(results, list)
    for atoms in results:
        assert isinstance(atoms, Atoms)
        assert len(atoms) > 0
        assert atoms.cell.rank == 3


def test_get_high_symmetry_modulations_from_yaml(phonopy_yaml):
    from atomchain.modulate import get_high_symmetry_modulations

    results = get_high_symmetry_modulations(
        phonopy_yaml,
        qpoint=[0, 0, 0],
        band_index=0,
        supercell_matrix=np.diag([2, 2, 2]),
        amplitude=0.1,
    )
    assert isinstance(results, list)
    for atoms in results:
        assert isinstance(atoms, Atoms)
        assert len(atoms) > 0
        assert atoms.cell.rank == 3


def test_get_high_symmetry_modulations_amplitude(phonon_spring):
    from atomchain.modulate import get_high_symmetry_modulations

    results = get_high_symmetry_modulations(
        phonon_spring,
        qpoint=[0, 0, 0],
        band_index=0,
        supercell_matrix=np.diag([2, 2, 2]),
        amplitude=0.5,
    )
    assert isinstance(results, list)
    assert len(results) > 0
    for atoms in results:
        assert isinstance(atoms, Atoms)


def test_get_high_symmetry_modulations_returns_correct_count(phonon_spring):
    from atomchain.modulate import get_high_symmetry_modulations

    results = get_high_symmetry_modulations(
        phonon_spring,
        qpoint=[0, 0, 0],
        band_index=0,
        supercell_matrix=np.diag([2, 2, 2]),
        amplitude=0.1,
    )
    assert len(results) > 0
    for atoms in results:
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 8
