import numpy as np
from ase.build import bulk
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from atomchain.ddb.conventions import (
    BOHR_PER_ANGSTROM,
    EV_PER_HARTREE,
    force_constant_ev_ang2_to_ha_bohr2,
)
from atomchain.ddb.phonopy_converter import (
    _cartesian_force_constants_to_ddb_reduced,
    _force_constants_at_qpoint,
    ddb_document_from_phonopy,
    qpoints_from_grid,
    write_ddb_from_phonopy,
)

THZ_TO_CM = 33.35640951981521


def synthetic_phonon():
    atoms = bulk("Al", "fcc", a=4.05)
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell.array,
    )
    phonon = Phonopy(unitcell, np.eye(3, dtype=int))
    fc = np.zeros((1, 1, 3, 3), dtype=float)
    fc[0, 0] = np.diag([1.0, 2.0, 3.0])
    phonon.force_constants = fc
    return atoms, phonon


def test_convert_phonopy_object_to_displacement_blocks():
    atoms, phonon = synthetic_phonon()
    document = ddb_document_from_phonopy(
        atoms=atoms, phonon=phonon, qpoints=[(0, 0, 0), (0.5, 0, 0)]
    )
    assert len(document.qpoint_blocks) == 2
    assert len(document.derivative_blocks) == 18
    assert all(
        block.perturbation_i.kind == "displacement"
        for block in document.derivative_blocks
    )


def test_qpoints_from_grid_returns_full_grid():
    qpoints = qpoints_from_grid((2, 2, 2))
    assert len(qpoints) == 8
    assert (0.0, 0.0, 0.0) in qpoints
    assert (0.5, 0.5, 0.5) in qpoints


def test_convert_phonopy_object_uses_qgrid():
    atoms, phonon = synthetic_phonon()
    document = ddb_document_from_phonopy(atoms=atoms, phonon=phonon, qgrid=(2, 2, 2))
    assert len(document.qpoint_blocks) == 8
    assert len(document.derivative_blocks) == 8 * 9


def test_write_ddb_from_phonopy_object(tmp_path):
    atoms, phonon = synthetic_phonon()
    output = tmp_path / "phonon.ddb"
    write_ddb_from_phonopy(atoms=atoms, phonon=phonon, filename=output)
    assert output.exists()
    assert (tmp_path / "phonon.ddb.yaml").exists()


def test_phonopy_band_structure_sanity_from_source_force_constants():
    _, phonon = synthetic_phonon()
    path = [np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])]

    phonon.run_band_structure(path)
    bands = phonon.get_band_structure_dict()

    assert bands["frequencies"][0].shape == (2, 3)
    assert np.all(np.isfinite(bands["frequencies"][0]))
    np.testing.assert_allclose(bands["frequencies"][0][0], bands["frequencies"][0][1])


def test_ddb_qpoint_blocks_reproduce_phonopy_frequencies():
    atoms, phonon = synthetic_phonon()
    qpoints = [(0, 0, 0), (0.5, 0, 0)]
    document = ddb_document_from_phonopy(atoms=atoms, phonon=phonon, qpoints=qpoints)

    phonon.run_band_structure([np.array(qpoints, dtype=float)])
    phonopy_freq_cm = phonon.get_band_structure_dict()["frequencies"][0] * THZ_TO_CM

    ddb_freq_cm = np.array(
        [_document_frequencies_cm(document, phonon, qpoint) for qpoint in qpoints]
    )
    np.testing.assert_allclose(ddb_freq_cm, phonopy_freq_cm, rtol=1e-10, atol=1e-10)


def test_displacement_blocks_are_written_as_reduced_derivatives():
    atoms, phonon = synthetic_phonon()
    document = ddb_document_from_phonopy(
        atoms=atoms, phonon=phonon, qpoints=[(0, 0, 0)]
    )

    lattice_bohr = np.asarray(atoms.cell.array) * BOHR_PER_ANGSTROM
    fc_ha_bohr2 = force_constant_ev_ang2_to_ha_bohr2(
        _force_constants_at_qpoint(phonon, (0, 0, 0), len(atoms))
    )
    expected = _cartesian_force_constants_to_ddb_reduced(fc_ha_bohr2, lattice_bohr)

    block = next(
        block
        for block in document.derivative_blocks
        if block.perturbation_i.cart_direction == 0
        and block.perturbation_j.cart_direction == 0
    )
    assert block.units == "Ha/reduced^2"
    np.testing.assert_allclose(block.value, expected[0, 0, 0, 0])


def _document_frequencies_cm(document, phonon, qpoint):
    natom = len(document.header.atomic_numbers)
    fc_red = np.zeros((natom, natom, 3, 3), dtype=complex)
    q = tuple(float(x) for x in qpoint)
    for block in document.derivative_blocks:
        if tuple(float(x) for x in block.qpoint) != q:
            continue
        pi = block.perturbation_i
        pj = block.perturbation_j
        fc_red[pi.atom_index, pj.atom_index, pi.cart_direction, pj.cart_direction] = (
            block.value
        )

    lattice_bohr = np.asarray(document.header.lattice_bohr, dtype=float)
    mat_mp = np.linalg.inv(lattice_bohr).T
    fc_ha_bohr2 = np.zeros_like(fc_red)
    for atom_i in range(natom):
        for atom_j in range(natom):
            fc_ha_bohr2[atom_i, atom_j, :, :] = (
                mat_mp @ fc_red[atom_i, atom_j, :, :] @ mat_mp.T
            )
    fc_ev_ang2 = fc_ha_bohr2 * EV_PER_HARTREE * BOHR_PER_ANGSTROM**2
    masses = np.asarray(phonon.primitive.masses, dtype=float)
    mass_weights = np.sqrt(
        np.outer(np.repeat(masses, 3), np.repeat(masses, 3))
    ).reshape(natom, 3, natom, 3)
    fc_atom_dir = fc_ev_ang2.transpose(0, 2, 1, 3)
    dynmat = (fc_atom_dir / mass_weights).reshape(3 * natom, 3 * natom)
    eigenvalues = np.linalg.eigvalsh(dynmat)
    frequencies_thz = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * phonon.unit_conversion_factor
    )
    return np.sort(frequencies_thz) * THZ_TO_CM
