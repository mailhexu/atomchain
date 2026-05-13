from fractions import Fraction

import numpy as np

from atomchain.ddb.conventions import (
    ANGSTROM_PER_BOHR,
    HARTREE_PER_EV,
    abinit_perturbation_indices,
    angstrom_to_bohr,
    force_constant_ev_ang2_to_ha_bohr2,
    strain_voigt_to_abinit_idir_ipert,
    tensor_to_voigt,
    voigt_to_tensor,
)
from atomchain.ddb.model import DdbPerturbation


def test_unit_conversions_are_named_and_exact():
    assert np.isclose(angstrom_to_bohr(ANGSTROM_PER_BOHR), 1.0)
    assert np.isclose(
        force_constant_ev_ang2_to_ha_bohr2(1.0), HARTREE_PER_EV * ANGSTROM_PER_BOHR**2
    )


def test_displacement_perturbation_indices_are_one_based():
    perturbation = DdbPerturbation(kind="displacement", atom_index=2, cart_direction=1)
    assert abinit_perturbation_indices(perturbation, natom=5) == (2, 3)


def test_strain_perturbation_indices_match_abinit_layout():
    assert strain_voigt_to_abinit_idir_ipert(0, natom=5) == (1, 8)
    assert strain_voigt_to_abinit_idir_ipert(2, natom=5) == (3, 8)
    assert strain_voigt_to_abinit_idir_ipert(3, natom=5) == (1, 9)
    assert strain_voigt_to_abinit_idir_ipert(5, natom=5) == (3, 9)


def test_voigt_strain_shear_round_trip():
    voigt = np.array([1.0, 2.0, 3.0, 0.4, 0.6, 0.8])
    tensor = voigt_to_tensor(voigt, strain=True)
    assert np.isclose(tensor[1, 2], 0.2)
    assert np.allclose(tensor_to_voigt(tensor, strain=True), voigt)


def test_qpoint_fraction_metadata_validation():
    perturbation = DdbPerturbation(
        kind="displacement",
        atom_index=0,
        cart_direction=0,
        qpoint=(Fraction(1, 2), Fraction(0), Fraction(0)),
    )
    assert perturbation.qpoint[0] == Fraction(1, 2)
