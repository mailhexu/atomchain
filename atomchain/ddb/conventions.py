"""Convention conversions for the supported ABINIT DDB subset.

The source decisions are documented in memnotes note
``ddb_abinit_conventions.md``. ABINIT remains canonical; AbiPy is used as a
secondary parser reference.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

ANGSTROM_PER_BOHR = 0.529177210903
BOHR_PER_ANGSTROM = 1.0 / ANGSTROM_PER_BOHR
HARTREE_PER_EV = 1.0 / 27.211386245988
EV_PER_HARTREE = 1.0 / HARTREE_PER_EV
HARTREE_PER_BOHR3_TO_GPA = 29421.02648438959

DDB_VERSION = 20230401
DDB_BLOCK_D2_NONSTATIONARY = "2nd derivatives (non-stat.)"


def angstrom_to_bohr(value):
    """Convert Angstrom lengths to Bohr."""
    return np.asarray(value, dtype=float) * BOHR_PER_ANGSTROM


def bohr_to_angstrom(value):
    """Convert Bohr lengths to Angstrom."""
    return np.asarray(value, dtype=float) * ANGSTROM_PER_BOHR


def ev_to_hartree(value):
    """Convert eV energies to Hartree."""
    return np.asarray(value, dtype=float) * HARTREE_PER_EV


def hartree_to_ev(value):
    """Convert Hartree energies to eV."""
    return np.asarray(value, dtype=float) * EV_PER_HARTREE


def force_constant_ev_ang2_to_ha_bohr2(value):
    """Convert force constants from eV/Angstrom^2 to Hartree/Bohr^2."""
    return np.asarray(value) * HARTREE_PER_EV / (BOHR_PER_ANGSTROM**2)


def stress_ev_ang3_to_ha_bohr3(value):
    """Convert stress-like quantities from eV/Angstrom^3 to Hartree/Bohr^3."""
    return np.asarray(value, dtype=float) * HARTREE_PER_EV / (BOHR_PER_ANGSTROM**3)


def stress_ha_bohr3_to_gpa(value):
    """Convert Hartree/Bohr^3 to GPa."""
    return np.asarray(value, dtype=float) * HARTREE_PER_BOHR3_TO_GPA


def elastic_ev_ang3_to_ddb_strain_strain(value, volume_ang3):
    """Convert elastic constants to DDB strain-strain second derivatives.

    ABINIT elastic extraction divides the stored strain-strain derivative by
    cell volume to obtain pressure, so we multiply pressure-like values by the
    cell volume in Bohr^3 before writing DDB values.
    """
    volume_bohr3 = float(volume_ang3) * BOHR_PER_ANGSTROM**3
    return stress_ev_ang3_to_ha_bohr3(value) * volume_bohr3


def elastic_ddb_strain_strain_to_ev_ang3(value, volume_ang3):
    """Convert DDB strain-strain second derivatives to elastic constants."""
    volume_bohr3 = float(volume_ang3) * BOHR_PER_ANGSTROM**3
    return (
        np.asarray(value, dtype=float)
        / volume_bohr3
        * EV_PER_HARTREE
        * (BOHR_PER_ANGSTROM**3)
    )


def normalize_qpoint(qpoint, max_denominator=512):
    """Return a q-point as a tuple of Fractions for stable metadata."""
    return tuple(Fraction(float(x)).limit_denominator(max_denominator) for x in qpoint)


def qpoint_to_floats(qpoint):
    """Return q-point components as floats."""
    return tuple(float(x) for x in qpoint)


def voigt_to_tensor(voigt, strain=False):
    """Convert Voigt ``xx, yy, zz, yz, xz, xy`` to a 3x3 tensor.

    When ``strain`` is true, engineering shear components are halved in the
    off-diagonal tensor entries.
    """
    v = np.asarray(voigt, dtype=float)
    if v.shape != (6,):
        raise ValueError("Voigt vector must have shape (6,)")
    shear = 0.5 if strain else 1.0
    return np.array(
        [
            [v[0], shear * v[5], shear * v[4]],
            [shear * v[5], v[1], shear * v[3]],
            [shear * v[4], shear * v[3], v[2]],
        ]
    )


def tensor_to_voigt(tensor, strain=False):
    """Convert a symmetric 3x3 tensor to Voigt ``xx, yy, zz, yz, xz, xy``."""
    t = np.asarray(tensor, dtype=float)
    if t.shape != (3, 3):
        raise ValueError("Tensor must have shape (3, 3)")
    shear = 2.0 if strain else 1.0
    return np.array(
        [t[0, 0], t[1, 1], t[2, 2], shear * t[1, 2], shear * t[0, 2], shear * t[0, 1]]
    )


def strain_voigt_to_abinit_idir_ipert(voigt_index, natom):
    """Map Voigt strain index 0..5 to ABINIT one-based ``idir, ipert``."""
    if not 0 <= int(voigt_index) <= 5:
        raise ValueError("voigt_index must be in 0..5")
    if voigt_index < 3:
        return int(voigt_index) + 1, int(natom) + 3
    return int(voigt_index) - 2, int(natom) + 4


def displacement_to_abinit_idir_ipert(atom_index, cart_direction):
    """Map zero-based atom/cartesian displacement to ABINIT one-based indices."""
    if atom_index is None or cart_direction is None:
        raise ValueError(
            "displacement perturbation requires atom_index and cart_direction"
        )
    return int(cart_direction) + 1, int(atom_index) + 1


def abinit_perturbation_indices(perturbation, natom):
    """Return ABINIT ``idir, ipert`` for a DdbPerturbation-like object."""
    if perturbation.kind == "displacement":
        return displacement_to_abinit_idir_ipert(
            perturbation.atom_index, perturbation.cart_direction
        )
    if perturbation.kind in {"strain", "stress"}:
        return strain_voigt_to_abinit_idir_ipert(perturbation.voigt_index, natom)
    raise ValueError(f"Unsupported perturbation kind: {perturbation.kind}")
