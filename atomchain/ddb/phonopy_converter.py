"""Convert phonopy data into the internal DDB model."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from phonopy import load

from .conventions import (
    ev_to_hartree,
    force_constant_ev_ang2_to_ha_bohr2,
    normalize_qpoint,
)
from .model import DdbDerivativeBlock, DdbDocument, DdbHeader, DdbPerturbation
from .writer import write_ddb


def ddb_document_from_phonopy(
    atoms=None,
    phonopy_yaml=None,
    phonon=None,
    qpoints=None,
    qgrid=None,
    symprec=1e-5,
    metadata=None,
    reference_energy_ev=None,
):
    """Build a phonon-only DDB document from a Phonopy object or YAML file.

    The displacement-displacement values are built from phonopy's dynamical
    matrix at each requested q-point, then un-mass-weighted back to Cartesian
    force constants and converted to ABINIT DDB reduced-coordinate derivatives.
    """
    if phonon is None:
        if phonopy_yaml is None:
            raise ValueError("phonopy_yaml or phonon must be provided")
        phonon = load(str(phonopy_yaml))
    if atoms is None:
        atoms = _atoms_from_phonopy_cell(phonon.unitcell)
    header = DdbHeader.from_atoms(atoms, symprec=symprec)
    document = DdbDocument(
        header=header,
        reference_energy_ha=_reference_energy_ha(atoms, reference_energy_ev),
        metadata={
            "source": "phonopy",
            "supported_subset": "phonon displacement-displacement blocks",
            "phonopy_yaml": str(phonopy_yaml) if phonopy_yaml is not None else None,
            **(metadata or {}),
        },
    )
    if qpoints is None:
        qpoints = qpoints_from_grid(qgrid or (1, 1, 1))
    qpoints = [normalize_qpoint(q) for q in qpoints]
    natom = len(atoms)
    for qpoint in qpoints:
        fc_ha_bohr2 = force_constant_ev_ang2_to_ha_bohr2(
            _force_constants_at_qpoint(phonon, qpoint, natom)
        )
        fc_ddb = _cartesian_force_constants_to_ddb_reduced(
            fc_ha_bohr2, header.lattice_bohr
        )
        for atom_i in range(natom):
            for atom_j in range(natom):
                for dir_i in range(3):
                    for dir_j in range(3):
                        value = complex(fc_ddb[atom_i, atom_j, dir_i, dir_j])
                        document.add_derivative(
                            DdbDerivativeBlock(
                                qpoint=qpoint,
                                perturbation_i=DdbPerturbation(
                                    kind="displacement",
                                    atom_index=atom_i,
                                    cart_direction=dir_i,
                                    qpoint=qpoint,
                                ),
                                perturbation_j=DdbPerturbation(
                                    kind="displacement",
                                    atom_index=atom_j,
                                    cart_direction=dir_j,
                                    qpoint=qpoint,
                                ),
                                value=value,
                                units="Ha/reduced^2",
                                source="phonopy",
                            )
                        )
    return document


def write_ddb_from_phonopy(
    atoms=None,
    phonopy_yaml=None,
    phonon=None,
    filename="out.ddb",
    qpoints=None,
    qgrid=None,
    symprec=1e-5,
    metadata_filename=None,
    reference_energy_ev=None,
):
    """Write a phonon-only DDB text file and YAML metadata sidecar."""
    document = ddb_document_from_phonopy(
        atoms=atoms,
        phonopy_yaml=phonopy_yaml,
        phonon=phonon,
        qpoints=qpoints,
        qgrid=qgrid,
        symprec=symprec,
        reference_energy_ev=reference_energy_ev,
    )
    return write_ddb(document, filename, metadata_filename=metadata_filename)


def qpoints_from_grid(qgrid):
    """Return a full Gamma-centered reduced q-point grid.

    A grid of ``(2, 2, 2)`` gives the eight points with components ``0`` and
    ``0.5``. These are valid DDB reduced reciprocal coordinates and match the
    natural supercell-compatible grid used by phonopy dynamical matrices.
    """
    grid = np.asarray(qgrid, dtype=int)
    if grid.shape != (3,) or np.any(grid <= 0):
        raise ValueError("qgrid must be three positive integers")
    return [
        (ix / grid[0], iy / grid[1], iz / grid[2])
        for ix in range(grid[0])
        for iy in range(grid[1])
        for iz in range(grid[2])
    ]


def _force_constants_at_qpoint(phonon, qpoint, natom):
    """Return primitive-cell force constants at q in eV/Angstrom^2."""
    phonon.dynamical_matrix.run([float(x) for x in qpoint])
    dynmat = np.asarray(phonon.dynamical_matrix.dynamical_matrix, dtype=complex)
    if dynmat.shape != (3 * natom, 3 * natom):
        raise ValueError(
            f"Cannot map dynamical matrix with shape {dynmat.shape} to natom={natom}"
        )

    masses = np.asarray(phonon.primitive.masses, dtype=float)
    if len(masses) != natom:
        raise ValueError(
            f"Cannot map primitive masses with length {len(masses)} to natom={natom}"
        )
    mass_weights = np.sqrt(np.outer(np.repeat(masses, 3), np.repeat(masses, 3)))
    return (dynmat * mass_weights).reshape(natom, 3, natom, 3).transpose(0, 2, 1, 3)


def _cartesian_force_constants_to_ddb_reduced(force_constants, lattice_bohr):
    """Convert Cartesian Ha/Bohr^2 IFC blocks to ABINIT DDB reduced derivatives."""
    fc = np.asarray(force_constants, dtype=complex)
    lattice = np.asarray(lattice_bohr, dtype=float)
    if fc.ndim != 4 or fc.shape[2:] != (3, 3):
        raise ValueError("force_constants must have shape (natom, natom, 3, 3)")
    if lattice.shape != (3, 3):
        raise ValueError("lattice_bohr must have shape (3, 3)")
    out = np.zeros_like(fc, dtype=complex)
    for atom_i in range(fc.shape[0]):
        for atom_j in range(fc.shape[1]):
            # pymultibinit decodes DDB blocks as inv(A).T @ Phi_red @ inv(A),
            # so write the inverse transform here. A stores lattice vectors in rows.
            out[atom_i, atom_j, :, :] = lattice.T @ fc[atom_i, atom_j, :, :] @ lattice
    return out


def _atoms_from_phonopy_cell(cell):
    return Atoms(
        symbols=list(cell.symbols),
        scaled_positions=np.asarray(cell.scaled_positions, dtype=float),
        cell=np.asarray(cell.cell, dtype=float),
        pbc=True,
    )


def _reference_energy_ha(atoms, reference_energy_ev):
    if reference_energy_ev is not None:
        return float(ev_to_hartree(reference_energy_ev))
    try:
        return float(ev_to_hartree(atoms.get_potential_energy()))
    except Exception:
        return None
