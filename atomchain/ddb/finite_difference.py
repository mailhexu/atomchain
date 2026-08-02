"""Finite-difference response helpers for DDB response blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from atomchain.phonon.frozenphonon import calculate_phonon

from .conventions import (
    ANGSTROM_PER_BOHR,
    HARTREE_PER_EV,
    elastic_ev_ang3_to_ddb_strain_strain,
    normalize_qpoint,
    tensor_to_voigt,
)
from .model import DdbDerivativeBlock, DdbPerturbation
from .phonopy_converter import ddb_document_from_phonopy
from .writer import write_ddb


def strain_atoms(atoms, voigt_index, amplitude):
    """Return a copy strained by an engineering Voigt component."""
    strained = atoms.copy()
    strain = np.zeros((3, 3), dtype=float)
    if voigt_index == 0:
        strain[0, 0] = amplitude
    elif voigt_index == 1:
        strain[1, 1] = amplitude
    elif voigt_index == 2:
        strain[2, 2] = amplitude
    elif voigt_index == 3:
        strain[1, 2] = strain[2, 1] = amplitude / 2.0
    elif voigt_index == 4:
        strain[0, 2] = strain[2, 0] = amplitude / 2.0
    elif voigt_index == 5:
        strain[0, 1] = strain[1, 0] = amplitude / 2.0
    else:
        raise ValueError("voigt_index must be in 0..5")
    deformation = np.eye(3) + strain
    strained.set_cell(np.dot(atoms.cell.array, deformation), scale_atoms=True)
    return strained


def calculate_stress_response(
    atoms, calc, strain_amplitude=1e-3, cache_dir=None, difference="central"
):
    """Compute stress response d(stress)/d(strain) via finite differences.

    Parameters
    ----------
    difference : str
        ``"central"`` uses 3-point central difference (2 evaluations per component).
        ``"central5"`` uses 5-point central difference (4 evaluations per component,
        O(h^4) accuracy).
    """
    responses = np.zeros((6, 6), dtype=float)
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / "stress_response.yaml"
        if cached.exists():
            with cached.open(encoding="utf-8") as handle:
                return np.asarray(
                    yaml.safe_load(handle)["stress_response_ev_ang3"], dtype=float
                )

    for idx in range(6):
        if difference == "central5":
            # 5-point central: (-f(+2h) + 8f(+h) - 8f(-h) + f(-2h)) / (12h)
            h = strain_amplitude
            atoms_p2 = strain_atoms(atoms, idx, 2 * h)
            atoms_p1 = strain_atoms(atoms, idx, h)
            atoms_m1 = strain_atoms(atoms, idx, -h)
            atoms_m2 = strain_atoms(atoms, idx, -2 * h)
            for a in (atoms_p2, atoms_p1, atoms_m1, atoms_m2):
                a.calc = calc
            try:
                s_p2 = tensor_to_voigt(atoms_p2.get_stress(voigt=False), strain=False)
                s_p1 = tensor_to_voigt(atoms_p1.get_stress(voigt=False), strain=False)
                s_m1 = tensor_to_voigt(atoms_m1.get_stress(voigt=False), strain=False)
                s_m2 = tensor_to_voigt(atoms_m2.get_stress(voigt=False), strain=False)
            except Exception as exc:
                raise RuntimeError(
                    "Calculator does not provide usable stress for finite differences"
                ) from exc
            responses[:, idx] = (-s_p2 + 8 * s_p1 - 8 * s_m1 + s_m2) / (12.0 * h)
        else:
            # 3-point central: (f(+h) - f(-h)) / (2h)
            plus = strain_atoms(atoms, idx, strain_amplitude)
            minus = strain_atoms(atoms, idx, -strain_amplitude)
            plus.calc = calc
            minus.calc = calc
            try:
                splus = tensor_to_voigt(plus.get_stress(voigt=False), strain=False)
                sminus = tensor_to_voigt(minus.get_stress(voigt=False), strain=False)
            except Exception as exc:
                raise RuntimeError(
                    "Calculator does not provide usable stress for finite differences"
                ) from exc
            responses[:, idx] = (splus - sminus) / (2.0 * strain_amplitude)

    if cache is not None:
        with (cache / "stress_response.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "stress_response_ev_ang3": responses.tolist(),
                    "difference": difference,
                },
                handle,
                sort_keys=True,
            )
    return responses


def calculate_internal_strain_response(
    atoms, calc, strain_amplitude=1e-3, cache_dir=None, difference="central"
):
    """Compute Gamma force-response internal strain dF/d(strain).

    The returned array has shape ``(natom, 3, 6)`` in eV/Angstrom. ABINIT stores
    the corresponding second derivative with the opposite sign, because
    ``m_ddb_internalstr.F90`` defines ``instrain = -blkval``.

    Parameters
    ----------
    difference : str
        ``"central"`` uses 3-point central difference (2 evaluations per component).
        ``"central5"`` uses 5-point central difference (4 evaluations per component,
        O(h^4) accuracy).
    """
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / "internal_strain_response.yaml"
        if cached.exists():
            with cached.open(encoding="utf-8") as handle:
                return np.asarray(
                    yaml.safe_load(handle)["internal_strain_response_ev_ang"],
                    dtype=float,
                )

    response = np.zeros((len(atoms), 3, 6), dtype=float)
    for idx in range(6):
        if difference == "central5":
            h = strain_amplitude
            atoms_p2 = strain_atoms(atoms, idx, 2 * h)
            atoms_p1 = strain_atoms(atoms, idx, h)
            atoms_m1 = strain_atoms(atoms, idx, -h)
            atoms_m2 = strain_atoms(atoms, idx, -2 * h)
            for a in (atoms_p2, atoms_p1, atoms_m1, atoms_m2):
                a.calc = calc
            try:
                f_p2 = np.asarray(atoms_p2.get_forces(), dtype=float)
                f_p1 = np.asarray(atoms_p1.get_forces(), dtype=float)
                f_m1 = np.asarray(atoms_m1.get_forces(), dtype=float)
                f_m2 = np.asarray(atoms_m2.get_forces(), dtype=float)
            except Exception as exc:
                raise RuntimeError(
                    "Calculator does not provide usable forces for internal strain finite differences"
                ) from exc
            response[:, :, idx] = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12.0 * h)
        else:
            plus = strain_atoms(atoms, idx, strain_amplitude)
            minus = strain_atoms(atoms, idx, -strain_amplitude)
            plus.calc = calc
            minus.calc = calc
            try:
                fplus = np.asarray(plus.get_forces(), dtype=float)
                fminus = np.asarray(minus.get_forces(), dtype=float)
            except Exception as exc:
                raise RuntimeError(
                    "Calculator does not provide usable forces for internal strain finite differences"
                ) from exc
            response[:, :, idx] = (fplus - fminus) / (2.0 * strain_amplitude)

    if cache is not None:
        with (cache / "internal_strain_response.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(
                {
                    "internal_strain_response_ev_ang": response.tolist(),
                    "difference": difference,
                },
                handle,
                sort_keys=True,
            )
    return response


def build_elastic_blocks(atoms, stress_response, qpoint=(0, 0, 0), source="stress_fd"):
    """Build strain-strain DDB blocks from stress response in eV/Angstrom^3."""
    values = elastic_ev_ang3_to_ddb_strain_strain(stress_response, atoms.get_volume())
    qpoint = normalize_qpoint(qpoint)
    blocks = []
    for i in range(6):
        for j in range(6):
            blocks.append(
                DdbDerivativeBlock(
                    qpoint=qpoint,
                    perturbation_i=DdbPerturbation(kind="strain", voigt_index=i),
                    perturbation_j=DdbPerturbation(kind="strain", voigt_index=j),
                    value=float(values[i, j]),
                    units="Ha",
                    source=source,
                )
            )
    return blocks


def build_internal_strain_blocks(
    internal_strain_response, qpoint=(0, 0, 0), source="internal_strain_fd"
):
    """Build Gamma displacement-strain DDB blocks from dF/d(strain).

    ``internal_strain_response`` is in eV/Angstrom. DDB stores
    ``d2E / dR_bohr dstrain = -dF/dstrain`` in Hartree/Bohr.
    """
    response = np.asarray(internal_strain_response, dtype=float)
    if response.ndim != 3 or response.shape[1:] != (3, 6):
        raise ValueError("internal_strain_response must have shape (natom, 3, 6)")
    qpoint = normalize_qpoint(qpoint)
    values = -response * HARTREE_PER_EV * ANGSTROM_PER_BOHR
    blocks = []
    for atom_index in range(response.shape[0]):
        for cart_direction in range(3):
            for strain_index in range(6):
                blocks.append(
                    DdbDerivativeBlock(
                        qpoint=qpoint,
                        perturbation_i=DdbPerturbation(
                            kind="displacement",
                            atom_index=atom_index,
                            cart_direction=cart_direction,
                            qpoint=qpoint,
                        ),
                        perturbation_j=DdbPerturbation(
                            kind="strain", voigt_index=strain_index
                        ),
                        value=float(values[atom_index, cart_direction, strain_index]),
                        units="Ha/Bohr",
                        source=source,
                    )
                )
    return blocks


def build_strain_phonon_blocks(
    minus_document,
    plus_document,
    strain_index,
    strain_amplitude,
    source="strain_phonon_fd",
):
    """Build Gamma-only displacement-strain cross blocks from phonon differences.

    Finite-q ``dPhi(q)/dstrain`` is a third-order response and must not be
    encoded as an ordinary second-order displacement-strain DDB block.
    """
    minus = _block_map(minus_document.derivative_blocks)
    blocks = []
    for plus_block in plus_document.derivative_blocks:
        if normalize_qpoint(plus_block.qpoint) != normalize_qpoint((0, 0, 0)):
            continue
        if (
            plus_block.perturbation_i.kind != "displacement"
            or plus_block.perturbation_j.kind != "displacement"
        ):
            continue
        key = _block_key(plus_block)
        if key not in minus:
            continue
        derivative = (complex(plus_block.value) - complex(minus[key].value)) / (
            2.0 * strain_amplitude
        )
        blocks.append(
            DdbDerivativeBlock(
                qpoint=plus_block.qpoint,
                perturbation_i=plus_block.perturbation_i,
                perturbation_j=DdbPerturbation(kind="strain", voigt_index=strain_index),
                value=derivative,
                units=f"{plus_block.units}/strain",
                source=source,
            )
        )
    return blocks


def write_ddb_from_finite_difference(
    atoms,
    calc,
    filename="out.ddb",
    phonon_ndim=(2, 2, 2),
    qpoints=None,
    qgrid=None,
    strain_amplitude=1e-3,
    include_stress=True,
    include_strain_phonon=True,
    difference="central",
    cache_dir="ddb_fd_cache",
    phonon_kwargs=None,
    symprec=1e-5,
):
    """Compute requested finite-difference response terms and write a DDB."""
    if difference not in ("central", "central5"):
        raise ValueError("Only 'central' and 'central5' differences are supported")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    phonon_kwargs = dict(phonon_kwargs or {})
    phonon_ndim = _phonon_ndim_matrix(phonon_ndim)
    if qgrid is None:
        qgrid = tuple(np.diag(phonon_ndim).astype(int).tolist())
    base_phonon = calculate_phonon(
        atoms,
        calc=calc,
        ndim=phonon_ndim,
        phonon_save_dir=str(cache / "phonon_base"),
        **phonon_kwargs,
    )
    document = ddb_document_from_phonopy(
        atoms=atoms, phonon=base_phonon, qpoints=qpoints, qgrid=qgrid, symprec=symprec
    )
    document.reference_energy_ha = _calculate_reference_energy_ha(atoms, calc)
    document.metadata.update(
        {
            "source": "finite_difference",
            "cache_dir": str(cache),
            "strain_amplitude": strain_amplitude,
            "difference": difference,
            "include_stress": include_stress,
            "include_internal_strain": include_strain_phonon,
            "finite_q_strain_phonon_written": False,
            "qgrid": list(qgrid) if qgrid is not None else None,
        }
    )

    if include_stress:
        response = calculate_stress_response(
            atoms,
            calc,
            strain_amplitude=strain_amplitude,
            cache_dir=cache,
            difference=difference,
        )
        for block in build_elastic_blocks(atoms, response):
            document.add_derivative(block)

    if include_strain_phonon:
        response = calculate_internal_strain_response(
            atoms,
            calc,
            strain_amplitude=strain_amplitude,
            cache_dir=cache,
            difference=difference,
        )
        for block in build_internal_strain_blocks(response):
            document.add_derivative(block)

    return write_ddb(document, filename)


def _block_key(block):
    return (
        block.qpoint,
        block.perturbation_i.atom_index,
        block.perturbation_i.cart_direction,
        block.perturbation_j.atom_index,
        block.perturbation_j.cart_direction,
    )


def _block_map(blocks):
    return {_block_key(block): block for block in blocks}


def _phonon_ndim_matrix(phonon_ndim):
    ndim = np.asarray(phonon_ndim, dtype=int)
    if ndim.shape == (3,):
        return np.diag(ndim).tolist()
    if ndim.shape == (3, 3):
        return ndim.tolist()
    raise ValueError("phonon_ndim must be a length-3 vector or 3x3 matrix")


def _calculate_reference_energy_ha(atoms, calc):
    work = atoms.copy()
    work.calc = calc
    return float(work.get_potential_energy()) * HARTREE_PER_EV
