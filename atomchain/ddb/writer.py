"""Deterministic ABINIT-style DDB text writer for the supported subset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .conventions import DDB_VERSION, abinit_perturbation_indices, qpoint_to_floats


def format_float(value):
    """Format floats with Fortran-style ``D`` exponents used by DDB text."""
    return f"{float(value): 22.14E}".replace("E", "D")


def format_header_label(label):
    """Format DDB header labels with ABINIT's fixed-width text layout."""
    return f"{label:>10}"


def format_header_int(label, value):
    return f"{format_header_label(label)}{int(value):10d}"


def format_header_floats(label, values):
    return f"{format_header_label(label)}" + "".join(format_float(v) for v in values)


def format_header_long_float(label, value):
    """Format current DDB fields whose label is wider than nine columns."""
    return f" {label:<11}" + format_float(value)


def format_qpt_line(qpoint, weight=1.0):
    values = "".join(f"{float(v):16.8E}" for v in qpoint)
    return f" qpt{values}{float(weight):6.1f}"


def format_header_ints(label, values, width=5):
    return f"{format_header_label(label)}     " + "".join(
        f"{int(v):{width}d}" for v in values
    )


def format_continuation_floats(values):
    return format_header_floats("", values)


def format_continuation_ints(values, width=5):
    return format_header_ints("", values, width=width)


def format_header_float_lines(label, values, per_line=3):
    values = list(values)
    lines = []
    for start in range(0, len(values), per_line):
        line_label = label if start == 0 else ""
        lines.append(format_header_floats(line_label, values[start : start + per_line]))
    return lines


def format_header_int_lines(label, values, per_line=12, width=5):
    values = list(values)
    lines = []
    for start in range(0, len(values), per_line):
        line_label = label if start == 0 else ""
        lines.append(
            format_header_ints(
                line_label, values[start : start + per_line], width=width
            )
        )
    return lines


def write_metadata(document, filename):
    """Write deterministic YAML sidecar metadata for a DDB document."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(document.to_metadata(), handle, sort_keys=True)
    return path


def write_ddb(document, filename, metadata_filename=None):
    """Write a deterministic ABINIT-style DDB text file and optional metadata."""
    document.validate()
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ddb_to_lines(document)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metadata_filename is None:
        metadata_filename = f"{path}.yaml"
    write_metadata(document, metadata_filename)
    return path


def ddb_to_lines(document):
    """Convert a DdbDocument to deterministic text lines."""
    h = document.header
    natom = len(h.atomic_numbers)
    unique_species = sorted(set(h.species))
    ntypat = len(unique_species)
    nsym = max(1, len(h.symmetry_operations))
    grouped = defaultdict(list)
    for block in document.sorted_derivative_blocks():
        grouped[block.qpoint].append(block)

    unique_atomic_numbers = []
    unique_masses = []
    for species_index in unique_species:
        idx = h.species.index(species_index)
        unique_atomic_numbers.append(h.atomic_numbers[idx])
        unique_masses.append(h.masses_amu[idx])

    symmetry_operations = h.symmetry_operations or [
        {"rotation": np.eye(3, dtype=int).tolist(), "translation": [0.0, 0.0, 0.0]}
    ]

    nband = 1

    lines = [
        "",
        " **** DERIVATIVE DATABASE ****    ",
        f"+DDB, Version number{DDB_VERSION:10d}",
        "",
        f" {h.description}",
        "",
        format_header_int("usepaw", 0),
        format_header_int("natom", natom),
        format_header_int("nkpt", 1),
        format_header_int("nsppol", 1),
        format_header_int("nsym", nsym),
        format_header_int("ntypat", ntypat),
        format_header_int("occopt", 1),
        format_header_int("nband", nband),
        format_header_floats("acell", [1.0, 1.0, 1.0]),
    ]
    lines.extend(format_header_float_lines("amu", unique_masses, per_line=3))
    lines.extend(
        [
            format_header_floats("dilatmx", [1.0]),
            format_header_floats("ecut", [0.0]),
            format_header_floats("ecutsm", [0.0]),
            format_header_int("intxc", 0),
            format_header_int("iscf", 0),
            format_header_int("ixc", 0),
            format_header_floats("kpt", [0.0, 0.0, 0.0]),
            format_header_floats("kptnrm", [1.0]),
            format_header_ints("ngfft", [1, 1, 1], width=5),
            format_header_int("nspden", 1),
            format_header_int("nspinor", 1),
        ]
    )
    lines.extend(format_header_float_lines("occ", [0.0] * nband, per_line=3))
    lines.append(
        format_header_label("rprim")
        + "".join(format_float(v) for v in h.lattice_bohr[0])
    )
    for row in h.lattice_bohr[1:]:
        lines.append(format_continuation_floats(row))
    lines.append(format_header_long_float("dfpt_sciss", 0.0))
    lines.append(format_header_floats("spinat", [0.0, 0.0, 0.0]))
    for _ in range(natom - 1):
        lines.append(format_continuation_floats([0.0, 0.0, 0.0]))
    lines.extend(format_header_int_lines("symafm", [1] * nsym, per_line=12, width=5))
    for isym, operation in enumerate(symmetry_operations):
        rotation = np.asarray(operation.get("rotation", np.eye(3)), dtype=int).reshape(
            3, 3
        )
        values = rotation.reshape(-1, order="F")
        lines.append(format_header_ints("symrel" if isym == 0 else "", values, width=5))
    for isym, operation in enumerate(symmetry_operations):
        translation = np.asarray(
            operation.get("translation", [0.0, 0.0, 0.0]), dtype=float
        )
        lines.append(format_header_floats("tnons" if isym == 0 else "", translation))
    lines.extend(
        [
            format_header_floats("tolwfr", [1.0]),
            format_header_floats("tphysel", [0.0]),
            format_header_floats("tsmear", [0.0]),
        ]
    )
    lines.extend(format_header_int_lines("typat", h.species, per_line=12, width=5))
    lines.extend(format_header_float_lines("wtk", [1.0], per_line=3))
    lines.append(format_header_floats("xred", h.reduced_positions[0]))
    for row in h.reduced_positions[1:]:
        lines.append(format_continuation_floats(row))
    lines.extend(format_header_float_lines("znucl", unique_atomic_numbers, per_line=3))
    lines.extend(format_header_float_lines("zion", unique_atomic_numbers, per_line=3))
    nblocks = len(grouped) + (1 if document.reference_energy_ha is not None else 0)
    lines.extend(
        [
            " ",
            " No information on the potentials yet",
            " **** Database of total energy derivatives ****",
            " ",
            f" Number of data blocks={nblocks:5d}",
        ]
    )

    if document.reference_energy_ha is not None:
        lines.append("")
        lines.append(" Total energy                 - # elements :           1")
        lines.append(f"{format_float(document.reference_energy_ha)}{format_float(0.0)}")

    for qpoint in sorted(grouped, key=qpoint_to_floats):
        blocks = sorted(grouped[qpoint], key=_line_sort_key)
        lines.append("")
        lines.append(f" 2nd derivatives (non-stat.)  - # elements :{len(blocks):12d}")
        qx, qy, qz = qpoint_to_floats(qpoint)
        lines.append(format_qpt_line([qx, qy, qz]))
        for block in blocks:
            idir1, ipert1 = abinit_perturbation_indices(block.perturbation_i, natom)
            idir2, ipert2 = abinit_perturbation_indices(block.perturbation_j, natom)
            value = complex(block.value)
            lines.append(
                f"{idir1:4d}{ipert1:5d}{idir2:5d}{ipert2:5d}"
                f" {format_float(value.real)} {format_float(value.imag)}"
            )
    lines.append("")
    lines.append("List of bloks and their characteristics")
    lines.append(" ")
    if document.reference_energy_ha is not None:
        lines.append(" Total energy                 - # elements :           1")
        lines.append(" ")
    for qpoint in sorted(grouped, key=qpoint_to_floats):
        blocks = grouped[qpoint]
        lines.append(f" 2nd derivatives (non-stat.)  - # elements :{len(blocks):12d}")
        lines.append(format_qpt_line(qpoint_to_floats(qpoint)))
        lines.append(" ")
    return lines


def _line_sort_key(block):
    def pkey(pert):
        return (
            pert.kind,
            pert.atom_index if pert.atom_index is not None else -1,
            pert.cart_direction if pert.cart_direction is not None else -1,
            pert.voigt_index if pert.voigt_index is not None else -1,
        )

    return pkey(block.perturbation_i) + pkey(block.perturbation_j)
