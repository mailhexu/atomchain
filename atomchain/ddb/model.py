"""Internal DDB data model based on explicit perturbation pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Literal

import numpy as np
from ase import Atoms

from .conventions import angstrom_to_bohr, normalize_qpoint, qpoint_to_floats

PerturbationKind = Literal["displacement", "strain", "stress"]


def _clean_value(value):
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return _clean_value(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [_clean_value(v) for v in value]
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _clean_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return value


@dataclass(frozen=True)
class DdbPerturbation:
    """One first-order perturbation in a DDB derivative pair."""

    kind: PerturbationKind
    atom_index: int | None = None
    cart_direction: int | None = None
    voigt_index: int | None = None
    qpoint: tuple[Fraction, Fraction, Fraction] | None = None
    label: str = ""

    def __post_init__(self):
        if self.kind == "displacement":
            if self.atom_index is None or self.cart_direction is None:
                raise ValueError(
                    "displacement perturbations require atom_index and cart_direction"
                )
            if not 0 <= self.cart_direction <= 2:
                raise ValueError("cart_direction must be in 0..2")
        elif self.kind in {"strain", "stress"}:
            if self.voigt_index is None or not 0 <= self.voigt_index <= 5:
                raise ValueError(
                    "strain/stress perturbations require voigt_index in 0..5"
                )
        else:
            raise ValueError(f"Unsupported perturbation kind: {self.kind}")

    def to_dict(self):
        return _clean_value(asdict(self))


@dataclass
class DdbDerivativeBlock:
    """One second-order derivative value for a perturbation pair."""

    qpoint: tuple[Fraction, Fraction, Fraction]
    perturbation_i: DdbPerturbation
    perturbation_j: DdbPerturbation
    value: complex | float
    units: str
    source: str

    def __post_init__(self):
        self.qpoint = normalize_qpoint(self.qpoint)

    def to_dict(self):
        return _clean_value(asdict(self))


@dataclass
class DdbQPointBlock:
    """Metadata for a q-point present in the document."""

    qpoint: tuple[Fraction, Fraction, Fraction]
    weight: float = 1.0

    def __post_init__(self):
        self.qpoint = normalize_qpoint(self.qpoint)

    def to_dict(self):
        return _clean_value(asdict(self))


@dataclass
class DdbHeader:
    """Structure metadata needed for the supported DDB subset."""

    lattice_bohr: np.ndarray
    reciprocal_lattice_bohr: np.ndarray
    atomic_numbers: list[int]
    masses_amu: list[float]
    reduced_positions: np.ndarray
    species: list[int]
    chemical_symbols: list[str]
    symmetry_operations: list[dict[str, Any]] = field(default_factory=list)
    description: str = "atomchain generated DDB subset"

    @classmethod
    def from_atoms(cls, atoms: Atoms, symprec=1e-5):
        masses = [float(m) for m in atoms.get_masses()]
        symbols = atoms.get_chemical_symbols()
        unique_symbols = []
        species = []
        for symbol in symbols:
            if symbol not in unique_symbols:
                unique_symbols.append(symbol)
            species.append(unique_symbols.index(symbol) + 1)
        lattice_bohr = angstrom_to_bohr(np.asarray(atoms.get_cell()))
        reciprocal_lattice_bohr = 2.0 * np.pi * np.linalg.inv(lattice_bohr).T
        return cls(
            lattice_bohr=lattice_bohr,
            reciprocal_lattice_bohr=reciprocal_lattice_bohr,
            atomic_numbers=[int(z) for z in atoms.get_atomic_numbers()],
            masses_amu=masses,
            reduced_positions=np.asarray(
                atoms.get_scaled_positions(wrap=True), dtype=float
            ),
            species=species,
            chemical_symbols=symbols,
            symmetry_operations=_symmetry_operations(atoms, symprec=symprec),
        )

    def to_dict(self):
        return _clean_value(asdict(self))


@dataclass
class DdbDocument:
    """Complete supported-subset DDB document."""

    header: DdbHeader
    reference_energy_ha: float | None = None
    qpoint_blocks: list[DdbQPointBlock] = field(default_factory=list)
    derivative_blocks: list[DdbDerivativeBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_derivative(self, block: DdbDerivativeBlock):
        self.derivative_blocks.append(block)
        qpoints = {q.qpoint for q in self.qpoint_blocks}
        if block.qpoint not in qpoints:
            self.qpoint_blocks.append(DdbQPointBlock(block.qpoint))

    def sorted_derivative_blocks(self):
        return sorted(
            self.derivative_blocks,
            key=lambda b: (
                qpoint_to_floats(b.qpoint),
                b.perturbation_i.kind,
                b.perturbation_i.atom_index
                if b.perturbation_i.atom_index is not None
                else -1,
                b.perturbation_i.cart_direction
                if b.perturbation_i.cart_direction is not None
                else -1,
                b.perturbation_i.voigt_index
                if b.perturbation_i.voigt_index is not None
                else -1,
                b.perturbation_j.kind,
                b.perturbation_j.atom_index
                if b.perturbation_j.atom_index is not None
                else -1,
                b.perturbation_j.cart_direction
                if b.perturbation_j.cart_direction is not None
                else -1,
                b.perturbation_j.voigt_index
                if b.perturbation_j.voigt_index is not None
                else -1,
            ),
        )

    def validate(self):
        natom = len(self.header.atomic_numbers)
        if self.header.reduced_positions.shape != (natom, 3):
            raise ValueError("reduced_positions must have shape (natom, 3)")
        if self.header.lattice_bohr.shape != (3, 3):
            raise ValueError("lattice_bohr must have shape (3, 3)")
        for block in self.derivative_blocks:
            for perturbation in (block.perturbation_i, block.perturbation_j):
                if (
                    perturbation.kind == "displacement"
                    and perturbation.atom_index >= natom
                ):
                    raise ValueError("displacement atom_index exceeds natom")
        return True

    def to_metadata(self):
        self.validate()
        return _clean_value(
            {
                "format": "atomchain-ddb-sidecar-v1",
                "header": self.header.to_dict(),
                "reference_energy_ha": None
                if self.reference_energy_ha is None
                else float(self.reference_energy_ha),
                "qpoint_blocks": [
                    q.to_dict()
                    for q in sorted(
                        self.qpoint_blocks, key=lambda q: qpoint_to_floats(q.qpoint)
                    )
                ],
                "derivative_blocks": [
                    b.to_dict() for b in self.sorted_derivative_blocks()
                ],
                "metadata": self.metadata,
            }
        )


def _symmetry_operations(atoms, symprec=1e-5):
    try:
        import spglib
    except Exception:
        return []
    cell = (
        np.asarray(atoms.get_cell()),
        atoms.get_scaled_positions(wrap=True),
        atoms.get_atomic_numbers(),
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        return []
    rotations = dataset["rotations"] if isinstance(dataset, dict) else dataset.rotations
    translations = (
        dataset["translations"] if isinstance(dataset, dict) else dataset.translations
    )
    return [
        {
            "rotation": np.asarray(rot, dtype=int).tolist(),
            "translation": np.asarray(trans, dtype=float).tolist(),
        }
        for rot, trans in zip(rotations, translations)
    ]
