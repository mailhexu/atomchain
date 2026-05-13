import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from atomchain.ddb.conventions import elastic_ddb_strain_strain_to_ev_ang3
from atomchain.ddb.finite_difference import (
    build_elastic_blocks,
    build_internal_strain_blocks,
    build_strain_phonon_blocks,
    calculate_internal_strain_response,
    calculate_stress_response,
    strain_atoms,
)
from atomchain.ddb.model import (
    DdbDerivativeBlock,
    DdbDocument,
    DdbHeader,
    DdbPerturbation,
)


class LinearStressCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cell = atoms.cell.array
        ref = np.eye(3) * 4.05
        deformation = np.linalg.solve(ref, cell)
        strain = deformation - np.eye(3)
        voigt = np.array(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                2 * strain[1, 2],
                2 * strain[0, 2],
                2 * strain[0, 1],
            ]
        )
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = 2.0 * voigt


class ElasticTensorCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, reference_atoms, elastic_tensor, **kwargs):
        super().__init__(**kwargs)
        self.reference_cell = reference_atoms.cell.array.copy()
        self.elastic_tensor = np.asarray(elastic_tensor, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        deformation = np.linalg.solve(self.reference_cell, atoms.cell.array)
        strain = deformation - np.eye(3)
        strain_voigt = np.array(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                2.0 * strain[1, 2],
                2.0 * strain[0, 2],
                2.0 * strain[0, 1],
            ]
        )
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = self.elastic_tensor @ strain_voigt


class InternalStrainCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, reference_atoms, response, **kwargs):
        super().__init__(**kwargs)
        self.reference_cell = reference_atoms.cell.array.copy()
        self.response = np.asarray(response, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        deformation = np.linalg.solve(self.reference_cell, atoms.cell.array)
        strain = deformation - np.eye(3)
        strain_voigt = np.array(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                2.0 * strain[1, 2],
                2.0 * strain[0, 2],
                2.0 * strain[0, 1],
            ]
        )
        self.results["energy"] = 0.0
        self.results["forces"] = np.einsum("aci,i->ac", self.response, strain_voigt)
        self.results["stress"] = np.zeros(6)


def test_strain_atoms_changes_cell_and_preserves_scaled_positions():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    strained = strain_atoms(atoms, 0, 0.01)
    assert np.isclose(strained.cell.lengths()[0], atoms.cell.lengths()[0] * 1.01)
    assert np.allclose(strained.get_scaled_positions(), atoms.get_scaled_positions())


def test_stress_finite_difference_with_synthetic_calculator(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    response = calculate_stress_response(
        atoms, LinearStressCalculator(), strain_amplitude=1e-4, cache_dir=tmp_path
    )
    assert response.shape == (6, 6)
    assert np.allclose(np.diag(response), 2.0, atol=1e-6)


def test_stress_finite_difference_recovers_full_elastic_tensor(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    elastic = np.array(
        [
            [220.0, 120.0, 110.0, 4.0, 5.0, 6.0],
            [120.0, 230.0, 115.0, 7.0, 8.0, 9.0],
            [110.0, 115.0, 240.0, 10.0, 11.0, 12.0],
            [4.0, 7.0, 10.0, 80.0, 13.0, 14.0],
            [5.0, 8.0, 11.0, 13.0, 90.0, 15.0],
            [6.0, 9.0, 12.0, 14.0, 15.0, 100.0],
        ]
    )
    response = calculate_stress_response(
        atoms,
        ElasticTensorCalculator(atoms, elastic),
        strain_amplitude=1e-5,
        cache_dir=tmp_path,
    )
    np.testing.assert_allclose(response, elastic, rtol=1e-10, atol=1e-10)


def test_missing_stress_support_raises_clear_error():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    with pytest.raises(RuntimeError, match="stress"):
        calculate_stress_response(atoms, Calculator(), strain_amplitude=1e-4)


def test_build_elastic_blocks_uses_strain_pairs():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    blocks = build_elastic_blocks(atoms, np.eye(6))
    assert len(blocks) == 36
    assert all(block.perturbation_i.kind == "strain" for block in blocks)
    assert all(block.perturbation_j.kind == "strain" for block in blocks)


def test_internal_strain_finite_difference_recovers_force_response(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    expected = (
        np.arange(len(atoms) * 3 * 6, dtype=float).reshape(len(atoms), 3, 6) / 10.0
    )
    response = calculate_internal_strain_response(
        atoms,
        InternalStrainCalculator(atoms, expected),
        strain_amplitude=1e-5,
        cache_dir=tmp_path,
    )
    np.testing.assert_allclose(response, expected, rtol=1e-10, atol=1e-10)


def test_build_internal_strain_blocks_are_gamma_displacement_strain():
    response = np.ones((2, 3, 6))
    blocks = build_internal_strain_blocks(response)
    assert len(blocks) == 36
    assert {block.qpoint for block in blocks} == {(0, 0, 0)}
    assert all(block.perturbation_i.kind == "displacement" for block in blocks)
    assert all(block.perturbation_j.kind == "strain" for block in blocks)
    assert all(block.value < 0.0 for block in blocks)


def test_elastic_blocks_round_trip_to_elastic_tensor():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    elastic = np.array(
        [
            [1.0, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 2.0, 0.7, 0.8, 0.9, 1.0],
            [0.3, 0.7, 3.0, 1.1, 1.2, 1.3],
            [0.4, 0.8, 1.1, 4.0, 1.4, 1.5],
            [0.5, 0.9, 1.2, 1.4, 5.0, 1.6],
            [0.6, 1.0, 1.3, 1.5, 1.6, 6.0],
        ]
    )
    blocks = build_elastic_blocks(atoms, elastic)
    stored = np.zeros((6, 6))
    for block in blocks:
        stored[block.perturbation_i.voigt_index, block.perturbation_j.voigt_index] = (
            block.value
        )

    recovered = elastic_ddb_strain_strain_to_ev_ang3(stored, atoms.get_volume())
    np.testing.assert_allclose(recovered, elastic, rtol=1e-12, atol=1e-12)


def test_build_strain_phonon_cross_blocks():
    atoms = bulk("Al", "fcc", a=4.05)
    header = DdbHeader.from_atoms(atoms)
    minus = DdbDocument(header=header)
    plus = DdbDocument(header=header)
    for document, value in [(minus, 1.0), (plus, 3.0)]:
        document.add_derivative(
            DdbDerivativeBlock(
                qpoint=(0, 0, 0),
                perturbation_i=DdbPerturbation(
                    kind="displacement", atom_index=0, cart_direction=0
                ),
                perturbation_j=DdbPerturbation(
                    kind="displacement", atom_index=0, cart_direction=0
                ),
                value=value,
                units="Ha/Bohr^2",
                source="phonopy",
            )
        )
    blocks = build_strain_phonon_blocks(
        minus, plus, strain_index=2, strain_amplitude=0.5
    )
    assert len(blocks) == 1
    assert blocks[0].perturbation_j.kind == "strain"
    assert blocks[0].value == 2.0


def test_build_strain_phonon_cross_blocks_skips_finite_q():
    atoms = bulk("Al", "fcc", a=4.05)
    header = DdbHeader.from_atoms(atoms)
    minus = DdbDocument(header=header)
    plus = DdbDocument(header=header)
    for document, value in [(minus, 1.0), (plus, 3.0)]:
        document.add_derivative(
            DdbDerivativeBlock(
                qpoint=(0.5, 0, 0),
                perturbation_i=DdbPerturbation(
                    kind="displacement", atom_index=0, cart_direction=0
                ),
                perturbation_j=DdbPerturbation(
                    kind="displacement", atom_index=0, cart_direction=0
                ),
                value=value,
                units="Ha/Bohr^2",
                source="phonopy",
            )
        )
    assert (
        build_strain_phonon_blocks(minus, plus, strain_index=2, strain_amplitude=0.5)
        == []
    )
