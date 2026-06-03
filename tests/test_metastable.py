import pickle
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk

from atomchain.metastable import (
    _build_combined_label,
    _checkpoint_metadata,
    _deduplicate_kpoints,
    _deduplicate_results,
    _get_spacegroup,
    _scale_modulation,
    explore_metastable_states,
)


def test_get_spacegroup():
    atoms = bulk("Al", "fcc", a=4.05)
    sg_number, sg_name = _get_spacegroup(atoms)
    assert sg_number == 225
    assert sg_name is not None


def test_deduplicate_kpoints():
    kpts = [[0.5, 0, 0], [0.5, 0, 0], [0, 0.5, 0]]
    unique = _deduplicate_kpoints(kpts)
    assert len(unique) == 2


def test_pipeline_with_mocks(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()

    mock_phonon = MagicMock()
    mock_phonon.save = MagicMock()

    with patch(
        "atomchain.metastable.calculate_phonon", return_value=mock_phonon
    ) as mock_calc_phonon, patch(
        "atomchain.metastable.get_all_labeled_modes", return_value={}
    ) as mock_label, patch(
        "atomchain.metastable.get_imaginary_modes", return_value=[]
    ) as mock_imag:
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=1,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

        mock_calc_phonon.assert_called_once()
        mock_label.assert_called_once()
        mock_imag.assert_called_once()
        assert isinstance(results, dict)
        assert isinstance(results["results"], list)
        assert len(results["results"]) == 0


def test_parent_is_symmetry_cell_relaxed_before_phonons(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()
    relaxed_parent = atoms.copy()
    relaxed_parent.info["relaxed_parent"] = True

    with patch(
        "atomchain.metastable.relax_with_ml", return_value=relaxed_parent
    ) as mock_relax, patch(
        "atomchain.metastable.calculate_phonon", return_value=MagicMock()
    ) as mock_phonon, patch(
        "atomchain.metastable.get_all_labeled_modes", return_value={}
    ), patch("atomchain.metastable.get_imaginary_modes", return_value=[]), patch(
        "atomchain.metastable.write"
    ) as mock_write:
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=1,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_kwargs={"fmax": 0.02},
        )

    mock_relax.assert_called_once_with(
        atoms, calc=mock_calc, sym=True, relax_cell=True, fmax=0.02
    )
    assert mock_phonon.call_args.args[0] is relaxed_parent
    assert results["parent_atoms"] is relaxed_parent
    mock_write.assert_called_once()


def test_pipeline_with_one_imaginary_mode(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()
    mock_calc.get_forces.return_value = np.zeros((1, 3))

    mock_phonon = MagicMock()
    mock_phonon.save = MagicMock()

    imaginary_modes = [
        {
            "kpoint_label": "X",
            "band_index": 0,
            "frequency": -1.5,
            "bcs_label": "X1",
            "mulliken_label": None,
            "degeneracy": 1,
        }
    ]

    all_labeled = {
        "X": {
            "frequencies": np.array([-1.5, 0.0, 0.0, 0.0, 5.0, 5.0]),
            "modes": [
                {
                    "band_index": i,
                    "frequency": f,
                    "bcs_label": None,
                    "mulliken_label": None,
                }
                for i, f in enumerate([-1.5, 0.0, 0.0, 0.0, 5.0, 5.0])
            ],
        }
    }

    from ase import Atoms

    modulated = bulk("Al", "fcc", a=4.05) * [2, 2, 2]

    mock_relaxed = MagicMock(spec=Atoms)
    mock_relaxed.get_potential_energy.return_value = -100.5
    mock_relaxed.get_cell.return_value = modulated.get_cell()
    mock_relaxed.get_scaled_positions.return_value = modulated.get_scaled_positions()
    mock_relaxed.get_atomic_numbers.return_value = modulated.get_atomic_numbers()
    mock_relaxed.__len__ = lambda self: len(modulated)

    mock_opd_result = {
        "atoms": modulated,
        "opd_label": "(a,0,0)",
        "opd_vector_eigvec": [1.0, 0.0, 0.0],
        "polarization_direction": [1.0, 0.0, 0.0],
        "is_maximal": True,
        "is_gamma": True,
    }

    with patch(
        "atomchain.metastable.calculate_phonon", return_value=mock_phonon
    ), patch(
        "atomchain.metastable.get_all_labeled_modes", return_value=all_labeled
    ), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=imaginary_modes
    ), patch(
        "atomchain.metastable.get_high_symmetry_kpoints",
        return_value=[{"label": "X", "qpoint": [0.5, 0.0, 0.0]}],
    ), patch(
        "atomchain.metastable.get_modulations_with_opd_info",
        return_value=[mock_opd_result],
    ), patch("atomchain.metastable.relax_with_ml", return_value=mock_relaxed), patch(
        "atomchain.metastable.write"
    ) as mock_write:
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=1,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

        assert len(results["results"]) == 1
        r = results["results"][0]
        assert "id" in r
        assert "energy" in r
        assert r["energy"] == -100.5
        assert "spacegroup_number" in r
        assert "source_modes" in r
        assert len(r["source_modes"]) == 1
        assert r["source_modes"][0]["kpoint_label"] == "X"
        assert r["source_modes"][0]["band_index"] == 0
        assert r["source_modes"][0]["frequency"] == -1.5
        assert mock_write.call_count == 2


def test_failed_relaxation_records_failed_result(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()
    mock_calc.get_forces.return_value = np.zeros((1, 3))
    mock_phonon = MagicMock()
    modulated = bulk("Al", "fcc", a=4.05) * [2, 2, 2]

    imaginary_modes = [
        {
            "kpoint_label": "X",
            "band_index": 0,
            "frequency": -1.5,
            "bcs_label": "X1",
        }
    ]
    mock_opd_result = {
        "atoms": modulated,
        "opd_label": "(a)",
        "is_maximal": True,
        "polarization_direction": None,
    }

    with patch(
        "atomchain.metastable.calculate_phonon", return_value=mock_phonon
    ), patch("atomchain.metastable.get_all_labeled_modes", return_value={}), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=imaginary_modes
    ), patch(
        "atomchain.metastable.get_high_symmetry_kpoints",
        return_value=[{"label": "X", "qpoint": [0.5, 0.0, 0.0]}],
    ), patch(
        "atomchain.metastable.get_modulations_with_opd_info",
        return_value=[mock_opd_result],
    ), patch(
        "atomchain.metastable.relax_with_ml", side_effect=RuntimeError("relax failed")
    ) as mock_relax, patch("atomchain.metastable.write"):
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=1,
            amplitude=0.5,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

    result = results["results"][0]
    assert result["status"] == "failed"
    assert result["error_stage"] == "relaxation"
    assert "relax failed" in result["error_message"]
    assert result["attempt"] == 3
    assert result["amplitude"] == 0.125
    assert mock_relax.call_count == 3


def test_failed_relaxation_retries_with_smaller_amplitude(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()
    mock_calc.get_forces.return_value = np.zeros((1, 3))
    mock_phonon = MagicMock()
    modulated = bulk("Al", "fcc", a=4.05) * [2, 2, 2]
    mock_relaxed = modulated.copy()
    mock_relaxed.calc = MagicMock()
    mock_relaxed.get_potential_energy = MagicMock(return_value=-100.5)

    imaginary_modes = [
        {
            "kpoint_label": "X",
            "band_index": 0,
            "frequency": -1.5,
            "bcs_label": "X1",
        }
    ]
    mock_opd_result = {
        "atoms": modulated,
        "opd_label": "(a)",
        "is_maximal": True,
        "polarization_direction": None,
    }

    with patch(
        "atomchain.metastable.calculate_phonon", return_value=mock_phonon
    ), patch("atomchain.metastable.get_all_labeled_modes", return_value={}), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=imaginary_modes
    ), patch(
        "atomchain.metastable.get_high_symmetry_kpoints",
        return_value=[{"label": "X", "qpoint": [0.5, 0.0, 0.0]}],
    ), patch(
        "atomchain.metastable.get_modulations_with_opd_info",
        return_value=[mock_opd_result],
    ), patch(
        "atomchain.metastable.relax_with_ml",
        side_effect=[RuntimeError("first failed"), mock_relaxed],
    ) as mock_relax, patch("atomchain.metastable.write"):
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=1,
            amplitude=0.5,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

    result = results["results"][0]
    assert result["status"] == "success"
    assert result["attempt"] == 2
    assert result["amplitude"] == 0.25
    assert mock_relax.call_count == 2


def test_checkpoint_skips_completed_candidates_by_default(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()
    mock_phonon = MagicMock()
    output_dir = tmp_path / "metastable"
    output_dir.mkdir()
    checkpoint_result = {
        "id": 1,
        "status": "success",
        "energy": -1.0,
        "energy_per_fu": -1.0,
        "delta_e_per_fu": 0.0,
        "spacegroup_number": 225,
        "spacegroup_name": "Fm-3m",
        "source_modes": [],
        "combined_label": "X1(a)",
        "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        "n_atoms": 8,
    }
    with open(output_dir / "checkpoint.pkl", "wb") as handle:
        pickle.dump(
            {
                "metadata": _checkpoint_metadata(
                    atoms, 1, 0.5, 64, np.diag([2, 2, 2]), 1e-5, True
                ),
                "results": [checkpoint_result],
            },
            handle,
        )

    modulated = bulk("Al", "fcc", a=4.05) * [2, 2, 2]
    imaginary_modes = [
        {
            "kpoint_label": "X",
            "band_index": 0,
            "frequency": -1.5,
            "bcs_label": "X1",
        }
    ]
    mock_opd_result = {
        "atoms": modulated,
        "opd_label": "(a)",
        "is_maximal": True,
        "polarization_direction": None,
    }

    with patch(
        "atomchain.metastable.calculate_phonon", return_value=mock_phonon
    ), patch("atomchain.metastable.get_all_labeled_modes", return_value={}), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=imaginary_modes
    ), patch(
        "atomchain.metastable.get_high_symmetry_kpoints",
        return_value=[{"label": "X", "qpoint": [0.5, 0.0, 0.0]}],
    ), patch(
        "atomchain.metastable.get_modulations_with_opd_info",
        return_value=[mock_opd_result],
    ), patch("atomchain.metastable.relax_with_ml") as mock_relax:
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=1,
            output_dir=str(output_dir),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

    assert len(results["results"]) == 1
    assert results["results"][0]["id"] == checkpoint_result["id"]
    assert (
        results["results"][0]["combined_label"] == checkpoint_result["combined_label"]
    )
    mock_relax.assert_not_called()


def test_pre_relaxation_force_screen_halves_amplitude(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()
    mock_calc.get_forces.side_effect = [
        np.array([[12.0, 0.0, 0.0]]),
        np.array([[6.0, 0.0, 0.0]]),
        np.array([[6.0, 0.0, 0.0]]),
    ]
    mock_phonon = MagicMock()
    modulated = bulk("Al", "fcc", a=4.05) * [2, 2, 2]
    mock_relaxed = modulated.copy()
    mock_relaxed.calc = MagicMock()
    mock_relaxed.get_potential_energy = MagicMock(return_value=-100.5)

    imaginary_modes = [
        {
            "kpoint_label": "X",
            "band_index": 0,
            "frequency": -1.5,
            "bcs_label": "X1",
        }
    ]
    mock_opd_result = {
        "atoms": modulated,
        "opd_label": "(a)",
        "is_maximal": True,
        "polarization_direction": None,
    }

    with patch(
        "atomchain.metastable.calculate_phonon", return_value=mock_phonon
    ), patch("atomchain.metastable.get_all_labeled_modes", return_value={}), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=imaginary_modes
    ), patch(
        "atomchain.metastable.get_high_symmetry_kpoints",
        return_value=[{"label": "X", "qpoint": [0.5, 0.0, 0.0]}],
    ), patch(
        "atomchain.metastable.get_modulations_with_opd_info",
        return_value=[mock_opd_result],
    ), patch("atomchain.metastable.relax_with_ml", return_value=mock_relaxed), patch(
        "atomchain.metastable.write"
    ):
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=1,
            amplitude=0.5,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

    result = results["results"][0]
    assert result["status"] == "success"
    assert result["amplitude"] == 0.25
    assert result["pre_relax_max_force"] == 6.0
    assert result["force_screen_reductions"] == 1


def test_scale_modulation_uses_periodic_minimum_image():
    atoms = bulk("Al", "fcc", a=4.05)
    reference = atoms.copy()
    modulated = atoms.copy()
    reference.set_scaled_positions([[0.99, 0.0, 0.0]])
    modulated.set_scaled_positions([[0.01, 0.0, 0.0]])

    scaled = _scale_modulation(modulated, reference, 0.5, 1.0)

    np.testing.assert_allclose(scaled.get_scaled_positions()[0], [0.0, 0.0, 0.0])


def test_max_cell_size_limits_generated_atom_count(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05) * [2, 1, 1]
    mock_calc = MagicMock()
    mock_phonon = MagicMock()
    imaginary_modes = [
        {
            "kpoint_label": "X",
            "band_index": 0,
            "frequency": -1.5,
            "bcs_label": "X1",
        }
    ]

    with patch(
        "atomchain.metastable.calculate_phonon", return_value=mock_phonon
    ), patch("atomchain.metastable.get_all_labeled_modes", return_value={}), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=imaginary_modes
    ), patch(
        "atomchain.metastable.get_high_symmetry_kpoints",
        return_value=[{"label": "X", "qpoint": [0.5, 0.0, 0.0]}],
    ), patch(
        "atomchain.metastable.get_modulations_with_opd_info"
    ) as mock_modulations, patch("atomchain.metastable.relax_with_ml") as mock_relax:
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=1,
            max_cell_size=3,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

    assert results["results"] == []
    mock_modulations.assert_not_called()
    mock_relax.assert_not_called()


def test_multimode_combinations(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()
    mock_calc.get_forces.return_value = np.zeros((1, 3))

    mock_phonon = MagicMock()

    imaginary_modes = [
        {
            "kpoint_label": "X",
            "band_index": 0,
            "frequency": -1.5,
            "bcs_label": "X1-",
            "mulliken_label": None,
            "degeneracy": 1,
        },
        {
            "kpoint_label": "R",
            "band_index": 1,
            "frequency": -0.8,
            "bcs_label": "R5-",
            "mulliken_label": None,
            "degeneracy": 1,
        },
    ]

    all_labeled = {}

    from ase import Atoms

    modulated = bulk("Al", "fcc", a=4.05) * [2, 2, 2]

    mock_relaxed = MagicMock(spec=Atoms)
    mock_relaxed.get_potential_energy.return_value = -200.0
    mock_relaxed.get_cell.return_value = modulated.get_cell()
    mock_relaxed.get_scaled_positions.return_value = modulated.get_scaled_positions()
    mock_relaxed.get_atomic_numbers.return_value = modulated.get_atomic_numbers()
    mock_relaxed.__len__ = lambda self: len(modulated)

    mock_opd_result = {
        "atoms": modulated,
        "opd_label": "(a)",
        "opd_vector_eigvec": [1.0],
        "polarization_direction": None,
        "is_maximal": True,
        "is_gamma": False,
    }

    mock_multi_result = {
        "atoms": modulated,
        "opd_labels": ["(a)", "(a)"],
        "opd_is_maximal": True,
    }

    with patch(
        "atomchain.metastable.calculate_phonon", return_value=mock_phonon
    ), patch(
        "atomchain.metastable.get_all_labeled_modes", return_value=all_labeled
    ), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=imaginary_modes
    ), patch(
        "atomchain.metastable.get_high_symmetry_kpoints",
        return_value=[
            {"label": "X", "qpoint": [0.5, 0.0, 0.0]},
            {"label": "R", "qpoint": [0.5, 0.5, 0.5]},
        ],
    ), patch(
        "atomchain.metastable.get_modulations_with_opd_info",
        return_value=[mock_opd_result],
    ), patch(
        "atomchain.metastable.get_multi_mode_modulations_with_info",
        return_value=([mock_multi_result], modulated),
    ), patch("atomchain.metastable.relax_with_ml", return_value=mock_relaxed), patch(
        "atomchain.metastable.write"
    ):
        results = explore_metastable_states(
            atoms,
            calc=mock_calc,
            nmax=2,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            deduplicate=False,
            relax_parent=False,
        )

        assert isinstance(results, dict)
        assert isinstance(results["results"], list)
        multi_mode_results = [
            r for r in results["results"] if len(r["source_modes"]) > 1
        ]
        assert len(multi_mode_results) >= 1
        r = multi_mode_results[0]
        assert "combined_label" in r
        assert "/" in r["combined_label"]
        assert "delta_e_per_fu" in r


def test_calc_string_init(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)

    with patch(
        "atomchain.metastable.init_calc", return_value=MagicMock()
    ) as mock_init, patch(
        "atomchain.metastable.calculate_phonon", return_value=MagicMock()
    ), patch("atomchain.metastable.get_all_labeled_modes", return_value={}), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=[]
    ):
        explore_metastable_states(
            atoms,
            calc="mace",
            nmax=1,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

        mock_init.assert_called_once_with(model_type="mace", model_path=None)


def test_calc_none_init(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)

    with patch(
        "atomchain.metastable.init_calc", return_value=MagicMock()
    ) as mock_init, patch(
        "atomchain.metastable.calculate_phonon", return_value=MagicMock()
    ), patch("atomchain.metastable.get_all_labeled_modes", return_value={}), patch(
        "atomchain.metastable.get_imaginary_modes", return_value=[]
    ):
        explore_metastable_states(
            atoms,
            calc=None,
            nmax=1,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
            relax_parent=False,
        )

        mock_init.assert_called_once_with(model_type="mace")


def test_deduplicate_results():
    results = [
        {"spacegroup_number": 99, "energy_per_fu": -10.001, "n_atoms": 5},
        {"spacegroup_number": 99, "energy_per_fu": -10.0012, "n_atoms": 5},
        {"spacegroup_number": 99, "energy_per_fu": -10.5, "n_atoms": 5},
        {"spacegroup_number": 38, "energy_per_fu": -10.001, "n_atoms": 5},
    ]
    deduped = _deduplicate_results(results, energy_tol=1e-2)
    assert len(deduped) == 3


def test_build_combined_label_with_opd():
    modes = [
        {"bcs_label": "GM4-", "kpoint_label": "GM"},
        {"bcs_label": "M2-", "kpoint_label": "M"},
    ]
    label_no_opd = _build_combined_label(modes)
    assert label_no_opd == "GM4-/M2-"
    label_with_opd = _build_combined_label(modes, opd_labels=["(a,0,0)", "(a)"])
    assert label_with_opd == "GM4-(a,0,0)/M2-(a)"
