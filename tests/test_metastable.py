from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk

from atomchain.metastable import (
    _build_combined_label,
    _deduplicate_kpoints,
    _deduplicate_results,
    _get_spacegroup,
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
        )

        mock_calc_phonon.assert_called_once()
        mock_label.assert_called_once()
        mock_imag.assert_called_once()
        assert isinstance(results, dict)
        assert isinstance(results["results"], list)
        assert len(results["results"]) == 0


def test_pipeline_with_one_imaginary_mode(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()

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
        mock_write.assert_called_once()


def test_multimode_combinations(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    mock_calc = MagicMock()

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
            calc="chgnet",
            nmax=1,
            output_dir=str(tmp_path / "metastable"),
            phonon_ndim=np.diag([2, 2, 2]),
        )

        mock_init.assert_called_once_with(model_type="chgnet", model_path=None)


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
        )

        mock_init.assert_called_once_with(model_type="chgnet")


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
