import importlib.util
import os

import numpy as np
import pytest

from atomchain.phonon.irreps import (
    get_all_labeled_modes,
    get_imaginary_modes,
    label_phonon_modes,
)


def _symphon_backend_available():
    try:
        from atomchain.kpoints import _patch_symphon

        _patch_symphon()
        if importlib.util.find_spec("symphon.irreps.backend") is None:
            return False

        return True
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.fixture
def phonopy_yaml(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from atomchain.phonon.frozenphonon import calculate_phonon

    atoms = bulk("Al", "fcc", a=4.05)
    calc = EMT()
    save_dir = str(tmp_path / "phonon_save")
    calculate_phonon(
        atoms,
        calc=calc,
        ndim=np.diag([2, 2, 2]),
        phonon_save_dir=save_dir,
        parallel=False,
    )
    return os.path.join(save_dir, "phonopy_params.yaml")


@pytest.mark.skipif(
    not _symphon_backend_available(),
    reason="symphon irreps backend not available (missing projection module)",
)
def test_label_phonon_modes_gamma(phonopy_yaml):
    result = label_phonon_modes(phonopy_yaml, [0, 0, 0], kpname="GM")
    assert "frequencies" in result
    assert "modes" in result
    assert len(result["modes"]) > 0
    for m in result["modes"]:
        assert "band_index" in m
        assert "frequency" in m
        assert "bcs_label" in m


@pytest.mark.skipif(
    not _symphon_backend_available(),
    reason="symphon irreps backend not available (missing projection module)",
)
def test_get_all_labeled_modes(phonopy_yaml):
    result = get_all_labeled_modes(phonopy_yaml)
    assert isinstance(result, dict)
    assert len(result) > 0
    for label, data in result.items():
        assert "frequencies" in data
        assert "modes" in data


def test_get_imaginary_modes_empty():
    all_modes = {
        "GM": {
            "frequencies": np.array([0.0, 5.0, 5.0, 10.0, 10.0, 10.0]),
            "modes": [
                {
                    "band_index": 0,
                    "frequency": 0.0,
                    "bcs_label": "T1u",
                    "mulliken_label": "T1u",
                },
                {
                    "band_index": 1,
                    "frequency": 5.0,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
                {
                    "band_index": 2,
                    "frequency": 5.0,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
                {
                    "band_index": 3,
                    "frequency": 10.0,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
                {
                    "band_index": 4,
                    "frequency": 10.0,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
                {
                    "band_index": 5,
                    "frequency": 10.0,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
            ],
        }
    }
    imag = get_imaginary_modes(all_modes)
    assert len(imag) == 0


def test_get_imaginary_modes_with_imaginary():
    all_modes = {
        "R": {
            "frequencies": np.array([1.0, 2.0, -3.5, -3.5, 5.0, 6.0]),
            "modes": [
                {
                    "band_index": 0,
                    "frequency": 1.0,
                    "bcs_label": "R1",
                    "mulliken_label": None,
                },
                {
                    "band_index": 1,
                    "frequency": 2.0,
                    "bcs_label": "R2",
                    "mulliken_label": None,
                },
                {
                    "band_index": 2,
                    "frequency": -3.5,
                    "bcs_label": "R3",
                    "mulliken_label": None,
                },
                {
                    "band_index": 3,
                    "frequency": -3.5,
                    "bcs_label": "R3",
                    "mulliken_label": None,
                },
                {
                    "band_index": 4,
                    "frequency": 5.0,
                    "bcs_label": "R4",
                    "mulliken_label": None,
                },
                {
                    "band_index": 5,
                    "frequency": 6.0,
                    "bcs_label": "R5",
                    "mulliken_label": None,
                },
            ],
        }
    }
    imag = get_imaginary_modes(all_modes)
    assert len(imag) == 2
    for m in imag:
        assert m["frequency"] < 0
        assert m["kpoint_label"] == "R"
        assert "band_index" in m
        assert "bcs_label" in m


def test_get_imaginary_modes_degeneracy():
    all_modes = {
        "X": {
            "frequencies": np.array([-2.0, -2.0, -2.0, 1.0, 3.0, 4.0]),
            "modes": [
                {
                    "band_index": 0,
                    "frequency": -2.0,
                    "bcs_label": "X1",
                    "mulliken_label": None,
                },
                {
                    "band_index": 1,
                    "frequency": -2.0,
                    "bcs_label": "X1",
                    "mulliken_label": None,
                },
                {
                    "band_index": 2,
                    "frequency": -2.0,
                    "bcs_label": "X1",
                    "mulliken_label": None,
                },
                {
                    "band_index": 3,
                    "frequency": 1.0,
                    "bcs_label": "X2",
                    "mulliken_label": None,
                },
                {
                    "band_index": 4,
                    "frequency": 3.0,
                    "bcs_label": "X3",
                    "mulliken_label": None,
                },
                {
                    "band_index": 5,
                    "frequency": 4.0,
                    "bcs_label": "X4",
                    "mulliken_label": None,
                },
            ],
        }
    }
    imag = get_imaginary_modes(all_modes)
    assert len(imag) == 3
    for m in imag:
        assert m["frequency"] == -2.0
        assert m["degeneracy"] == 3


def test_get_imaginary_modes_threshold():
    all_modes = {
        "X": {
            "frequencies": np.array([-1.0, 0.5, 1.0, 2.0]),
            "modes": [
                {
                    "band_index": 0,
                    "frequency": -1.0,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
                {
                    "band_index": 1,
                    "frequency": 0.5,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
                {
                    "band_index": 2,
                    "frequency": 1.0,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
                {
                    "band_index": 3,
                    "frequency": 2.0,
                    "bcs_label": None,
                    "mulliken_label": None,
                },
            ],
        }
    }
    imag = get_imaginary_modes(all_modes, threshold=0.0)
    assert len(imag) == 1
    assert imag[0]["frequency"] == -1.0

    imag2 = get_imaginary_modes(all_modes, threshold=0.6)
    assert len(imag2) == 2
    assert imag2[0]["frequency"] == -1.0
    assert imag2[1]["frequency"] == 0.5


def test_get_imaginary_modes_excludes_gamma_acoustic_modes():
    all_modes = {
        "GM": {
            "frequencies": np.array([-0.03, 0.01, -0.02, -1.2, 2.0, 3.0]),
            "modes": [
                {
                    "band_index": 0,
                    "frequency": -0.03,
                    "bcs_label": "GM-acoustic",
                    "mulliken_label": None,
                },
                {
                    "band_index": 1,
                    "frequency": 0.01,
                    "bcs_label": "GM-acoustic",
                    "mulliken_label": None,
                },
                {
                    "band_index": 2,
                    "frequency": -0.02,
                    "bcs_label": "GM-acoustic",
                    "mulliken_label": None,
                },
                {
                    "band_index": 3,
                    "frequency": -1.2,
                    "bcs_label": "GM4-",
                    "mulliken_label": "T1u",
                },
                {
                    "band_index": 4,
                    "frequency": 2.0,
                    "bcs_label": "GM5+",
                    "mulliken_label": None,
                },
                {
                    "band_index": 5,
                    "frequency": 3.0,
                    "bcs_label": "GM6+",
                    "mulliken_label": None,
                },
            ],
        }
    }

    imag = get_imaginary_modes(all_modes)

    assert [mode["band_index"] for mode in imag] == [3]
    assert imag[0]["bcs_label"] == "GM4-"
