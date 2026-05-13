import numpy as np
import pytest
from ase.build import bulk

from atomchain.kpoints import get_high_symmetry_kpoints, get_kpoint_star


@pytest.fixture
def catio3_cubic():
    from ase import Atoms

    a = 3.8
    atoms = Atoms(
        symbols=["Ca", "Ti", "O"],
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0],
        ],
        cell=np.eye(3) * a,
        pbc=True,
    )
    return atoms


@pytest.fixture
def al_fcc():
    return bulk("Al", "fcc", a=4.05)


def test_catio3_has_expected_kpoints(catio3_cubic):
    kpts = get_high_symmetry_kpoints(catio3_cubic)
    labels = [k["label"] for k in kpts]
    assert "GM" in labels
    assert len(kpts) >= 4
    for k in kpts:
        assert len(k["qpoint"]) == 3


def test_gamma_is_present(al_fcc):
    kpts = get_high_symmetry_kpoints(al_fcc)
    labels = [k["label"] for k in kpts]
    assert "GM" in labels
    gm = [k for k in kpts if k["label"] == "GM"][0]
    np.testing.assert_allclose(gm["qpoint"], [0, 0, 0], atol=1e-5)


def test_kpoint_star_x_point(catio3_cubic):
    star = get_kpoint_star([0, 0.5, 0], catio3_cubic)
    assert star.shape[1] == 3
    assert star.shape[0] >= 2
    assert np.all(np.abs(star) <= 0.5 + 1e-6)


def test_kpoint_star_gamma(catio3_cubic):
    star = get_kpoint_star([0, 0, 0], catio3_cubic)
    assert star.shape == (1, 3)
    np.testing.assert_allclose(star[0], [0, 0, 0], atol=1e-5)
