"""Tests for per-frame fit-weight specs in the MULTIBINIT workflow."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from atomchain import multibinit_workflow as mw
from atomchain.io.hist import write_abinit_hist

HA_BOHR_EV_ANG = 51.422067


def _hist_with_forces(tmp_path, force_lists):
    """Write a HIST file whose frames carry the given forces (Ha/Bohr rows)."""
    atoms = Atoms(
        "H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], cell=np.eye(3) * 4.0
    )
    frames = []
    for forces in force_lists:
        frame = atoms.copy()
        frame.calc = SinglePointCalculator(
            frame,
            energy=0.0,
            forces=np.asarray(forces, dtype=float),
            stress=np.zeros(6),
        )
        frames.append(frame)
    hist = tmp_path / "train_HIST.nc"
    write_abinit_hist(frames, hist)
    return hist


def test_force_rms_boltzmann_weights_follow_reference_forces(tmp_path):
    hist = _hist_with_forces(
        tmp_path,
        [
            np.zeros((2, 3)),  # F_rms = 0 -> weight 1
            np.full((2, 3), 0.1),  # eV/Ang; writer converts to Ha/Bohr
            np.full((2, 3), 0.3),  # eV/Ang
        ],
    )
    weights = mw._frame_weights_from_spec(
        {"mode": "force_rms_boltzmann", "sigma_ev_ang": 0.1}, hist
    )
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(np.exp(-0.5))
    assert weights[2] == pytest.approx(np.exp(-4.5))


def test_force_rms_weights_respect_floor(tmp_path):
    hist = _hist_with_forces(tmp_path, [np.full((2, 3), 1.0)])
    weights = mw._frame_weights_from_spec({"sigma_ev_ang": 0.05, "floor": 0.02}, hist)
    assert weights == [pytest.approx(0.02)]


def test_force_rms_weights_reject_bad_specs(tmp_path):
    hist = _hist_with_forces(tmp_path, [np.zeros((2, 3))])
    with pytest.raises(ValueError, match="sigma_ev_ang"):
        mw._frame_weights_from_spec({"sigma_ev_ang": 0.0}, hist)
    with pytest.raises(ValueError, match="mode"):
        mw._frame_weights_from_spec({"mode": "energy", "sigma_ev_ang": 0.1}, hist)


def test_python_fit_kwargs_expands_spec_to_weight_list(tmp_path):
    hist = _hist_with_forces(tmp_path, [np.zeros((2, 3)), np.zeros((2, 3))])
    options = {
        "weights": {"mode": "force_rms_boltzmann", "sigma_ev_ang": 0.1},
        "selection": "screened_greedy",
    }
    context = mw.WorkflowContext(
        mw.WorkflowConfig(
            structure=Atoms(
                "H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=np.eye(3) * 4.0
            ),
            output_dir=tmp_path,
            training=mw.TrainingStageConfig(options=options),
        ),
        Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=np.eye(3) * 4.0),
    )
    kwargs = mw._python_fit_kwargs(context, train_hist=hist)
    assert kwargs["weights"] == pytest.approx([1.0, 1.0])
    assert "selection" not in kwargs
    with pytest.raises(ValueError, match="train HIST"):
        mw._python_fit_kwargs(context)
