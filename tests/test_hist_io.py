import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from atomchain.io.hist import (
    HIST_SCHEMA,
    energy_ev_to_ha,
    energy_ha_to_ev,
    force_ev_ang_to_ha_bohr,
    force_ha_bohr_to_ev_ang,
    hist_to_traj,
    read_abinit_hist,
    stress_ev_ang3_to_ha_bohr3,
    stress_ha_bohr3_to_ev_ang3,
    traj_to_hist,
    write_abinit_hist,
)


def frame(a=4.05, energy=-1.25):
    atoms = bulk("Al", "fcc", a=a, cubic=True)
    forces = np.arange(len(atoms) * 3, dtype=float).reshape(len(atoms), 3) * 0.01
    stress = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6]) * 1e-3
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=forces, stress=stress
    )
    atoms.info["source"] = "test"
    return atoms


def assert_frames_close(a, b):
    assert a.get_chemical_symbols() == b.get_chemical_symbols()
    np.testing.assert_allclose(a.cell.array, b.cell.array, atol=1e-12)
    np.testing.assert_allclose(
        a.get_scaled_positions(), b.get_scaled_positions(), atol=1e-12
    )
    np.testing.assert_allclose(
        a.get_potential_energy(), b.get_potential_energy(), atol=1e-12
    )
    np.testing.assert_allclose(a.get_forces(), b.get_forces(), atol=1e-12)
    np.testing.assert_allclose(
        a.get_stress(voigt=True), b.get_stress(voigt=True), atol=1e-12
    )


def test_hist_schema_lists_required_variables():
    for name in [
        "typat",
        "znucl",
        "rprimd",
        "xred",
        "etotal",
        "fcart",
        "fred",
        "strten",
    ]:
        assert name in HIST_SCHEMA


def test_hist_unit_conversions_round_trip():
    energy = np.array([-1.0, 2.0])
    forces = np.array([[1.0, -2.0, 3.0]])
    stress = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(energy_ha_to_ev(energy_ev_to_ha(energy)), energy)
    np.testing.assert_allclose(
        force_ha_bohr_to_ev_ang(force_ev_ang_to_ha_bohr(forces)), forces
    )
    np.testing.assert_allclose(
        stress_ha_bohr3_to_ev_ang3(stress_ev_ang3_to_ha_bohr3(stress)), stress
    )


def test_hist_read_write_round_trip(tmp_path):
    frames = [frame(4.05, -1.0), frame(4.10, -0.9)]
    path = tmp_path / "test_HIST.nc"
    write_abinit_hist(frames, path, metadata=tmp_path / "test_HIST.yaml")
    loaded = read_abinit_hist(path)
    assert len(loaded) == 2
    for expected, actual in zip(frames, loaded):
        assert_frames_close(expected, actual)
    assert (tmp_path / "test_HIST.yaml").exists()


def test_hist_writer_emits_abipy_energy_companion_terms(tmp_path):
    from scipy.io import netcdf_file

    path = tmp_path / "energy_terms_HIST.nc"
    write_abinit_hist([frame()], path)
    with netcdf_file(str(path), "r", mmap=False) as nc:
        assert "etotal" in nc.variables
        assert "ekin" in nc.variables
        assert "entropy" in nc.variables


def test_hist_round_trip_preserves_unwrapped_scaled_positions(tmp_path):
    atoms = frame()
    atoms.set_scaled_positions(
        atoms.get_scaled_positions(wrap=False) + [1.1, -0.2, 0.0]
    )
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.25, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6)
    )
    path = tmp_path / "unwrapped_HIST.nc"
    write_abinit_hist([atoms], path)
    loaded = read_abinit_hist(path)[0]
    np.testing.assert_allclose(
        loaded.get_scaled_positions(wrap=False),
        atoms.get_scaled_positions(wrap=False),
        atol=1e-12,
    )


def test_hist_traj_conversion_round_trip(tmp_path):
    traj = tmp_path / "input.traj"
    hist = tmp_path / "out_HIST.nc"
    out = tmp_path / "out.traj"
    frames = [frame()]
    write(str(traj), frames)
    traj_to_hist(traj, hist)
    hist_to_traj(hist, out)
    loaded = read(str(out), ":")
    assert_frames_close(frames[0], loaded[0])


def test_hist_writer_rejects_missing_results(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    with pytest.raises(ValueError, match="missing energy"):
        write_abinit_hist([atoms], tmp_path / "bad_HIST.nc")


def test_hist_strict_false_preserves_energy_forces_when_stress_missing(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    forces = np.ones((len(atoms), 3))
    atoms.calc = SinglePointCalculator(atoms, energy=-2.0, forces=forces)
    path = tmp_path / "partial_HIST.nc"
    write_abinit_hist([atoms], path, strict=False)
    loaded = read_abinit_hist(path)[0]
    assert loaded.get_potential_energy() == -2.0
    np.testing.assert_allclose(loaded.get_forces(), forces)
    np.testing.assert_allclose(loaded.get_stress(voigt=True), np.zeros(6))
