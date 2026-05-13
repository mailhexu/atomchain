"""ABINIT HIST.nc interoperability.

This module implements the supported atomchain subset of ABINIT HIST NetCDF
files: structures, total energies, Cartesian forces, reduced forces, and stress.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols
from ase.io import read, write

from atomchain.ddb.conventions import (
    BOHR_PER_ANGSTROM,
    EV_PER_HARTREE,
    HARTREE_PER_EV,
    angstrom_to_bohr,
    bohr_to_angstrom,
)

HIST_SCHEMA = {
    "typat": {"shape": ("natom",), "units": "one-based species index"},
    "znucl": {"shape": ("ntypat",), "units": "atomic number"},
    "rprimd": {"shape": ("time", 3, 3), "units": "Bohr"},
    "xred": {"shape": ("time", "natom", 3), "units": "fractional"},
    "xcart": {"shape": ("time", "natom", 3), "units": "Bohr"},
    "etotal": {"shape": ("time",), "units": "Hartree"},
    "ekin": {"shape": ("time",), "units": "Hartree"},
    "entropy": {"shape": ("time",), "units": "Hartree"},
    "fcart": {"shape": ("time", "natom", 3), "units": "Hartree/Bohr"},
    "fred": {"shape": ("time", "natom", 3), "units": "Hartree"},
    "strten": {"shape": ("time", 6), "units": "Hartree/Bohr^3"},
}


def energy_ev_to_ha(value):
    return np.asarray(value, dtype=float) * HARTREE_PER_EV


def energy_ha_to_ev(value):
    return np.asarray(value, dtype=float) * EV_PER_HARTREE


def force_ev_ang_to_ha_bohr(value):
    return np.asarray(value, dtype=float) * HARTREE_PER_EV / BOHR_PER_ANGSTROM


def force_ha_bohr_to_ev_ang(value):
    return np.asarray(value, dtype=float) * EV_PER_HARTREE * BOHR_PER_ANGSTROM


def stress_ev_ang3_to_ha_bohr3(value):
    return np.asarray(value, dtype=float) * HARTREE_PER_EV / (BOHR_PER_ANGSTROM**3)


def stress_ha_bohr3_to_ev_ang3(value):
    return np.asarray(value, dtype=float) * EV_PER_HARTREE * (BOHR_PER_ANGSTROM**3)


def read_abinit_hist(filename):
    """Read a supported ABINIT HIST.nc file into ASE Atoms frames."""
    from scipy.io import netcdf_file

    with netcdf_file(str(filename), "r", mmap=False) as nc:
        typat = np.array(nc.variables["typat"].data, dtype=int)
        znucl = np.array(nc.variables["znucl"].data, dtype=int)
        rprimd = np.array(nc.variables["rprimd"].data, dtype=float)
        xred = np.array(nc.variables["xred"].data, dtype=float)
        etotal = _optional_var(nc, "etotal")
        fcart = _optional_var(nc, "fcart")
        strten = _optional_var(nc, "strten")

    symbols = [chemical_symbols[int(znucl[i - 1])] for i in typat]
    frames = []
    for iframe in range(rprimd.shape[0]):
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=xred[iframe],
            cell=bohr_to_angstrom(rprimd[iframe]),
            pbc=True,
        )
        results = {}
        if etotal is not None:
            results["energy"] = float(energy_ha_to_ev(etotal[iframe]))
        if fcart is not None:
            results["forces"] = force_ha_bohr_to_ev_ang(fcart[iframe])
        if strten is not None:
            results["stress"] = stress_ha_bohr3_to_ev_ang3(strten[iframe])
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)
        frames.append(atoms)
    return frames


def write_abinit_hist(frames, filename, metadata=None, strict=True):
    """Write ASE Atoms frames to a supported ABINIT HIST.nc file."""
    from scipy.io import netcdf_file

    frames = _load_frames(frames)
    if not frames:
        raise ValueError("Cannot write HIST with no frames")
    _validate_compatible_frames(frames)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    symbols = frames[0].get_chemical_symbols()
    unique_symbols = []
    typat = []
    for symbol in symbols:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
        typat.append(unique_symbols.index(symbol) + 1)
    znucl = [chemical_symbols.index(symbol) for symbol in unique_symbols]

    ntime = len(frames)
    natom = len(frames[0])
    ntypat = len(unique_symbols)

    rprimd = np.array(
        [angstrom_to_bohr(frame.cell.array) for frame in frames], dtype=float
    )
    xred = np.array(
        [frame.get_scaled_positions(wrap=False) for frame in frames], dtype=float
    )
    xcart = np.array(
        [angstrom_to_bohr(frame.get_positions()) for frame in frames], dtype=float
    )
    etotal = []
    fcart = []
    strten = []
    fred = []
    for iframe, frame in enumerate(frames):
        energy = _get_frame_property(
            frame, iframe, "energy", strict, lambda: frame.get_potential_energy(), 0.0
        )
        forces = _get_frame_property(
            frame,
            iframe,
            "forces",
            strict,
            lambda: frame.get_forces(),
            np.zeros((natom, 3), dtype=float),
        )
        stress = _get_frame_property(
            frame,
            iframe,
            "stress",
            strict,
            lambda: frame.get_stress(voigt=True),
            np.zeros(6, dtype=float),
        )
        etotal.append(float(energy_ev_to_ha(energy)))
        fcart_ha = force_ev_ang_to_ha_bohr(forces)
        fcart.append(fcart_ha)
        strten.append(stress_ev_ang3_to_ha_bohr3(stress))
        fred.append(fcart_ha @ rprimd[iframe].T)

    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("time", ntime)
        nc.createDimension("natom", natom)
        nc.createDimension("ntypat", ntypat)
        nc.createDimension("npsp", ntypat)
        nc.createDimension("three", 3)
        nc.createDimension("six", 6)
        _write_var(nc, "typat", "i", ("natom",), np.array(typat, dtype=np.int32))
        _write_var(nc, "znucl", "d", ("ntypat",), np.array(znucl, dtype=float))
        _write_var(nc, "rprimd", "d", ("time", "three", "three"), rprimd)
        _write_var(nc, "xred", "d", ("time", "natom", "three"), xred)
        _write_var(nc, "xcart", "d", ("time", "natom", "three"), xcart)
        _write_var(nc, "etotal", "d", ("time",), np.array(etotal, dtype=float))
        _write_var(nc, "ekin", "d", ("time",), np.zeros(ntime, dtype=float))
        _write_var(nc, "entropy", "d", ("time",), np.zeros(ntime, dtype=float))
        _write_var(
            nc, "fcart", "d", ("time", "natom", "three"), np.array(fcart, dtype=float)
        )
        _write_var(
            nc, "fred", "d", ("time", "natom", "three"), np.array(fred, dtype=float)
        )
        _write_var(nc, "strten", "d", ("time", "six"), np.array(strten, dtype=float))

    if metadata is not None:
        write_hist_metadata(frames, metadata, hist_filename=str(path))
    return path


def hist_to_traj(hist_file, traj_file):
    frames = read_abinit_hist(hist_file)
    write(str(traj_file), frames)
    return Path(traj_file)


def traj_to_hist(traj_file, hist_file, metadata=None, strict=True):
    return write_abinit_hist(
        read(str(traj_file), ":"), hist_file, metadata=metadata, strict=strict
    )


def write_hist_metadata(frames, filename, hist_filename=None):
    data = {
        "format": "atomchain-hist-sidecar-v1",
        "hist_file": hist_filename,
        "nframes": len(frames),
        "provenance": [dict(frame.info) for frame in frames],
    }
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)
    return path


def _load_frames(frames):
    if isinstance(frames, (str, Path)):
        loaded = read(str(frames), ":")
        if isinstance(loaded, Atoms):
            return [loaded]
        return list(loaded)
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def _validate_compatible_frames(frames):
    symbols = frames[0].get_chemical_symbols()
    natom = len(frames[0])
    for iframe, frame in enumerate(frames):
        if len(frame) != natom:
            raise ValueError(f"Frame {iframe} has different atom count")
        if frame.get_chemical_symbols() != symbols:
            raise ValueError(f"Frame {iframe} has different atom ordering or symbols")


def _get_frame_property(frame, iframe, name, strict, getter, default):
    try:
        return getter()
    except Exception as exc:
        if strict:
            raise ValueError(f"Frame {iframe} is missing {name}") from exc
        return default


def _optional_var(nc, name):
    if name not in nc.variables:
        return None
    return np.array(nc.variables[name].data, dtype=float)


def _write_var(nc, name, dtype, dimensions, data):
    var = nc.createVariable(name, dtype, dimensions)
    var[:] = data
    return var
