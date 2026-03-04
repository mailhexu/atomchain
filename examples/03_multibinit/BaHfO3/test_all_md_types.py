"""
Test all MD thermostats and ensembles
======================================

Tests NVT thermostats: Andersen, Bussi
Tests NPT ensembles: NPTBerendsen, NPT

How to run:
    python test_all_md_types.py
"""

from ase.io import read
from atomchain.md import (
    md_nvt_andersen,
    md_nvt_bussi,
    md_npt_berendsen,
    md_npt,
)
import os

# Check if reference structure exists
if not os.path.exists('BaHfO3_ref.vasp'):
    print("ERROR: BaHfO3_ref.vasp not found. Run multibinit_relaxation.py first.")
    exit(1)

atoms = read('BaHfO3_ref.vasp')

print("=" * 70)
print("Test 1: NVT with Andersen thermostat")
print("=" * 70)

final_atoms, dyn = md_nvt_andersen(
    atoms=atoms,
    calc='multibinit',
    model_path='BaHfO3_config.conf',
    timestep=1.0,
    steps=50,
    temperature=300.0,
    andersen_prob=0.01,
    trajectory='test_andersen.traj',
    logfile='test_andersen.log',
    loginterval=10
)

print(f"Andersen completed. Final T: {final_atoms.get_kinetic_energy() / (1.5 * 8.617333262e-5 * len(atoms)):.2f} K")

print("\n" + "=" * 70)
print("Test 2: NVT with Bussi thermostat")
print("=" * 70)

final_atoms, dyn = md_nvt_bussi(
    atoms=atoms,
    calc='multibinit',
    model_path='BaHfO3_config.conf',
    timestep=1.0,
    steps=50,
    temperature=300.0,
    taut=100.0,
    trajectory='test_bussi.traj',
    logfile='test_bussi.log',
    loginterval=10
)

print(f"Bussi completed. Final T: {final_atoms.get_kinetic_energy() / (1.5 * 8.617333262e-5 * len(atoms)):.2f} K")

print("\n" + "=" * 70)
print("Test 3: NPT with Berendsen barostat")
print("=" * 70)

final_atoms, dyn = md_npt_berendsen(
    atoms=atoms,
    calc='multibinit',
    model_path='BaHfO3_config.conf',
    timestep=1.0,
    steps=50,
    temperature=300.0,
    pressure=1.01325,
    taut=100.0,
    taup=1000.0,
    compressibility=4.57e-5,
    trajectory='test_npt_berendsen.traj',
    logfile='test_npt_berendsen.log',
    loginterval=10
)

initial_vol = atoms.get_volume()
final_vol = final_atoms.get_volume()
print(f"NPT Berendsen completed. Volume: {initial_vol:.2f} -> {final_vol:.2f} Å³")
print(f"Final T: {final_atoms.get_kinetic_energy() / (1.5 * 8.617333262e-5 * len(atoms)):.2f} K")

print("\n" + "=" * 70)
print("Test 4: NPT with Nose-Hoover-like dynamics")
print("=" * 70)

final_atoms, dyn = md_npt(
    atoms=atoms,
    calc='multibinit',
    model_path='BaHfO3_config.conf',
    timestep=1.0,
    steps=50,
    temperature=300.0,
    pressure=1.01325,
    ttime=25.0,
    trajectory='test_npt.traj',
    logfile='test_npt.log',
    loginterval=10
)

final_vol = final_atoms.get_volume()
print(f"NPT completed. Volume: {initial_vol:.2f} -> {final_vol:.2f} Å³")
print(f"Final T: {final_atoms.get_kinetic_energy() / (1.5 * 8.617333262e-5 * len(atoms)):.2f} K")

print("\n" + "=" * 70)
print("All MD types tested successfully!")
print("=" * 70)
