"""
Test MD functions from atomchain.md module
===========================================

Tests the new MD wrapper functions with MULTIBINIT.

How to run:
    python test_md_functions.py
"""

from ase.io import read
from atomchain.md import md_nve_velocity_verlet, md_nvt_langevin
import os

# Check if reference structure exists
if not os.path.exists('BaHfO3_ref.vasp'):
    print("ERROR: BaHfO3_ref.vasp not found. Run multibinit_relaxation.py first.")
    exit(1)

atoms = read('BaHfO3_ref.vasp')

print("=" * 70)
print("Test 1: NVE MD with md_nve_velocity_verlet()")
print("=" * 70)

final_atoms_nve, dyn_nve = md_nve_velocity_verlet(
    atoms=atoms,
    calc='multibinit',
    model_path='BaHfO3_config.conf',
    timestep=1.0,
    steps=50,
    temperature=300.0,
    trajectory='test_nve.traj',
    logfile='test_nve.log',
    loginterval=10
)

print("\nNVE MD completed.")
print(f"Final energy: {final_atoms_nve.get_potential_energy():.6f} eV")
print(f"Final temperature: {final_atoms_nve.get_kinetic_energy() / (1.5 * 8.617333262e-5 * len(atoms)):.2f} K")

print("\n" + "=" * 70)
print("Test 2: NVT MD with md_nvt_langevin()")
print("=" * 70)

final_atoms_nvt, dyn_nvt = md_nvt_langevin(
    atoms=atoms,
    calc='multibinit',
    model_path='BaHfO3_config.conf',
    timestep=1.0,
    steps=50,
    temperature=300.0,
    friction=0.002,
    trajectory='test_nvt.traj',
    logfile='test_nvt.log',
    loginterval=10
)

print("\nNVT MD completed.")
print(f"Final energy: {final_atoms_nvt.get_potential_energy():.6f} eV")
print(f"Final temperature: {final_atoms_nvt.get_kinetic_energy() / (1.5 * 8.617333262e-5 * len(atoms)):.2f} K")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("Output files: test_nve.traj, test_nve.log, test_nvt.traj, test_nvt.log")
print("=" * 70)
