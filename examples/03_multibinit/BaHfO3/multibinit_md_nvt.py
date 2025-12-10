"""
NVT Molecular Dynamics with MULTIBINIT
=======================================

Runs NVT (canonical ensemble) MD simulation on BaHfO3 using MULTIBINIT potential.
Uses Langevin thermostat to maintain constant temperature.

How to run:
    python multibinit_md_nvt.py

Output:
    - BaHfO3_md_nvt.traj: Trajectory file
    - BaHfO3_md_nvt.log: MD log with energy and temperature
"""

from ase.io import read
from atomchain.md import md_nvt_langevin
import os

# Load initial structure
if not os.path.exists('BaHfO3_ref.vasp'):
    print("ERROR: BaHfO3_ref.vasp not found. Run multibinit_relaxation.py first.")
    exit(1)

atoms = read('BaHfO3_ref.vasp')

# Run NVT MD simulation with Langevin thermostat
temperature = 300  # Target temperature in K
final_atoms, dyn = md_nvt_langevin(
    atoms=atoms,
    calc='multibinit',
    model_path='BaHfO3_config.conf',
    timestep=1.0,              # fs
    steps=100,
    temperature=temperature,   # K
    friction=0.002,            # 1/fs
    trajectory='BaHfO3_md_nvt.traj',
    logfile='BaHfO3_md_nvt.log',
    loginterval=1
)

# Print final state
print("\nNVT simulation complete.")
print(f"Target temperature: {temperature} K")
print(f"Final energy: {final_atoms.get_potential_energy():.6f} eV")
print(f"Final temperature: {final_atoms.get_kinetic_energy() / (1.5 * 8.617333262e-5 * len(atoms)):.2f} K")
print("\nOutput files:")
print("  - BaHfO3_md_nvt.traj: Trajectory")
print("  - BaHfO3_md_nvt.log: Energy/temperature log")
print(f"\nTemperature should fluctuate around {temperature} K (check log file).")
