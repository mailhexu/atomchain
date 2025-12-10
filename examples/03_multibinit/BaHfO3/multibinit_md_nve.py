"""
NVE Molecular Dynamics with MULTIBINIT
=======================================

Runs NVE (microcanonical ensemble) MD simulation on BaHfO3 using MULTIBINIT potential.

How to run:
    python multibinit_md_nve.py

Output:
    - BaHfO3_md_nve.traj: Trajectory file
    - BaHfO3_md_nve.log: MD log with energy and temperature
"""

from ase.io import read
from atomchain.md import md_nve_velocity_verlet
import os

# Load initial structure
if not os.path.exists('BaHfO3_ref.vasp'):
    print("ERROR: BaHfO3_ref.vasp not found. Run multibinit_relaxation.py first.")
    exit(1)

atoms = read('BaHfO3_ref.vasp')

# Run NVE MD simulation
final_atoms, dyn = md_nve_velocity_verlet(
    atoms=atoms,
    calc='multibinit',
    model_path='BaHfO3_config.conf',
    timestep=1.0,              # fs
    steps=100,
    temperature=300.0,         # K (initial)
    trajectory='BaHfO3_md_nve.traj',
    logfile='BaHfO3_md_nve.log',
    loginterval=1
)

# Print final state
print("\nNVE simulation complete.")
print(f"Final energy: {final_atoms.get_potential_energy():.6f} eV")
print(f"Final temperature: {final_atoms.get_kinetic_energy() / (1.5 * 8.617333262e-5 * len(atoms)):.2f} K")
print("\nOutput files:")
print("  - BaHfO3_md_nve.traj: Trajectory")
print("  - BaHfO3_md_nve.log: Energy/temperature log")
print("\nTotal energy should be conserved (check log file).")
