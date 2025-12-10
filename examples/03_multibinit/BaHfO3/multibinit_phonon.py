"""
Minimal MULTIBINIT Phonon Calculation Example
=============================================

Calculates phonon band structure for BaHfO3 using MULTIBINIT effective potential.
"""

from ase.io import read
from atomchain.phonon import phonon_with_ml
import os

# Load structure
# This assumes BaHfO3_ref.vasp exists (e.g. from relaxation)
# If not, please provide a valid structure file
if not os.path.exists('BaHfO3_ref.vasp'):
    raise FileNotFoundError("Please run multibinit_relaxation.py first or provide 'BaHfO3_ref.vasp'")

atoms = read('BaHfO3_ref.vasp')

# Calculate phonons
# ndim must match the ncell expected by MULTIBINIT config
# For 5-atom unit cell and ncell=2 2 2 in config, we use ndim=2 2 2 
# (Note: This is a simplification; for exact match with internal supercell one might need adjustments)
phonon_with_ml(
    atoms=atoms,
    calc="multibinit",              
    model_path="BaHfO3_config.conf", 
    supercell=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], # 2x2x2 supercell
    figname="phonon.png",
    relax=False
)

print("Phonon calculation complete. Output saved to 'phonon_save/' and 'phonon.png'.")
