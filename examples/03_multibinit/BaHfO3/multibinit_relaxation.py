"""
Minimal MULTIBINIT Relaxation Example
=====================================

Relaxes a BaHfO3 structure using MULTIBINIT effective potential.
"""

from ase.io import read
from atomchain.relax import relax_with_ml
import os

# Load initial structure (create one if not exists or use provided file)
# Ideally, users should have an initial structure. 
# For this example, we check for a file or fail gracefully.
if not os.path.exists('BaHfO3_initial.vasp'):
    # Create a simple initial structure if file doesn't exist
    from ase import Atoms
    a = 4.00
    atoms = Atoms(
        symbols=['Ba', 'Hf', 'O', 'O', 'O'],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.52, 0.52, 0.52], # Slightly distorted
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True
    )
else:
    atoms = read('BaHfO3_initial.vasp')

# Relax structure
# Just specify the calculator type and path to config file
relaxed_atoms = relax_with_ml(
    atoms=atoms,
    calc="multibinit",
    model_path="BaHfO3_config.conf",
    fmax=0.01,
    relax_cell=True,
    traj_file="BaHfO3_relax.traj"
)

# Save result
relaxed_atoms.write('BaHfO3_ref.vasp')
print("Relaxation complete. Structure saved to 'BaHfO3_ref.vasp'.")
