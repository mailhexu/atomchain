"""
Debug script to check if phonon calculation triggers relaxation.

Purpose:
    Trace through the phonon_with_ml call to see where FIRE relaxation output comes from.

How to run:
    python debug_phonon_relax.py

What it tests:
    - Whether relax=False is respected
    - If get_forces() on displaced structures triggers relaxation
    - Where the FIRE output with 3325 steps comes from
"""

from ase.io import read
from atomchain.init_model import init_calc
from atomchain.phonon import phonon_with_ml
import numpy as np

# Load reference structure (5 atoms)
atoms_list = read('BaHfO3_ref.vasp')
from ase import Atoms
atoms = atoms_list if isinstance(atoms_list, Atoms) else atoms_list[0]
print(f"Loaded structure: {len(atoms)} atoms")
print(f"Formula: {atoms.get_chemical_formula()}")

# Initialize calculator
calc = init_calc(model_type="multibinit", model_path="BaHfO3_config.conf")
print("\nCalculator initialized")

# Test 1: Single force calculation (should NOT relax)
print("\n" + "="*70)
print("TEST 1: Single get_forces() call")
print("="*70)
atoms.calc = calc
print("Calling get_forces() - this should NOT show FIRE output...")
forces = atoms.get_forces()
print(f"Forces shape: {forces.shape}")
print(f"Max force: {np.abs(forces).max():.6f} eV/Å")
print("✓ No FIRE output - single point calculation works correctly")

# Test 2: Phonon with relax=False (should NOT relax)
print("\n" + "="*70)
print("TEST 2: phonon_with_ml with relax=False")
print("="*70)
print("This should NOT show FIRE output...")

phonon_with_ml(
    atoms=atoms,
    calc="multibinit",
    model_path="BaHfO3_config.conf",
    supercell=np.eye(3),  # 1x1x1 - just 5 atoms
    displacement=0.05,
    npoints=50,
    relax=False,  # EXPLICITLY False
    plot=False,  # Skip plotting
    save_folder='debug_phonon_norelax',
)

print("\n✓ Phonon calculation completed")
print("If you see FIRE output above, that's the bug!")

# Test 3: Phonon with relax=True (SHOULD relax)
print("\n" + "="*70)
print("TEST 3: phonon_with_ml with relax=True")
print("="*70)
print("This SHOULD show FIRE output before phonon calculation...")

# Create a slightly distorted structure to see relaxation
atoms_distorted = atoms.copy()
pos = atoms_distorted.positions.copy()
pos[0] += [0.1, 0.1, 0.1]  # Displace Ba atom
atoms_distorted.positions = pos

phonon_with_ml(
    atoms=atoms_distorted,
    calc="multibinit",
    model_path="BaHfO3_config.conf",
    supercell=np.eye(3),
    displacement=0.05,
    npoints=50,
    relax=True,  # EXPLICITLY True
    plot=False,
    save_folder='debug_phonon_withrelax',
)

print("\n✓ All tests completed")
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("1. Single get_forces(): No FIRE output (correct)")
print("2. phonon_with_ml(relax=False): Check if FIRE appeared (bug if yes)")
print("3. phonon_with_ml(relax=True): FIRE should appear (expected)")
