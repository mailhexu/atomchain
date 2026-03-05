"""
Test supercell size validation in MultibinitPotential.

This test verifies that the potential correctly detects and reports
mismatches between the expected supercell size and the input structure.
"""
from ase.io import read
import sys
sys.path.insert(0, '/Users/hexu/projects/abinit_git/pymultibinit_dev/pymultibinit/src')

from pymultibinit import MultibinitPotential

# Initialize potential
print("Initializing potential from config...")
pot = MultibinitPotential.from_config_file('./BaHfO3_config.conf')
print(f"Expected natoms: {pot.expected_natoms}")

# Read unit cell
atoms = read('./BaHfO3_ref.vasp')
print(f"Unit cell: {len(atoms)} atoms\n")

# Test 1: Correct size (should work)
print("="*60)
print("Test 1: Unit cell (5 atoms) - should succeed")
print("="*60)
try:
    energy, forces, stress = pot.evaluate(atoms.positions, atoms.cell.array)
    print(f"✓ SUCCESS: E={energy:.6f} eV")
    print(f"  Max force: {abs(forces).max():.6e} eV/Å")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: Wrong size (should fail with clear message)
print("\n" + "="*60)
print("Test 2: 2x2x2 supercell (40 atoms) - should fail")
print("="*60)
supercell = atoms * (2, 2, 2)
try:
    energy, forces, stress = pot.evaluate(supercell.positions, supercell.cell.array)
    print(f"✗ FAILED: Should have raised ValueError")
    print(f"  Got energy: {energy:.6f} eV (incorrect)")
except ValueError as e:
    print(f"✓ SUCCESS: Got expected ValueError")
    print(f"  Message: {e}")
except Exception as e:
    print(f"✗ FAILED: Unexpected error type {type(e).__name__}")
    print(f"  Message: {e}")

print("\n" + "="*60)
print("Validation test complete")
print("="*60)
