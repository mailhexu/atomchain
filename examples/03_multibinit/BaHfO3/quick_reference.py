"""
Quick Reference: MULTIBINIT with AtomChain
==========================================

This is a minimal working example for quick copy-paste usage.
For detailed explanations, see README.md and the full examples.
"""

# ============================================================================
# 1. BASIC SETUP
# ============================================================================

from ase.io import read, write
from atomchain.init_model import init_calc
from atomchain.relax import relax_with_ml
from atomchain.phonon import phonon_with_ml

# ============================================================================
# 2. INITIALIZE CALCULATOR
# ============================================================================

# Method 1: Full name
calc = init_calc(model_type="multibinit", model_path="config.conf")

# Method 2: Short alias
calc = init_calc(model_type="mb", model_path="config.conf")

# ============================================================================
# 3. STRUCTURE RELAXATION
# ============================================================================

# Load structure
atoms = read('structure.vasp')

# Relax
relaxed = relax_with_ml(
    atoms=atoms,
    calc=calc,
    fmax=0.01,
    relax_cell=True,
    traj_file='relax.traj'
)

# Save
write('relaxed.vasp', relaxed)

# ============================================================================
# 4. PHONON CALCULATION
# ============================================================================

# Load structure
atoms = read('relaxed.vasp')

# Method 1: Pass calculator object (if you already created it)
phonon_with_ml(
    atoms=atoms,
    calc=calc,
    supercell=[2, 2, 2],
    displacement=0.01,
    npoints=100,
    use_seek_path=True,
    save_folder='phonon_output'
)

# Method 2: Pass model_type and model_path directly (simpler!)
phonon_with_ml(
    atoms=atoms,
    calc="multibinit",           # or "mb"
    model_path="config.conf",    # Path to configuration file
    supercell=[2, 2, 2],
    displacement=0.01,
    npoints=100,
    use_seek_path=True,
    save_folder='phonon_output'
)

# ============================================================================
# 5. SINGLE-POINT CALCULATION
# ============================================================================

from ase import Atoms

atoms_list = read('structure.vasp')
atoms = atoms_list if isinstance(atoms_list, Atoms) else atoms_list[0]
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f"Energy: {energy:.6f} eV")
print(f"Max force: {(forces**2).sum(axis=1).max()**0.5:.6f} eV/Å")

# ============================================================================
# 6. MINIMAL CONFIGURATION FILE
# ============================================================================

"""
# config.conf
[files]
ddb_file = system_DDB
sys_file = system.xml

[parameters]
ncell = 2 2 2
ngqpt = 4 4 4
dipdip = 1

[backend]
use_atomic_units = false
auto_match_atoms = true
"""

# ============================================================================
# 7. ERROR HANDLING
# ============================================================================

try:
    calc = init_calc("multibinit", model_path="config.conf")
except ImportError:
    print("pymultibinit not installed")
except FileNotFoundError:
    print("Config file not found")
except ValueError:
    print("Config file invalid")

# ============================================================================
# 8. BATCH PROCESSING
# ============================================================================

from pathlib import Path

# Process multiple structures
for structure_file in Path('.').glob('*.vasp'):
    atoms = read(structure_file)
    calc = init_calc("mb", model_path="config.conf")
    relaxed = relax_with_ml(atoms, calc=calc, fmax=0.01)
    write(f'relaxed_{structure_file.stem}.vasp', relaxed)

# ============================================================================
# END QUICK REFERENCE
# ============================================================================
