# MULTIBINIT Tutorial

Complete guide for using MULTIBINIT effective potentials with AtomChain.

## Overview

MULTIBINIT provides effective potentials trained from DFT that enable fast calculations while maintaining accuracy. Use it for:
- Structure relaxation
- Phonon calculations  
- Molecular dynamics
- Large-scale simulations

## Setup

### 1. Install PyMultibinit and atomchain

```bash
pip install pymultibinit
pip install atomchain
```

### 2. Set Library Path

```bash
# Option A: LIBABINIT_PATH (recommended)
export LIBABINIT_PATH=/path/to/abinit/build/src/98_main/libabinit.dylib  # macOS
export LIBABINIT_PATH=/path/to/abinit/build/src/98_main/libabinit.so     # Linux

# Option B: LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/abinit/build/src/98_main:$LD_LIBRARY_PATH

# Add to ~/.bashrc or ~/.zshrc for persistence
```

### 3. Prepare Configuration File

Create `multibinit.conf`:

```ini
# Input files
ddb_file: BaHfO3_DDB
sys_file: BaHfO3.xml

# Supercell size (MUST match your structure!)
ncell: 2 2 2

# Optional parameters
ngqpt: 4 4 4      # Q-point grid
dipdip: 1         # Dipole-dipole interactions
auto_match_atoms: true
match_tolerance: 0.1
```

**Important**: `ncell` must match your input structure size. If `ncell: 2 2 2`, use a 2×2×2 supercell.

## Usage

### Structure Relaxation

```bash
# Basic relaxation
mlrelax structure.cif \
  --model multibinit \
  --model_path multibinit.conf \
  --output relaxed.cif

# With options
mlrelax structure.vasp \
  --model multibinit \
  --model_path multibinit.conf \
  --fmax 0.01 \
  --output POSCAR_relaxed
```

### Phonon Calculation

```bash
# Calculate phonon band structure
mlphonon structure.cif \
  --model multibinit \
  --model_path multibinit.conf \
  --ndim 2 2 2 \
  --figname phonon.pdf

# The output includes:
# - phonon.pdf: Band structure plot
# - phonopy.yaml: Phonopy format data
# - FORCE_SETS: Force constants
```

### Molecular Dynamics

```python
from atomchain.md import md_nvt_langevin
from atomchain.init_model import init_calc
from ase.io import read

# Load structure
atoms = read('structure.cif')

# Initialize calculator
calc = init_calc('multibinit', 'multibinit.conf')

# Run NVT MD
md_nvt_langevin(
    atoms=atoms,
    calc=calc,
    model_path='multibinit.conf',
    temperature=300,      # K
    timestep=1.0,         # fs
    steps=10000,
    friction=0.01,
    trajectory='md.traj',
    logfile='md.log'
)
```

See [md.md](md.md) for more MD options (NVE, NPT, different thermostats).

## Complete Workflow Example

### 1. Prepare Structure

```bash
# Create unit cell (5 atoms)
cat > unit_cell.cif << 'EOF'
# BaHfO3 unit cell
data_BaHfO3
_cell_length_a    4.15
_cell_length_b    4.15
_cell_length_c    4.15
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ba 0.0 0.0 0.0
Hf 0.5 0.5 0.5
O  0.5 0.0 0.5
O  0.0 0.5 0.5
O  0.5 0.5 0.0
EOF

# Create 2×2×2 supercell (40 atoms)
mlsupercell unit_cell.cif --size 2 --output supercell_222.cif
```

### 2. Create Configuration

```bash
cat > multibinit.conf << 'EOF'
ddb_file: BaHfO3_DDB
sys_file: BaHfO3.xml
ncell: 2 2 2       # Match supercell size!
ngqpt: 4 4 4
dipdip: 1
EOF
```

### 3. Relax Structure

```bash
mlrelax supercell_222.cif \
  --model multibinit \
  --model_path multibinit.conf \
  --fmax 0.01 \
  --output relaxed.cif
```

### 4. Calculate Phonons

```bash
mlphonon relaxed.cif \
  --model multibinit \
  --model_path multibinit.conf \
  --ndim 2 2 2 \
  --figname phonon_bands.pdf
```

### 5. Run MD Simulation

```python
from atomchain.md import md_nvt_langevin
from atomchain.init_model import init_calc
from ase.io import read

atoms = read('relaxed.cif')
calc = init_calc('multibinit', 'multibinit.conf')

md_nvt_langevin(
    atoms=atoms,
    calc=calc,
    model_path='multibinit.conf',
    temperature=300,
    timestep=1.0,
    steps=10000,
    friction=0.01,
    trajectory='md.traj',
    logfile='md.log'
)
```

## Python API

### Single Point Calculation

```python
from pymultibinit import MultibinitCalculator
from ase.io import read

# Load structure
atoms = read('structure.cif')

# Create calculator
calc = MultibinitCalculator.from_config_file('multibinit.conf')
atoms.calc = calc

# Calculate properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
stress = atoms.get_stress()            # eV/Angstrom^3

print(f"Energy: {energy} eV")
```

### Structure Optimization

```python
from pymultibinit import MultibinitCalculator
from ase.io import read, write
from ase.optimize import BFGS

atoms = read('structure.cif')
calc = MultibinitCalculator.from_config_file('multibinit.conf')
atoms.calc = calc

# Optimize
opt = BFGS(atoms, trajectory='opt.traj')
opt.run(fmax=0.01)

# Save result
write('optimized.cif', atoms)
```

### Export Reference Structure

```python
from pymultibinit import MultibinitPotential
import numpy as np

# Initialize potential
pot = MultibinitPotential.from_config_file('multibinit.conf')

# Do one evaluation
positions = np.zeros((40, 3))
lattice = np.eye(3) * 8.0
pot.evaluate(positions, lattice)

# Export MULTIBINIT's internal reference structure
pot.export_supercell_to_file('reference.cif')

# Or get ASE Atoms object
atoms = pot.export_supercell_to_ase()
atoms.set_chemical_symbols(['Ba']*8 + ['Hf']*8 + ['O']*24)
atoms.write('reference_with_symbols.cif')
```

## Troubleshooting

### Library Not Found

```bash
# Error: "Could not find libabinit"
# Solution: Set library path
export LIBABINIT_PATH=/path/to/libabinit.dylib
```

### Supercell Size Mismatch

```
Error: mb_evaluate failed with status 3
```

**Cause**: Structure size doesn't match `ncell` parameter.

**Solution**: 
- If `ncell: 2 2 2`, use a 2×2×2 supercell (natom_unit × 8 atoms)
- Use `mlsupercell` to create the correct size

### Atom Ordering Issues

If forces/energies look wrong:

```ini
# Enable automatic atom matching in config
auto_match_atoms: true
match_tolerance: 0.1
```

## Performance Tips

1. **Use appropriate supercell**: Match `ncell` to your system size
2. **Reuse calculators**: Initialize once, use for multiple structures
3. **MD timestep**: 1-2 fs typical for MULTIBINIT potentials
4. **Phonon supercell**: Start with 2×2×2, increase if needed

## Advanced: Configuration File Options

Complete list of configuration parameters:

```ini
# Required
ddb_file: system_DDB         # DFT database
sys_file: system.xml          # System definition
ncell: 2 2 2                  # Supercell size

# Optional physics
ngqpt: 4 4 4                  # Q-point grid (default: 1 1 1)
dipdip: 1                     # Dipole interactions (0=off, 1=on)

# Optional technical
auto_match_atoms: true        # Automatic atom reordering
match_tolerance: 0.1          # Match tolerance (Angstrom)
use_atomic_units: false       # Use Bohr/Hartree (default: false)
lib_path: /path/to/lib        # Override library location
```

## References

- PyMultibinit: `../../pymultibinit/README.md`
- MULTIBINIT Manual: https://docs.abinit.org/guide/multibinit/
- ABINIT: https://www.abinit.org

## Examples

Complete examples available in:
- `atomchain/examples/03_multibinit/BaHfO3/`
  - `multibinit_relax.py`
  - `multibinit_phonon.py`
  - `multibinit_md_nve.py`
  - `multibinit_md_nvt.py`
