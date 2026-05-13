# Phonon Calculations

Calculate phonon band structures and density of states using ML potentials with the frozen phonon method.

## CLI Usage

```bash
# Basic phonon calculation
mlphonon structure.cif

# With structure relaxation first
mlphonon POSCAR --relax

# Specify ML model
mlphonon structure.cif --model mace

# Custom supercell size
mlphonon POSCAR --ndim 3 3 3

# Custom k-path for band structure
mlphonon structure.cif --kpath GXMG

# Custom output figure name
mlphonon POSCAR --figname phonon_bands.pdf --npoints 200
```

### Options
- `fname` - Input structure file (POSCAR, CIF, XYZ, etc.)
- `--model, -m` - Calculator name accepted by `init_calc()` (default: `chgnet`), such as `chgnet`, `m3gnet`, `matgl`, `mace`, `mace-r2scan`, `deepmd`, `multibinit`, or `mb`
- `--model_path` - Optional model/config path for calculators that need one, such as `multibinit` or `deepmd`
- `--relax, -r` - Relax structure before phonon calculation (default: False)
- `--ndim, -n` - Supercell size (3 integers, default: `2 2 2`)
- `--kpath, -k` - K-path string (e.g., `GXMG`). Auto-detected if not specified
- `--npoints, -p` - Number of points in band structure (default: 100)
- `--figname, -f` - Output figure file name (CLI default: `phonon.png`)

## Python API

```python
from atomchain.phonon.mlphonon import phonon_with_ml
from ase.io import read
import numpy as np

# Load structure
atoms = read('POSCAR')

# Basic phonon calculation
phonon_with_ml(atoms, calc='chgnet')

# With relaxation
phonon_with_ml(
    atoms,
    calc='chgnet',
    relax=True
)

# Custom supercell and k-path
phonon_with_ml(
    atoms,
    calc='mace',
    ndim=np.diag([3, 3, 3]),
    knames='GXMG',
    npoints=200,
    figname='phonon_bands.pdf'
)

# Advanced: pass additional phonopy parameters
phonon_with_ml(
    atoms,
    calc='chgnet',
    distance=0.03,          # Displacement distance (Å)
    is_symmetry=True,       # Use crystal symmetry
    symprec=1e-5,          # Symmetry precision
    primitive_matrix=np.eye(3)  # Primitive cell transformation
)
```

### Parameters
- `atoms` - ASE Atoms object
- `calc` - Calculator: string name or ASE Calculator object (default: `'chgnet'`)
- `relax` - Relax structure before calculation (default: `False`)
- `plot` - Generate band structure plot (default: `True`)
- `knames` - K-path string or None for auto-detection (optional)
- `kvectors` - Custom k-point vectors (optional)
- `npoints` - Number of points in band structure (default: 100)
- `model_path` - Optional model/config path for calculators that need one
- `figname` - Output figure file name (Python API default: `'phonon.pdf'`)
- `**kwargs` - Additional parameters passed to `calculate_phonon()`:
  - `ndim` - Supercell matrix (default: `np.diag([2,2,2])`)
  - `distance` - Displacement distance in Å (default: 0.05)
  - `is_symmetry` - Use crystal symmetry (default: `True`)
  - `symprec` - Symmetry precision (default: 1e-3)
  - `primitive_matrix` - Primitive cell matrix (default: `np.eye(3)`)

### Output
- Phonon data saved to `phonon_save/` directory
- Band structure plot saved to specified figure file
- Displays plot automatically if `plot=True`

## Workflow

The phonon calculation follows these steps:

1. **Structure preparation** - Optionally relax input structure
2. **Supercell generation** - Create supercell for finite displacement
3. **Displacement generation** - Generate symmetry-inequivalent displacements
4. **Force calculations** - Calculate forces for each displaced structure
5. **Force constants** - Build dynamical matrix from forces
6. **Band structure** - Calculate phonon frequencies along k-path
7. **Plotting** - Generate and display band structure plot

## Output Files

Generated in current directory:
- `phonon_save/` - Directory containing phonopy files
  - `phonopy.yaml` - Main phonopy configuration
  - `FORCE_SETS` - Calculated forces
  - `band.yaml` - Band structure data
- `phonon.pdf` - Band structure plot (or custom name)

## Notes

- Uses frozen phonon method via Phonopy
- Finite displacement approach (default: 0.05 Å)
- Automatic symmetry detection reduces calculations
- K-path auto-detection using pymatgen/seekpath if available
- Frequencies in THz (VASP units)
- Band structure plotting supports custom k-paths
- Results compatible with Phonopy tools

## Example: Complete Workflow

```python
from atomchain.phonon.mlphonon import phonon_with_ml
from atomchain.relax import relax_with_ml
from ase.io import read

# 1. Load structure
atoms = read('POSCAR')

# 2. Relax (optional but recommended)
atoms = relax_with_ml(atoms, calc='chgnet', relax_cell=True)

# 3. Calculate phonons
phonon_with_ml(
    atoms,
    calc='chgnet',
    ndim=np.diag([3, 3, 3]),
    knames='GXMG',
    figname='phonon_chgnet.pdf'
)

# Results in:
# - phonon_save/ directory with phonopy files
# - phonon_chgnet.pdf band structure plot
```
