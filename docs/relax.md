# Structure Relaxation

Relax atomic positions and/or cell shape using ML potentials with the FIRE optimizer.

## CLI Usage

```bash
# Basic relaxation (positions only)
mlrelax structure.cif -o relaxed.vasp

# Relax both positions and cell
mlrelax POSCAR --relax_cell -o relaxed.vasp

# With symmetry constraints
mlrelax structure.cif --sym --relax_cell -o relaxed.vasp

# Specify ML model
mlrelax POSCAR --model mace --relax_cell -o relaxed.vasp

# Adjust force convergence
mlrelax structure.cif --fmax 0.01 -o relaxed.vasp

# Fix specific atoms
mlrelax POSCAR --fix_atoms 0 1 2 -o relaxed.vasp
```

### Options
- `fname` - Input structure file (POSCAR, CIF, XYZ, etc.)
- `--model, -m` - Calculator name accepted by `init_calc()` (default: `chgnet`), such as `chgnet`, `m3gnet`, `matgl`, `mace`, `mace-r2scan`, `deepmd`, `multibinit`, or `mb`
- `--relax_cell, -r` - Relax cell shape (default: False, positions only)
- `--sym, -s` - Apply symmetry constraints (default: False)
- `--fmax, -f` - Max force convergence in eV/Å (default: 0.001)
- `--cell_factor, -c` - Cell scaling factor for stress (default: 100)
- `--output_file, -o` - Output file (default: `POSCAR_relax.vasp`)
- `--model_path, -p` - Path to custom model/config file (default: `model.dp`)
- `--fix_atoms, -fa` - Indices of atoms to fix (e.g., `0 1 2`)

## Python API

```python
from atomchain.relax import relax_with_ml
from ase.io import read, write

# Load structure
atoms = read('POSCAR')

# Basic relaxation (positions only)
relaxed = relax_with_ml(atoms, calc='chgnet')

# Relax positions and cell
relaxed = relax_with_ml(
    atoms,
    calc='chgnet',
    relax_cell=True,
    fmax=0.01
)

# With symmetry constraints
relaxed = relax_with_ml(
    atoms,
    calc='mace',
    relax_cell=True,
    sym=True,
    fmax=0.001
)

# Fix specific atoms
relaxed = relax_with_ml(
    atoms,
    calc='chgnet',
    fix_atoms=[0, 1, 2],
    relax_cell=True
)

# Custom calculator object
from atomchain.init_model import init_calc
calc = init_calc('chgnet')
relaxed = relax_with_ml(atoms, calc=calc)

# Save result
write('relaxed.vasp', relaxed)
```

### Parameters
- `atoms` - ASE Atoms object to relax
- `calc` - Calculator: string name or ASE Calculator object (default: `'chgnet'`)
- `relax_cell` - Relax cell shape (default: `True`)
- `sym` - Apply symmetry constraints (default: `True`)
- `traj_file` - Trajectory file name (default: `'relax.traj'`)
- `model_path` - Path to custom model file (optional)
- `fmax` - Force convergence criterion in eV/Å (default: 0.001)
- `cell_factor` - Cell stress scaling factor (default: 100)
- `rattle` - Initial random displacement (optional)
- `fix_atoms` - List of atom indices to fix (optional)
- `**ucf_kwargs` - Additional UnitCellFilter arguments

### Returns
Relaxed ASE Atoms object

## Notes

- Uses FIRE optimizer (fast inertial relaxation engine)
- Cell relaxation uses UnitCellFilter wrapper
- Two-stage relaxation when `relax_cell=True`:
  1. Coarse: `fmax × 10` for 3500 steps
  2. Fine: `fmax` for 5000 steps
- Symmetry constraints preserve space group
- Fixed atoms remain at original positions
- Trajectory saved to `relax.traj` during relaxation
