# Supercell Generation

Generate supercells from primitive structures with flexible scaling options.

## CLI Usage

```bash
# Isotropic 2×2×2 supercell
mlsupercell structure.cif --size 2 -o supercell.vasp

# Diagonal 2×2×3 supercell
mlsupercell POSCAR --diagonal 2,2,3 -o SPOSCAR

# Custom transformation matrix
mlsupercell input.cif --matrix 2,0,0,0,2,0,0,0,3 -o output.cif

# VASP-compatible atom ordering
mlsupercell POSCAR --diagonal 2,2,2 --order atom-major -o SPOSCAR
```

### Options
- `fname` - Input structure file (POSCAR, CIF, XYZ, etc.)
- `--size, -s` - Isotropic supercell size (e.g., `2` for 2×2×2)
- `--diagonal, -d` - Diagonal scaling (e.g., `2,2,3`)
- `--matrix, -m` - 3×3 transformation matrix (9 comma-separated values, row-major)
- `--output, -o` - Output file (default: `<input>_supercell.<ext>`)
- `--format, -f` - Output format (default: same as input)
- `--order` - Atom ordering: `cell-major` (default) or `atom-major` (VASP)
- `--no-wrap` - Do not wrap positions into supercell

## Python API

```python
from atomchain.supercell import make_supercell_structure
from ase.build import bulk
from ase.io import write

atoms = bulk('Al', 'fcc', a=4.05)

# Isotropic 2×2×2
supercell = make_supercell_structure(atoms, 2)

# Diagonal 2×2×3
supercell = make_supercell_structure(atoms, [2, 2, 3])

# Custom matrix
import numpy as np
P = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 3]])
supercell = make_supercell_structure(atoms, P)

# VASP ordering
supercell = make_supercell_structure(atoms, [2, 2, 2], order='atom-major')

# From file
supercell = make_supercell_structure('POSCAR', [2, 2, 2])

write('SPOSCAR', supercell)
```

### Parameters
- `atoms` - ASE Atoms object or file path
- `P` - Transformation specification:
  - `int`: Isotropic (e.g., `2` → 2×2×2)
  - `List[int]`: Diagonal (e.g., `[2,2,3]`)
  - `np.ndarray`: 3×3 matrix
- `wrap` - Wrap positions into supercell (default: `True`)
- `order` - Atom ordering: `'cell-major'` (default) or `'atom-major'`

### Returns
ASE Atoms object with `N × det(P)` atoms

### Notes
- Matrix must contain only integers
- Determinant must be positive
- Atom ordering affects VASP compatibility
