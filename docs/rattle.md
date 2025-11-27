# Rattle Dataset Generation

Generate training datasets with random atomic displacements and optional cell strain. Structures are saved to trajectory files for later property calculations.

## CLI Usage

```bash
# Basic generation
mlrattle structure.cif --nstruct 100 --stdev 0.05 -o dataset.traj

# With supercell
mlrattle POSCAR --nstruct 50 --supercell 2,2,2 -o dataset.traj

# With cell strain
mlrattle input.cif --nstruct 100 --stdev 0.05 --cell-stdev 0.02

# Reproducible with seed
mlrattle structure.cif --nstruct 50 --seed 42 -o dataset.traj

# Quiet mode
mlrattle POSCAR --nstruct 100 --stdev 0.05 -o dataset.traj --quiet
```

### Options
- `fname` - Input structure file (POSCAR, CIF, XYZ, etc.)
- `--nstruct, -n` - Number of structures to generate (default: 100)
- `--stdev, -s` - Atomic displacement std dev in Å (default: 0.05)
- `--supercell, -sc` - Supercell: single int (`2`) or diagonal (`2,2,3`)
- `--cell-stdev, -cs` - Cell strain std dev (default: None)
- `--seed` - Random seed for reproducibility
- `--output, -o` - Output trajectory file (default: `rattle_dataset.traj`)
- `--quiet, -q` - Suppress progress messages

## Python API

```python
from atomchain.rattle import generate_rattle_dataset
from ase.build import bulk
from ase.io import read

# Basic generation
atoms = bulk('Al', 'fcc', a=4.05)
traj_path = generate_rattle_dataset(
    atoms, 
    n_struct=100, 
    stdev=0.05,
    output='dataset.traj'
)

# With supercell and cell strain
traj_path = generate_rattle_dataset(
    'POSCAR',
    n_struct=100,
    stdev=0.05,
    supercell=[2, 2, 2],
    cell_stdev=0.02,
    seed=42
)

# Read generated structures
structures = read(traj_path, ':')
print(f"Generated {len(structures)} structures")

# Structures can be used for property calculations
for atoms in structures:
    # Calculate properties with your preferred method
    pass
```

### Parameters
- `atoms` - ASE Atoms object or file path
- `n_struct` - Number of structures to generate (default: 100)
- `stdev` - Atomic displacement std dev in Å (default: 0.05)
- `supercell` - Supercell specification: `int` or `List[int]` (optional)
- `cell_stdev` - Cell strain std dev (optional)
- `seed` - Random seed for reproducibility (optional)
- `output` - Output trajectory file (default: `'rattle_dataset.traj'`)
- `verbose` - Print progress messages (default: `True`)

### Returns
Path to output trajectory file (str)

## Workflow

1. **Generate structures** - Creates structures with random displacements
2. **Calculate properties** - Use your preferred method separately
3. **Training** - Use structures and properties for ML training

### Example: Full Workflow

```python
from atomchain.rattle import generate_rattle_dataset
from atomchain.singlepoint import calculate_single_point
from ase.io import read, write

# 1. Generate rattled structures
traj_path = generate_rattle_dataset(
    'POSCAR',
    n_struct=100,
    stdev=0.05,
    supercell=[2, 2, 2]
)

# 2. Calculate properties (example with CHGNet)
structures = read(traj_path, ':')
for i, atoms in enumerate(structures):
    result = calculate_single_point(atoms, calc='chgnet')
    
    # Store properties in atoms object
    atoms.info['energy'] = result.energy
    atoms.arrays['forces'] = result.forces
    atoms.info['stress'] = result.stress.tolist()

# 3. Save with properties
write('dataset_with_properties.traj', structures)
```

## Parameter Guidelines

**Atomic Displacement (`stdev`):**
- Small: 0.01-0.03 Å (near equilibrium)
- Medium: 0.05-0.08 Å (typical training)
- Large: 0.10-0.15 Å (high temperature, melting)

**Cell Strain (`cell_stdev`):**
- Small: 0.01-0.02 (subtle variations)
- Medium: 0.03-0.05 (typical training)
- Avoid > 0.1 (may create unphysical structures)

**Number of Structures:**
- Quick test: 10-50
- Small dataset: 100-500
- Training dataset: 1000-10000

## Notes

- Structures saved to ASE trajectory format (.traj)
- No property calculations performed (calculate separately)
- Supercell applied before rattling
- Cell strain applied before atomic displacement
- Use `seed` parameter for reproducible datasets
- Progress reported every 10% by default
