# Single Point Calculations

Calculate energy, forces, and stress for atomic structures using ML potentials.

## CLI Usage

```bash
# Basic usage
mlsinglepoint structure.cif -o results.yaml

# Specify ML model
mlsinglepoint POSCAR --model mace -o results.yaml

# Custom model file
mlsinglepoint structure.cif -m deepmd -p model.pb -o results.yaml
```

### Options
- `fname` - Input structure file (POSCAR, CIF, XYZ, etc.)
- `--model, -m` - Calculator name accepted by `init_calc()` (default: `chgnet`), such as `chgnet`, `m3gnet`, `matgl`, `mace`, `mace-r2scan`, `deepmd`, `multibinit`, or `mb`
- `--output_file, -o` - Output YAML file (default: `singlepoint_result.yaml`)
- `--model_path, -p` - Optional model/config path for calculators that need one

## Python API

```python
from atomchain.singlepoint import calculate_single_point
from ase.build import bulk

# From Atoms object
atoms = bulk('Al', 'fcc', a=4.05)
result = calculate_single_point(atoms, calc='chgnet')

print(f"Energy: {result.energy} eV")
print(f"Max force: {result.forces.max()} eV/Å")
print(f"Stress: {result.stress}")

# Save to YAML
result.to_yaml('results.yaml')

# Load from YAML
from atomchain.singlepoint import SinglePointResult
loaded = SinglePointResult.from_yaml('results.yaml')
```

### Parameters
- `atoms` - ASE Atoms object or file path
- `calc` - Calculator name (str) or ASE Calculator object (default: `'chgnet'`)
- `model_path` - Path to custom model file (optional)

### Returns
`SinglePointResult` object with:
- `energy` - Potential energy (eV)
- `forces` - Forces array (eV/Å), shape (natoms, 3)
- `stress` - Stress in Voigt notation (eV/Å³), shape (6,)
- `atoms` - ASE Atoms object
- `calculator_name` - Name of calculator used
- `timestamp` - ISO 8601 timestamp

## Output Format

YAML file contains:
```yaml
calculator: chgnet
timestamp: 2025-11-27T23:00:00.000000
energy: -3.45678
forces:
  - [0.0, 0.0, 0.0]
  - [0.01, -0.02, 0.03]
stress: [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
structure:
  cell: [[...], [...], [...]]
  positions: [[...], ...]
  symbols: [Al, Al]
  pbc: [true, true, true]
```
