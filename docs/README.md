# AtomChain Documentation

Complete documentation for AtomChain CLI tools and Python APIs.

## CLI Tools Overview

AtomChain provides command-line tools for common atomistic modeling workflows:

| Tool | Purpose | Output |
|------|---------|--------|
| `mlsinglepoint` | Single point energy/forces/stress calculations | YAML file |
| `mlsupercell` | Generate supercells with transformation matrices | Structure file |
| `mlrattle` | Generate rattled structure datasets | Trajectory file |
| `mlbatch` | Batch process trajectories with ML potentials | Trajectory file |
| `mlrelax` | Relax atomic structures | Structure file |
| `mlphonon` | Calculate phonon properties | Phonopy files |

## Detailed Documentation

### Structure Manipulation
- **[supercell.md](supercell.md)** - Generate supercells with various transformations
  - Isotropic, diagonal, and matrix supercells
  - Atom ordering options (cell-major/atom-major)
  - VASP compatibility features

- **[rattle.md](rattle.md)** - Generate perturbed structure datasets
  - Random atomic displacements for training data
  - Supercell generation with cell strain
  - Workflow for dataset generation and property calculation

### Calculations with ML Potentials
- **[batch.md](batch.md)** - Batch trajectory processing
  - Calculate properties for all structures in a trajectory
  - Efficient calculator reuse across structures
  - Complete workflow: generation → calculation → analysis
- **[singlepoint.md](singlepoint.md)** - Single point calculations
  - Energy, forces, stress calculations
  - Multiple calculator support (CHGNet, M3GNet, MACE)
  - YAML output format

- **[relax.md](relax.md)** - Structure relaxation
  - FIRE optimizer with convergence criteria
  - Optional symmetry constraints
  - Cell relaxation options

- **[phonon.md](phonon.md)** - Phonon calculations
  - Frozen phonon method with supercells
  - Band structure plotting
  - Phonopy integration

## Quick Start Examples

### Basic Workflow
```bash
# 1. Generate a supercell
mlsupercell POSCAR --size 2 --output POSCAR_2x2x2

# 2. Calculate single point properties
mlsinglepoint POSCAR_2x2x2 --calculator chgnet --output results.yaml

# 3. Relax the structure
mlrelax POSCAR_2x2x2 --calculator chgnet --output POSCAR_relaxed

# 4. Calculate phonons
mlphonon POSCAR_relaxed --calculator chgnet --supercell 2,2,2
```

### Training Dataset Generation
```bash
# Generate 100 rattled structures for training
mlrattle POSCAR --stdev 0.05 --count 100 --output training_structures.traj

# Calculate properties in batch
mlbatch training_structures.traj --calculator chgnet --output training_data.traj
```

## Python API

All tools have Python APIs for programmatic use. Import from the corresponding module:

```python
# Structure manipulation
from atomchain.supercell import make_supercell_structure
from atomchain.rattle import generate_rattle_dataset

# Calculations
from atomchain.singlepoint import calculate_single_point
from atomchain.batch import calculate_trajectory_batch
from atomchain.relax import relax_structure
from atomchain.phonon.mlphonon import calculate_phonon
```

See individual documentation files for detailed API examples.

## Supported File Formats

Input/Output formats supported by ASE:
- VASP (POSCAR/CONTCAR)
- CIF
- XYZ
- Trajectory (.traj)
- And many more via ASE

## Calculator Support

Supported ML potential calculators:
- **CHGNet** - Materials property prediction
- **M3GNet** - Multi-element graph networks
- **MACE** - Message passing neural networks
- Custom ASE calculators (via Python API)

## Requirements

- Python 3.9+
- ASE (Atomic Simulation Environment)
- Optional: CHGNet, M3GNet, MACE for specific calculators
- Optional: Phonopy for phonon analysis and visualization

## Getting Help

Each CLI tool has built-in help:
```bash
mlsinglepoint --help
mlsupercell --help
mlrattle --help
mlbatch --help
mlrelax --help
mlphonon --help
```

For issues or questions, refer to the main project README.
