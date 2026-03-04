# AtomChain Documentation

Complete documentation for AtomChain CLI tools and Python APIs.

## Quick Links

- 🚀 **[MULTIBINIT Tutorial](multibinit.md)** - Complete guide for MULTIBINIT effective potentials
- 📖 [MD Guide](md.md) - Molecular dynamics with 7 different ensembles/thermostats
- 📊 [Phonon Guide](phonon.md) - Calculate phonon band structures
- 🔧 [Relax Guide](relax.md) - Structure optimization

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

## Documentation by Topic

### Getting Started
- **[multibinit.md](multibinit.md)** - **MULTIBINIT Tutorial** (START HERE for MULTIBINIT users)
  - Complete setup guide (installation, library path, config file)
  - Workflow examples (relaxation, phonon, MD)
  - Python API and troubleshooting

### Structure Manipulation
- **[supercell.md](supercell.md)** - Generate supercells
- **[rattle.md](rattle.md)** - Generate perturbed structures for training

### Calculations
- **[singlepoint.md](singlepoint.md)** - Energy/forces/stress calculations
- **[relax.md](relax.md)** - Structure optimization
- **[phonon.md](phonon.md)** - Phonon band structures
- **[md.md](md.md)** - Molecular dynamics (NVE, NVT, NPT)
- **[batch.md](batch.md)** - Batch process trajectories

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
from atomchain.relax import relax_with_ml
from atomchain.phonon.mlphonon import phonon_with_ml

# Molecular dynamics
from atomchain.md import (
    md_nve_velocity_verlet,
    md_nvt_langevin,
    md_nvt_berendsen,
    md_npt_berendsen,
    md_npt
)
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
