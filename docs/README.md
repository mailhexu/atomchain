# AtomChain Documentation

Complete documentation for AtomChain CLI tools and Python APIs.

## Installation

AtomChain requires Python 3.10 or newer.

For development from the repository root:

```bash
uv sync
```

For editable installation with `pip`:

```bash
pip install -e .
```

MACE support is optional. Install it with:

```bash
pip install -e '.[mace]'
```

or:

```bash
uv sync --extra mace
```

For `mace-r2scan`, AtomChain automatically downloads the MACE-MH-1 model to `~/.config/mace/mace-mh-1.model` when it is missing. If automatic download fails, manually download `https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model` and save it at that path.

## Quick Links

- **[MULTIBINIT Tutorial](multibinit.md)** - MULTIBINIT calculator use through `pymultibinit`
- **[DDB Guide](ddb.md)** - ABINIT-style DDB writer from phonopy and finite-difference workflows
- **[HIST/Training Guide](hist_training.md)** - ABINIT HIST conversion and MULTIBINIT training artifacts
- **[MD Guide](md.md)** - Molecular dynamics with ML calculators

## CLI Tools Overview

AtomChain provides command-line tools for common atomistic modeling workflows:

| Tool | Purpose | Output |
|------|---------|--------|
| `mlsinglepoint` | Single point energy/forces/stress calculations | YAML file |
| `mlsupercell` | Generate supercells with transformation matrices | Structure file |
| `mlrattle` | Generate rattled structure datasets | Trajectory file |
| `mlbatch` | Batch process trajectories with ML potentials | Trajectory file |
| `mlcompare` | Compare two property-bearing trajectories | Plot and metrics |
| `mlrelax` | Relax atomic structures | Structure file |
| `mlphonon` | Calculate phonon properties | Phonopy files |
| `mlgap` | Predict MatGL band gaps | Console output |
| `mlneb` | Run NEB pathway calculations | Images, trajectory/log files |
| `mlcollect` | Collect structures into one trajectory | Trajectory file |
| `mlconvert` | Convert between ASE-supported formats | Structure/trajectory file |
| `mlmetastable` | Explore symmetry-mode metastable structures | Structures, reports, plots |
| `mlddb` | Write ABINIT-style DDB files | `.ddb` plus `.ddb.yaml` |
| `mlhist` | Convert ASE trajectory and ABINIT HIST.nc | `.traj` or `HIST.nc` |
| `mltraining` | Generate/delegate MULTIBINIT training artifacts | Trajectory, DDB, HIST, delegated output |

## Documentation by Topic

### Getting Started
- **[multibinit.md](multibinit.md)** - **MULTIBINIT Tutorial** (START HERE for MULTIBINIT users)
  - Calculator setup and config file examples
  - Workflow examples for relaxation, phonons, and MD
  - Python API and troubleshooting
- **[ddb.md](ddb.md)** - ABINIT-style DDB writer subset, units, q-grid, and validation
- **[hist_training.md](hist_training.md)** - ABINIT HIST I/O and MULTIBINIT training artifact generation

### Structure Manipulation
- **[supercell.md](supercell.md)** - Generate supercells
- **[rattle.md](rattle.md)** - Generate perturbed structures for training

### Calculations
- **[singlepoint.md](singlepoint.md)** - Energy/forces/stress calculations
- **[relax.md](relax.md)** - Structure optimization
- **[phonon.md](phonon.md)** - Phonon band structures
- **[metastable.md](metastable.md)** - Metastable state exploration from imaginary phonon modes
- **[md.md](md.md)** - Molecular dynamics (NVE, NVT, NPT)
- **[batch.md](batch.md)** - Batch process trajectories
- **[compare.md](compare.md)** - Compare trajectories with energy/force/stress metrics

## Quick Start Examples

### Basic Workflow
```bash
# 1. Generate a supercell
mlsupercell POSCAR --size 2 --output POSCAR_2x2x2

# 2. Calculate single point properties
mlsinglepoint POSCAR_2x2x2 --model chgnet --output_file results.yaml

# 3. Relax the structure
mlrelax POSCAR_2x2x2 --model chgnet --output_file POSCAR_relaxed

# 4. Calculate phonons
mlphonon POSCAR_relaxed --model chgnet --ndim 2 2 2
```

### Training Dataset Generation
```bash
# Generate 100 rattled structures for training
mlrattle POSCAR --stdev 0.05 --nstruct 100 --output training_structures.traj

# Calculate properties in batch
mlbatch training_structures.traj --calculator chgnet --output training_data.traj

# Convert to ABINIT HIST.nc for MULTIBINIT-oriented workflows
mlhist training_data.traj training_HIST.nc --to hist
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
    md_npt,
)

# DDB/HIST/training workflows
from atomchain.ddb import write_ddb_from_finite_difference, write_ddb_from_phonopy
from atomchain.io import read_abinit_hist, write_abinit_hist
from atomchain.training import generate_multibinit_training_artifacts, generate_training_trajectory
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
- **CHGNet** - Materials property prediction (`chgnet`)
- **M3GNet/matgl** - Graph-network potentials (`m3gnet`, `matgl`)
- **MACE** - Foundation models including `mace` and project-specific `mace-r2scan`
- **DeePMD-kit** - `deepmd` with a model path
- **MULTIBINIT** - `multibinit`/`mb` through `pymultibinit` with a config file
- **XQ** - `xq` if `atomic_potential_xq` is installed
- Custom ASE calculators via Python APIs

## Requirements

- Python 3.10+
- ASE (Atomic Simulation Environment)
- Phonopy 3.0.0
- Optional calculator packages depending on the selected model: CHGNet, M3GNet/matgl, MACE (`.[mace]`), DeePMD-kit, pymultibinit, atomic_potential_xq

## Getting Help

Each CLI tool has built-in help:
```bash
mlsinglepoint --help
mlsupercell --help
mlrattle --help
mlbatch --help
mlrelax --help
mlphonon --help
mlddb --help
mlhist --help
mltraining --help
```

For issues or questions, refer to the main project README.
