# atomchain

AtomChain provides CLI tools and Python APIs for atomic structure manipulation, ML-potential calculations, phonon/DDB workflows, and ABINIT HIST/MULTIBINIT training artifact preparation.

## CLI Tools

AtomChain includes several command-line tools for common atomistic workflows:

- **`mlrelax`** - Relax atomic structures using ML potentials
- **`mlphonon`** - Calculate phonon properties and band structures
- **`mlgap`** - Predict band gap using ML potentials
- **`mlsinglepoint`** - Single point energy/forces/stress calculations
- **`mlsupercell`** - Generate supercells with various transformation matrices
- **`mlrattle`** - Generate rattled structure datasets for training
- **`mlbatch`** - Batch process trajectories with ML potentials
- **`mlcompare`** - Compare calculated properties between two trajectories
- **`mlneb`** - Nudged elastic band calculations for reaction pathways
- **`mlcollect`** - Collect structures from many files into one trajectory
- **`mlconvert`** - Convert structures between ASE-supported file formats
- **`mlmetastable`** - Explore symmetry-mode metastable structures
- **`mlddb`** - Write ABINIT-style DDB files from phonopy and ML finite-difference workflows
- **`mlhist`** - Convert between ABINIT HIST.nc and ASE trajectory files
- **`mltraining`** - Generate MULTIBINIT training trajectories/artifacts and delegate training to pymultibinit

## Installation

AtomChain requires Python 3.10 or newer.

For development from this repository, use `uv`:

```bash
uv sync
```

For an editable installation with `pip`, use this from the repository root:

```bash
pip install -e .
```

MACE calculators are optional. To install AtomChain with MACE support, use:

```bash
pip install -e '.[mace]'
```

or with `uv`:

```bash
uv sync --extra mace
```

The `mace-r2scan` calculator uses the MACE-MH-1 model. When the model file is missing, AtomChain tries to download it automatically from Hugging Face to `~/.config/mace/mace-mh-1.model`. If automatic download fails, download it manually from:

```text
https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model
```

and save it as:

```text
~/.config/mace/mace-mh-1.model
```

## Quick Start

### Single Point Calculation
```bash
mlsinglepoint input.vasp --model chgnet --output_file results.yaml
```

### Generate Supercell
```bash
mlsupercell input.vasp --size 2 --output supercell.vasp
```

### Generate Training Dataset
```bash
mlrattle input.vasp --stdev 0.05 --nstruct 100 --output structures.traj
```

### Batch Process Trajectory
```bash
mlbatch structures.traj --calculator chgnet --output results.traj
```

### Compare Trajectories
```bash
mlcompare dft.traj ml.traj --labels "DFT" "CHGNet" --output comparison.png
```

### Relax Structure
```bash
mlrelax input.vasp --model chgnet --output_file relaxed.vasp
```

### Calculate Phonons
```bash
mlphonon input.vasp --model chgnet --ndim 2 2 2
```

### Write DDB From Phonopy
```bash
mlddb BaTiO3.vasp --phonopy-yaml phonon_save/phonopy_params.yaml --output BaTiO3.ddb --validate
```

### Write ABINIT HIST From Trajectory
```bash
mlhist training.traj training_HIST.nc --to hist
```

### Generate Training Trajectory
```bash
mltraining generate BaTiO3.vasp --sources md phonon_modes --model mace-r2scan --output training.traj
```

## Documentation

Detailed documentation for each tool is available in the `docs/` directory:

- [docs/singlepoint.md](docs/singlepoint.md) - Single point calculations
- [docs/supercell.md](docs/supercell.md) - Supercell generation
- [docs/rattle.md](docs/rattle.md) - Dataset generation
- [docs/batch.md](docs/batch.md) - Batch trajectory processing
- [docs/compare.md](docs/compare.md) - Trajectory comparison
- [docs/metastable.md](docs/metastable.md) - Metastable state exploration from imaginary phonon modes
- [docs/relax.md](docs/relax.md) - Structure relaxation
- [docs/phonon.md](docs/phonon.md) - Phonon calculations
- [docs/md.md](docs/md.md) - Molecular dynamics
- [docs/multibinit.md](docs/multibinit.md) - MULTIBINIT calculator usage through pymultibinit
- [docs/ddb.md](docs/ddb.md) - ABINIT-style DDB writer
- [docs/hist_training.md](docs/hist_training.md) - ABINIT HIST and MULTIBINIT training artifacts

## Python API

All CLI tools have corresponding Python APIs for programmatic use:

```python
from ase.io import read
from atomchain import (
    calculate_single_point,
    calculate_trajectory_batch,
    calculate_neb,
    compare_trajectories,
    explore_metastable_states,
    generate_multibinit_training_artifacts,
    generate_rattle_dataset,
    generate_training_trajectory,
    init_calc,
    make_supercell_structure,
    phonon_with_ml,
    read_abinit_hist,
    relax_with_ml,
    write_ddb_from_finite_difference,
    write_ddb_from_phonopy,
    write_abinit_hist,
)

atoms = read("structure.vasp")

# Relax structure
relaxed_atoms = relax_with_ml(atoms, calc="chgnet")

# Calculate phonons
phonon_with_ml(atoms, calc="chgnet", ndim=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])

# Single point calculation
results = calculate_single_point(atoms, calc="chgnet")

# Generate supercell
supercell = make_supercell_structure(atoms, 2)

# Generate dataset
generate_rattle_dataset(
    atoms,
    stdev=0.05,
    n_struct=100,
    output="dataset.traj"
)

# Batch process trajectory
results = calculate_trajectory_batch(
    "dataset.traj",
    calculator="chgnet",
    output="results.traj"
)

# Compare trajectories
compare_trajectories("dft.traj", "ml.traj", labels=["DFT", "CHGNet"])

# NEB calculation
initial = read("initial.vasp")
final = read("final.vasp")
calculate_neb(initial, final, calculator="chgnet")

# HIST conversion and training trajectory generation
frames = generate_training_trajectory(atoms, sources=["phonon_modes"], evaluate=False)
write_abinit_hist(frames, "training_HIST.nc", strict=False)
loaded_frames = read_abinit_hist("training_HIST.nc")
```

## Requirements

- Python 3.10+
- ASE (Atomic Simulation Environment)
- Phonopy 3.0.0
- Optional: CHGNet, M3GNet, matgl, MACE (`pip install -e '.[mace]'`), DeePMD-kit, atomic_potential_xq, and pymultibinit depending on selected calculator/workflow

## License

BSD-2-Clause
