# atomchain

AtomChain provides CLI tools and Python APIs for atomic structure manipulation and ML potential calculations.

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
- **`mlddb`** - Write ABINIT-style DDB files from phonopy and ML finite-difference workflows
- **`mlhist`** - Convert between ABINIT HIST.nc and ASE trajectory files
- **`mltraining`** - Generate MULTIBINIT training trajectories/artifacts and delegate training to pymultibinit

## Installation

```bash
pip install -e .
```

## Quick Start

### Single Point Calculation
```bash
mlsinglepoint input.vasp --calculator chgnet --output results.yaml
```

### Generate Supercell
```bash
mlsupercell input.vasp --size 2 --output supercell.vasp
```

### Generate Training Dataset
```bash
mlrattle input.vasp --stdev 0.05 --count 100 --output structures.traj
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
mlrelax input.vasp --calculator chgnet --output relaxed.vasp
```

### Calculate Phonons
```bash
mlphonon input.vasp --calculator chgnet --supercell 2,2,2
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
- [docs/relax.md](docs/relax.md) - Structure relaxation
- [docs/phonon.md](docs/phonon.md) - Phonon calculations
- [docs/ddb.md](docs/ddb.md) - ABINIT-style DDB writer
- [docs/hist_training.md](docs/hist_training.md) - ABINIT HIST and MULTIBINIT training artifacts

## Python API

All CLI tools have corresponding Python APIs for programmatic use:

```python
from ase.io import read
from atomchain import (
    relax_with_ml,
    phonon_with_ml,
    init_calc,
    calculate_single_point,
    make_supercell_structure,
    generate_rattle_dataset,
    calculate_trajectory_batch,
    compare_trajectories,
    calculate_neb,
    predict_gap,
    read_abinit_hist,
    write_abinit_hist,
    generate_training_trajectory,
)

atoms = read("structure.vasp")

# Relax structure
relaxed_atoms = relax_with_ml(atoms, calculator="chgnet")

# Calculate phonons
phonon_with_ml(atoms, calculator="chgnet", supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]])

# Predict band gap
gap = predict_gap(atoms, xc="PBE")

# Single point calculation
results = calculate_single_point(atoms, calculator="chgnet")

# Generate supercell
supercell = make_supercell_structure(atoms, size=2)

# Generate dataset
generate_rattle_dataset(
    atoms,
    stdev=0.05,
    count=100,
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
images = [read(f"image{i}.vasp") for i in range(5)]
calculate_neb(images, calculator="chgnet")

# HIST conversion and training trajectory generation
frames = generate_training_trajectory(atoms, sources=["phonon_modes"], evaluate=False)
write_abinit_hist(frames, "training_HIST.nc", strict=False)
loaded_frames = read_abinit_hist("training_HIST.nc")
```

## Requirements

- Python 3.9+
- ASE (Atomic Simulation Environment)
- Optional: CHGNet, M3GNet, MACE for ML potentials
- Optional: Phonopy for phonon analysis

## License

[Your license here]
