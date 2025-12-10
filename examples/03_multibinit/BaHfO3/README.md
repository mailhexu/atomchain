# MULTIBINIT Examples for AtomChain

This directory contains comprehensive examples demonstrating how to use ABINIT's MULTIBINIT effective potential with AtomChain for structure relaxation and phonon calculations.

## Overview

**MULTIBINIT** provides physics-based effective potentials derived from first-principles DFT calculations. These potentials capture:
- Anharmonic effects beyond harmonic approximation
- Long-range dipole-dipole interactions (crucial for ferroelectrics)
- Temperature-dependent behavior
- Efficient evaluation for large supercells and long MD simulations

**Use cases:**
- Ferroelectric materials (BaHfO3, BaTiO3, PbTiO3, etc.)
- Structural phase transitions
- Domain wall dynamics
- Long-timescale molecular dynamics (ns-μs)
- Thermodynamic property calculations

## Prerequisites

### 1. Install pymultibinit

```bash
# Navigate to pymultibinit directory
cd /Users/hexu/projects/abinit_git/pymultibinit_dev/pymultibinit

# Install with pip in development mode
pip install -e .
```

**Requirement:** pymultibinit >= 0.2.0

### 2. Generate MULTIBINIT Potential Files

Before running these examples, you need DDB and XML files from ABINIT:

**Step 1: DFT Calculation with ABINIT**
```bash
# Run DFPT calculation to generate DDB
abinit input.abi
```

**Step 2: MULTIBINIT Training**
```bash
# Train effective potential
multibinit < multibinit.in
```

This generates:
- `system_DDB` - Derivative database with force constants
- `system.xml` - Effective Hamiltonian coefficients

**Resources:**
- [ABINIT MULTIBINIT Tutorial](https://docs.abinit.org/tutorial/lattice_model/)
- [MULTIBINIT Documentation](https://docs.abinit.org/topics/LatticeModel/)

## Files in This Directory

```
03_multibinit/
├── README.md                      # This file
├── BaHfO3_config.conf             # Configuration for BaHfO3
├── multibinit_relaxation.py       # Structure relaxation example
├── multibinit_phonon.py           # Phonon calculation example
├── cli_relaxation.sh              # CLI usage notes
└── cli_phonon.sh                  # CLI usage notes
```

## Example 1: Structure Relaxation

### BaHfO3 (Barium Hafnate)

**System:** Prototypical ferroelectric with perovskite structure
- 5 atoms/primitive cell (Ba, Ti, 3×O)
- Cubic → Tetragonal phase transition at ~120°C
- Large spontaneous polarization

**What it demonstrates:**
- Loading/creating atomic structures
- Initializing MULTIBINIT calculator with config file
- Full structure relaxation (positions + cell)
- Analyzing ferroelectric distortions

### Run the Example

```bash
cd examples/03_multibinit

# Make sure you have BaHfO3_DDB and BaHfO3.xml in this directory
# (Generated from ABINIT, see Prerequisites above)

python multibinit_relaxation.py
```

### Expected Output

```
Initial structure created: 5 atoms
Chemical formula: BaHfO3

Initializing MULTIBINIT calculator...
✓ MULTIBINIT calculator initialized successfully

Initial state:
  Energy: -XX.XXXXXX eV
  Max force: X.XXXX eV/Å

Starting structure relaxation...
✓ Relaxation completed

Relaxation Results
------------------
Energetics:
  Energy change: -0.XXX eV
Forces:
  Final max force: 0.008 eV/Å (converged!)
Cell parameters:
  Volume change: X.XXX Å³
Ti displacement (ferroelectric order parameter):
  Final: [0.02, 0.02, 0.02]

Output files:
  BaHfO3_relaxed.vasp
  BaHfO3_relaxed.cif
  BaHfO3_relax.traj
```

### Python API Usage

```python
from ase.io import read
from atomchain.init_model import init_calc
from atomchain.relax import relax_with_ml

# Load structure
atoms = read('BaHfO3_initial.vasp')

# Initialize MULTIBINIT calculator
calc = init_calc(model_type="multibinit", model_path="BaHfO3_config.conf")

# Or use short alias:
# calc = init_calc(model_type="mb", model_path="BaHfO3_config.conf")

# Perform relaxation
relaxed = relax_with_ml(
    atoms=atoms,
    calc=calc,
    fmax=0.01,
    relax_cell=True,
    traj_file='relax.traj'
)
```

## Example 2: Phonon Calculation

**What it demonstrates:**
- Phonon band structure calculation
- Phonon density of states
- Identifying soft modes (ferroelectric instabilities)
- Analyzing vibrational properties

### Run the Example

```bash
python multibinit_phonon.py
```

### Expected Output

```
Loaded relaxed structure from: BaHfO3_relaxed.vasp
Structure: 5 atoms
Cell volume: 64.000 Å³

Setting up phonon calculation...
Supercell: 2×2×2
Total atoms in supercell: 40

Calculating phonon band structure and DOS...
✓ Phonon calculation completed successfully!

Output files in 'phonon_multibinit/':
  band.yaml                - Phonon band structure
  total_dos.dat            - Density of states
  phonon_band.pdf          - Band structure plot
  phonon_dos.pdf           - DOS plot
  force_constants.hdf5     - Force constants

Physical Interpretation for BaHfO3:
----------------------------------
1. Soft modes near Γ point indicate ferroelectric instability
2. Acoustic branches start at zero (long-wavelength sound)
3. Optical branches: Ba/Ti (low), O (high frequency)
```

### Python API Usage

```python
from ase.io import read
from atomchain.init_model import init_calc
from atomchain.phonon import phonon_with_ml

# Load structure
atoms = read('BaHfO3_relaxed.vasp')

# Initialize calculator
calc = init_calc(model_type="multibinit", model_path="BaHfO3_config.conf")

# Calculate phonons
phonon_with_ml(
    atoms=atoms,
    calc=calc,
    supercell=[2, 2, 2],
    displacement=0.01,
    npoints=100,
    use_seek_path=True,
    save_folder='phonon_output'
)
```

## Configuration File Format

MULTIBINIT requires a configuration file specifying potential parameters:

```ini
[files]
# Required: DDB and XML files from ABINIT
ddb_file = system_DDB
sys_file = system.xml

[parameters]
# Supercell dimensions [nx, ny, nz]
ncell = 3 3 3

# Q-point grid (must match DDB)
ngqpt = 4 4 4

# Enable dipole-dipole interactions
dipdip = 1

[backend]
# Use Angstrom/eV for ASE compatibility
use_atomic_units = false

# Automatic atom matching
auto_match_atoms = true
```

**Key parameters:**

- `ncell`: Supercell size for simulation (larger = more accurate but slower)
- `ngqpt`: Q-point grid used in DFT (must match your DDB file)
- `dipdip`: Enable (1) or disable (0) long-range electrostatic interactions
- `use_atomic_units`: false for ASE (Angstrom/eV), true for Bohr/Hartree
- `auto_match_atoms`: Handles atom ordering and PBC automatically

## CLI Usage

AtomChain's CLI tools (`mlrelax`, `mlphonon`) work best with calculators that don't require configuration files (CHGNet, MACE, etc.). For MULTIBINIT, **use the Python API** as shown in the examples above.

**Why?** MULTIBINIT requires:
- Configuration file with DDB/XML paths
- Supercell parameters matching your DFT calculation
- System-specific settings

These are better handled through explicit Python scripts.

**CLI examples for other calculators:**
```bash
# These work with simple calculators:
mlrelax structure.vasp --calc chgnet
mlphonon structure.vasp --calc mace --supercell 2 2 2
```

## Real-World Applications

### 1. Ferroelectric Phase Transitions
Study temperature-dependent phases and transition mechanisms:
```python
# Relax at different temperatures using MD
# Calculate phonons to identify soft modes
# Analyze order parameters (Ti displacement)
```

### 2. Domain Wall Structure
Optimize domain wall configurations:
```python
# Create supercell with domain wall
# Relax with fixed boundaries
# Analyze polarization profile
```

### 3. Defect Calculations
Study point defects and impurities:
```python
# Create supercell with vacancy/substitution
# Relax to find stable configuration
# Calculate formation energies
```

### 4. Thermodynamic Properties
Use phonon results for finite-temperature properties:
```bash
# After phonon calculation:
phonopy -t --dim="2 2 2" band.yaml
# Calculate free energy, heat capacity, entropy
```

## Advantages of MULTIBINIT over ML Potentials

| Feature | MULTIBINIT | ML Potentials |
|---------|-----------|---------------|
| **Physics** | Based on DFT | Data-driven |
| **Anharmonicity** | ✓ Explicit | ✗ Limited |
| **Long-range** | ✓ Dipole-dipole | ✗ Short-range |
| **Transferability** | System-specific | General |
| **Speed** | Fast | Very fast |
| **Training** | DFT + fitting | Large datasets |
| **Accuracy** | High for target system | Variable |

**Use MULTIBINIT when:**
- You have ABINIT DFT calculations
- Studying ferroelectrics/polar materials
- Need anharmonic effects
- Long-range interactions matter
- System-specific accuracy is critical

**Use ML potentials when:**
- Exploring unknown systems
- Need general transferability
- No DFT data available
- Speed is paramount

## Troubleshooting

### ImportError: No module named 'pymultibinit'
```bash
pip install -e /Users/hexu/projects/abinit_git/pymultibinit_dev/pymultibinit
```

### FileNotFoundError: Configuration file not found
- Check config file path is correct
- Ensure config file exists in working directory
- Use absolute paths if needed

### FileNotFoundError: DDB or XML not found
- Generate these files using ABINIT first
- Check paths in configuration file
- Paths are relative to config file location

### Calculator initialization fails
- Verify pymultibinit >= 0.2.0: `python -c "import pymultibinit; print(pymultibinit.__version__)"`
- Check libabinit.so is accessible
- Review error messages for specific issues

### Phonon calculation gives imaginary frequencies
- This is **expected** for unstable phases (e.g., cubic BaHfO3)
- Imaginary frequencies indicate soft modes
- These drive phase transitions to stable structures

## Additional Resources

**AtomChain Documentation:**
- [Relaxation Guide](../docs/relax.md)
- [Phonon Guide](../docs/phonon.md)

**ABINIT Resources:**
- [MULTIBINIT Documentation](https://docs.abinit.org/topics/LatticeModel/)
- [Lattice Model Tutorial](https://docs.abinit.org/tutorial/lattice_model/)
- [DFPT Tutorial](https://docs.abinit.org/tutorial/rf1/)

**pymultibinit:**
- API Reference: `/path/to/pymultibinit/docs/api/`
- Configuration Guide: `/path/to/pymultibinit/docs/CONFIG_FILE_USAGE.md`

## Citation

If you use MULTIBINIT in your research, please cite:

```
@article{gonze2020abinit,
  title={The ABINIT project: Impact, environment and recent developments},
  author={Gonze, Xavier and others},
  journal={Computer Physics Communications},
  volume={248},
  pages={107042},
  year={2020}
}
```

For AtomChain:
```
# Add AtomChain citation when available
```

## Contact & Support

- AtomChain issues: Create issue in repository
- MULTIBINIT questions: ABINIT forum
- pymultibinit issues: Contact maintainer

---

**Last updated:** December 2024  
**AtomChain version:** 0.1.1+  
**pymultibinit version:** 0.2.0+
