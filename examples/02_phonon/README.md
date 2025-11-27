# Phonon Calculation Example

This example demonstrates the new phonon workflow after the structure refactoring.

## What's New (Post-Refactor)

1. **New import paths**: `atomchain.phonon.*` instead of `atomchain.*`
2. **Organized output**: All phonon files go to `phonon_save/` directory
3. **ASE k-path generation**: No external dependencies for band structure k-paths

## Files

- `Al_fcc.vasp` - Aluminum FCC crystal structure
- `phonon_example.py` - Complete phonon calculation workflow

## Usage

```bash
cd atomchain/examples/02_phonon
python phonon_example.py
```

## Requirements

```bash
pip install atomchain mace-torch
```

## Output

The script creates:
- `phonon_save/` - Directory with all phonon calculation files
  - `disp.yaml` - Displacement information
  - `FORCE_CONSTANTS` - Force constants matrix
  - `phonon.pickle` - Serialized Phonopy object
  - `phonopy_params.yaml` - Phonopy parameters
  - `PHON_CELL*/` - Displaced structure directories
- `phonon_bands.png` - Phonon band structure plot

## Migration from Old Code

**Before (old import paths):**
```python
from atomchain.frozenphonon import calculate_phonon
from atomchain.plotphonopy import plot_phonon

calculate_phonon(atoms, ...)  # Files scattered in current directory
```

**After (new import paths):**
```python
from atomchain.phonon.frozenphonon import calculate_phonon
from atomchain.phonon.plotphonopy import plot_phonon

calculate_phonon(atoms, phonon_save_dir="phonon_save", ...)  # Organized in subdirectory
```

## Key Changes

1. **Module organization**: All phonon code in `atomchain/phonon/` subdirectory
2. **Output organization**: `phonon_save_dir` parameter (default: `"phonon_save"`)
3. **K-path generation**: Uses ASE's `Cell.bandpath()` instead of pyDFTutils
4. **Cleaner API**: `kpath` parameter uses standard k-point labels (e.g., "GXMG")
