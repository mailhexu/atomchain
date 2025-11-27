# Test Fixtures Directory

This directory contains structure files used for testing.

## Required Structure Files

Please add small structure files here for testing. The test fixtures will automatically load them.

### Recommended Files:

1. **Small crystal structure** (e.g., `test_crystal.vasp` or `test_crystal.cif`)
   - A small crystal (e.g., 2-8 atoms)
   - Used for relaxation and phonon tests
   - Any ASE-readable format works

2. **Simple molecule** (optional, e.g., `test_molecule.xyz`)
   - Small molecule structure
   - Used for quick tests

### Supported Formats:

- VASP: `.vasp`, `POSCAR`
- CIF: `.cif`
- XYZ: `.xyz`
- Any format readable by ASE `read()` function

### Example:

```bash
# Copy a small structure from examples or create one
cp ../examples/01_relax/Cc_minimal.vasp fixtures/test_structure.vasp
```

## Notes:

- Keep structures small for fast testing
- The `test_structure` fixture in `conftest.py` will automatically find and load these files
- Tests will be skipped if no structure files are found
