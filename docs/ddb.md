# DDB Writer

`atomchain.ddb` writes a conservative ABINIT-style DDB text subset from phonopy and finite-difference ML workflows. ABINIT, ANADDB, AbiPy, and pymultibinit are not runtime dependencies.

## Supported Subset

- Structure metadata: lattice, reduced positions, species, masses, and spglib symmetry metadata in the text DDB and YAML sidecar.
- Second-order displacement-displacement blocks from phonopy dynamical matrices on arbitrary full q-point grids.
- Optional stress/strain finite-difference blocks from ASE calculators with stress support.
- Optional Gamma displacement-strain internal-strain blocks from finite differences of forces under homogeneous strain.
- Deterministic text output plus `*.ddb.yaml` sidecar metadata.

This is not a full ANADDB replacement and does not write electric-field, Born-charge, dielectric, Raman, or long-wave third-order data.

Finite-q strain dependence of phonons, `d Phi(q) / d strain`, is a third-order response and is not written as an ordinary second-order DDB displacement-strain block.

## Conventions

ABINIT DDB second-order data lines use one-based perturbation pairs:

```text
idir1 ipert1 idir2 ipert2 real_value imag_value
```

Mappings used by atomchain:

- Atomic displacement: `ipert=1..natom`, `idir=1..3`.
- Diagonal strain: `ipert=natom+3`, `idir=1,2,3` for `xx, yy, zz`.
- Shear strain: `ipert=natom+4`, `idir=1,2,3` for `yz, xz, xy`.
- q-points are reduced reciprocal coordinates.
- Homogeneous strain perturbations are Gamma-only in the second-order DDB convention used by ABINIT.
- ABINIT reads internal strain at Gamma from displacement-strain second derivatives and uses `instrain = -blkval`.
- ABINIT reads elastic constants at Gamma from strain-strain second derivatives and divides by cell volume.
- Lengths are converted from Angstrom to Bohr.
- Energies are converted from eV to Hartree.
- Force constants are converted from `eV/Angstrom^2` to `Ha/Bohr^2`.
- Stress-like quantities are converted from `eV/Angstrom^3` to `Ha/Bohr^3`.

## CLI Examples

Phonon-only DDB from an existing phonopy calculation:

```bash
mlddb BaTiO3.vasp --phonopy-yaml phonon_save/phonopy_params.yaml --qgrid 2 2 2 --output BaTiO3.ddb --validate
```

Finite-difference stress/elastic blocks with MACE-r2scan:

```bash
mlddb BaTiO3.vasp --model mace-r2scan --include-stress --phonon-ndim 2 2 2 --qgrid 2 2 2 --output BaTiO3_fd.ddb
```

Full q-grid phonon plus Gamma elastic/internal-strain workflow:

```bash
mlddb BaTiO3.vasp --model mace-r2scan --include-stress --include-strain-phonon --phonon-ndim 2 2 2 --qgrid 2 2 2 --strain-amplitude 1e-3 --cache .tmp/batio3_ddb_cache --output BaTiO3_full.ddb
```

## Python API

```python
from ase.io import read
from atomchain.ddb import write_ddb_from_phonopy

atoms = read("BaTiO3.vasp")
write_ddb_from_phonopy(
    atoms=atoms,
    phonopy_yaml="phonon_save/phonopy_params.yaml",
    filename="BaTiO3.ddb",
    qgrid=(2, 2, 2),
)
```

Finite-difference response with MACE-r2scan:

```python
from ase.io import read
from atomchain.init_model import init_calc
from atomchain.ddb import write_ddb_from_finite_difference

atoms = read("BaTiO3.vasp")
calc = init_calc("mace-r2scan")
write_ddb_from_finite_difference(
    atoms,
    calc,
    filename="BaTiO3_fd.ddb",
    phonon_ndim=(2, 2, 2),
    qgrid=(2, 2, 2),
    include_stress=True,
    include_strain_phonon=True,
    cache_dir=".tmp/batio3_ddb_cache",
)
```

`qgrid` defaults to the diagonal `phonon_ndim` grid for finite-difference workflows. A `(2, 2, 2)` grid writes eight q-point blocks with components `0` and `0.5`.

`include_strain_phonon` is retained as the CLI/API flag for compatibility, but the DDB writer only emits the ABINIT-compatible Gamma internal-strain second-order blocks. It does not emit finite-q `dPhi(q)/dstrain` terms.

## Validation

Every DDB write also writes a YAML sidecar. Validate it with:

```python
from atomchain.ddb import validate_ddb_metadata

errors = validate_ddb_metadata("BaTiO3.ddb.yaml")
assert not errors
```
