# ABINIT HIST and MULTIBINIT Training Artifacts

atomchain can prepare ABINIT-oriented training inputs from ASE workflows:

- `HIST.nc` files with structures, total energies, forces, and stress.
- ABINIT-style DDB files from existing phonopy calculations.
- Training trajectories generated from MD-like sampling, phonon-mode displacements, metastable structures, and metastable linear combinations.
- A thin wrapper that delegates actual MULTIBINIT model training to `pymultibinit`.

## HIST Conversion

Convert an evaluated ASE trajectory to ABINIT HIST:

```bash
mlhist training.traj training_HIST.nc --to hist
```

Convert HIST back to an ASE trajectory:

```bash
mlhist training_HIST.nc training_from_hist.traj --to traj
```

Python API:

```python
from atomchain.io.hist import read_abinit_hist, write_abinit_hist

frames = read_abinit_hist("training_HIST.nc")
write_abinit_hist(frames, "roundtrip_HIST.nc")
```

## Units

The supported HIST subset uses ABINIT atomic units internally:

| Quantity | ASE unit | HIST unit |
|----------|----------|-----------|
| lattice `rprimd` | Angstrom | Bohr |
| energy `etotal` | eV | Hartree |
| forces `fcart` | eV/Angstrom | Hartree/Bohr |
| stress `strten` | eV/Angstrom^3 | Hartree/Bohr^3 |

Stress uses Voigt order `(xx, yy, zz, yz, xz, xy)`.

## Training Trajectory Generation

Generate and evaluate a trajectory:

```bash
mltraining generate BaTiO3.vasp \
  --sources md phonon_modes \
  --model mace-r2scan \
  --output training.traj
```

Python API:

```python
from atomchain.training import generate_training_trajectory

frames = generate_training_trajectory(
    "BaTiO3.vasp",
    model="mace-r2scan",
    sources=["md", "phonon_modes"],
    output="training.traj",
)
```

Each generated frame stores provenance in `Atoms.info`, including the source strategy and source-specific parameters.

## DDB + HIST Artifact Bundle

Generate a bundle from an evaluated trajectory and a phonopy calculation:

```bash
mltraining artifacts BaTiO3.vasp \
  --trajectory training.traj \
  --phonopy-yaml phonon_save/phonopy_params.yaml \
  --ddb BaTiO3.ddb \
  --hist BaTiO3_HIST.nc \
  --output-dir training_bundle
```

## Delegated MULTIBINIT Training

Actual MULTIBINIT training execution belongs to `pymultibinit`. atomchain only delegates:

```bash
mltraining train \
  --ddb training_bundle/BaTiO3.ddb \
  --hist training_bundle/BaTiO3_HIST.nc \
  --config training.conf \
  --output-dir multibinit_model
```

`pymultibinit` should implement training by calling the `multibinit` binary. It should not depend on ABINIT library-mode training. If `pymultibinit` does not expose the training API, atomchain raises a setup error. Unit tests mock this API; real MULTIBINIT binary execution should be covered by optional integration tests once available.

## Fitted Model Validation

After training, validate a fitted ASE-compatible model against held-out reference data:

```python
from atomchain.training import validate_fitted_model

result = validate_fitted_model(
    "test_HIST.nc",              # also accepts ASE .traj, Atoms, or list[Atoms]
    calculator="multibinit",
    model_path="BaTiO3_fit_coeffs.xml",
    output="validation.json",
)
print(result.metrics["energy_ev"].mae)
print(result.metrics["forces_ev_ang"].rmse)
```

For metastable-state checks, compare relative energies to the reference high-energy state:

```python
from atomchain.training import validate_metastable_energy_differences

result = validate_metastable_energy_differences(
    high_energy_state,
    metastable_states,
    calculator="multibinit",
    model_path="BaTiO3_fit_coeffs.xml",
)
```

This reports errors in `E(metastable) - E(reference)` between the fitted model and the original/reference single-point data. If the input structures do not already contain reference energies, pass `reference_calculator=` to evaluate them first.
