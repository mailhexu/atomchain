# MULTIBINIT Model Workflow

`atomchain.multibinit_workflow` provides the `mlmbmodel` command for orchestrating a full MULTIBINIT-compatible model-building workflow.

The workflow stages are:

1. Prepare the parent structure and configuration.
2. Generate a DDB with phonon, elastic, and optional internal-strain blocks, then compare DDB-derived phonon bands against the phonopy finite-difference cache when available.
3. Optionally run metastable structure exploration.
4. Generate train/test frames from random and metastable-derived samplers.
5. Evaluate frames with the reference calculator and write trajectories plus HIST files.
6. Train through a configurable pymultibinit backend, defaulting to `python`.
7. Validate the fitted model and write metrics/plots.
8. Write final `report.md` and `report.yaml` files with relative artifact links.

The workflow is checkpointed. Each stage writes `<output>/<stage>/stage.yaml`, and `manifest.yaml` records all completed outputs. Reruns reuse completed stages when the stage signature matches. Use `--force-stage <stage>` to rerun that stage and all downstream stages, or `--force-all` to rebuild everything.

## CLI

```bash
mlmbmodel POSCAR \
  --model mace-r2scan \
  --output-dir model_workflow \
  --phonon-supercell 2 2 2 \
  --qgrid 2 2 2 \
  --metastable \
  --train-size 100 \
  --test-size 20 \
  --training-backend python \
  --force-rmse-threshold 0.01
```

Use `--dry-run` to print the stage plan without running expensive calculations.

Useful stage controls:

```bash
# Stop after generating evaluated train/test HIST files.
mlmbmodel POSCAR --output-dir model_workflow --stop-after evaluate

# Refit, revalidate, and regenerate the report from existing inputs.
mlmbmodel POSCAR --output-dir model_workflow --force-stage train

# Regenerate only the final report from existing manifest data.
mlmbmodel POSCAR --output-dir model_workflow --force-stage report
```

## Python API

```python
from atomchain.multibinit_workflow import WorkflowConfig, run_model_build_workflow

result = run_model_build_workflow(
    WorkflowConfig(
        structure="POSCAR",
        model="mace-r2scan",
        output_dir="model_workflow",
    )
)
print(result.report_md)
```

The default validation gate is test-set force RMSE <= `0.01 eV/Angstrom`.

## Outputs

The output directory contains both machine-readable data and a human-readable report:

- `manifest.yaml`: stage metadata, configuration, signatures, outputs, warnings, and errors.
- `report.yaml`: full manifest snapshot used by the final report.
- `report.md`: procedure, compact configuration summary, full sanitized input parameters, stage status, artifact links, data tables, complete metastable exploration details, validation metrics, and embedded figures.
- `ddb/model.ddb`: finite-difference DDB with phonon, elastic, and internal-strain blocks when enabled.
- `ddb/phonon_band_comparison.png` and `ddb/phonon_band_comparison.json`: best-effort DDB-vs-phonopy phonon-band comparison generated from the DDB and `ddb/fd_cache/phonon_base/phonopy_params.yaml`.
- `ddb/harmonic_validation.json`: tiny-displacement/strain validation comparing the DDB-only pyeffpot harmonic response against the source calculator, including elastic/internal-strain presence checks and force/stress response errors.
- `sampling/train_raw.traj` and `sampling/test_raw.traj`: sampled train/test frames before reference evaluation.
- `training/train.traj`, `training/test.traj`, `training/train_HIST.nc`, and `training/test_HIST.nc`: reference-evaluated training data.
- `model/basis.xml`, `model/fitted.xml`, and `model/config.conf`: generated basis, fitted coefficients, and a MULTIBINIT-style config artifact.
- `validation/metrics.json`: numerical validation metrics and pass/fail status.
- `validation/*.png`: real validation figures generated from reference and model-evaluated trajectories.

## Validation Figures

The DDB stage also attempts to generate `ddb/phonon_band_comparison.png`, overlaying phonopy finite-difference bands with DDB/pyeffpot bands along an ASE high-symmetry path. The companion JSON includes absolute difference summary statistics in `cm^-1`. If the optional parser/plotting path is unavailable, the workflow records a DDB-stage warning and continues.

The comparison uses the same harmonic toggles as pure-Python validation: `training.options.dipdip` and `training.options.asr`. The workflow default is `dipdip: 1` and `asr: 0`, preserving raw generated-DDB/phonopy parity unless ASR is explicitly requested.

The DDB stage also validates the harmonic model on a few tiny randomly displaced and strained structures. The validation compares response relative to the parent structure, so residual source-model forces or reference stress offsets do not mask whether the DDB harmonic/elastic response works. The report shows whether parsed elastic constants and internal-strain couplings are present and summarizes response-energy, force, and stress errors.

Validation figures are generated from `training/test.traj` and `validation/model_test.traj`:

- `energy_parity.png`: reference energy versus fitted-model energy.
- `force_parity.png`: reference force components versus fitted-model force components.
- `stress_parity.png`: reference stress versus fitted-model stress when stress data are available.
- `frame_error_trends.png`: per-frame absolute energy error and max force error.
- `rmse_summary.png`: RMSE summary for energy, forces, and stress.

For `training.backend: python`, validation uses pymultibinit's pure-Python `pyeffpot` path with the generated DDB and XML coefficients. It does not load `libabinit.so`. The generated `model/config.conf` is kept as an artifact for downstream tools and reproducibility.

## Calculator Notes

MACE calculators backed by CUDA should not be forked into phonon worker processes. The workflow passes `phonon_kwargs: {parallel: false}` to DDB phonon generation by default. If the local CUDA/NVRTC installation is incomplete, run the command on CPU:

```bash
CUDA_VISIBLE_DEVICES="" mlmbmodel --config workflow.yaml
```

## Example

See `examples/07_multibinit_workflow_batio3/` for a BaTiO3 `workflow.yaml`, POSCAR, and dry-run script.

That example includes a generated `batio3_multibinit_model/report.md` after a full CPU run. The report intentionally shows validation results from a small demonstration dataset; it may fail the strict default force-RMSE threshold and should not be interpreted as a production-quality fitted model.

## Training-Set Sources

The sample stage keeps the original `include_random` and `include_metastable` switches, and also accepts extensible entries under `sampling.sources`. Each source writes frame provenance into `sampling/frame_manifest.yaml`; downstream evaluation, HIST writing, training, validation, and reporting use the same paths regardless of source.

```yaml
sampling:
  supercell: [2, 2, 2]
  include_random: false
  include_metastable: false
  sources:
    - name: rattle
      count: 20
      stdev: 0.05
      cell_stdev: 0.001
      split: train
    - name: metastable_interpolation
      lambdas: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
      split: train
    - name: contour
      steps: 200
      sample_interval: 10
      maxstep: 0.05
      parallel_drift: 0.1
      split: train
    - name: npt_md
      steps: 500
      sample_interval: 25
      temperature: 300
      pressure: 1.01325
      timestep: 1.0
      split: train
```

Available source names are `rattle`, `metastable_interpolation`, `contour`, and `npt_md`. `rattle` is calculator-free. `contour` and `npt_md` require a reference calculator during sampling because they propagate structures with ASE dynamics. `metastable_interpolation` consumes compatible same-atom-count metastable structures from the metastable stage and can extrapolate beyond the parent-to-metastable segment; the default lambda grid extends to `3.0`.

When `sampling.supercell` is set, evaluated primitive frames are expanded to that fixed supercell before HIST writing, fitting, and validation. Energies are multiplied by the supercell multiplicity, forces are copied by translation symmetry, and stress is unchanged. The Python MULTIBINIT backend uses the same value as the default fitting/evaluation `ncell` unless `training.options.ncell` is set explicitly.

The coefficient-basis enumeration is independent of the HIST supercell. In pymultibinit, `generate_displacement_basis(..., ncell=...)` controls the translated neighbor cells considered while enumerating displacement pair factors, not the number of cells in the HIST trajectory. The workflow therefore uses `training.options.basis_ncell` for coefficient generation, defaulting to `[1, 1, 1]`, while `sampling.supercell`/`training.options.ncell` controls the evaluated model supercell.

Large generated bases should use the Python backend's screened-greedy path instead of dense all-candidate fitting. Set `training.options.selection: screened_greedy` with `candidate_pool_size`, `feature_chunk_size`, and `screening_frame_count` to rank all candidates on a small frame subset, then greedily fit only the retained pool on the full training set. Set `feature_backend: memmap` and `feature_memmap_dir` to keep final pool feature arrays disk-backed. Existing `model/basis.xml` files are reused only when `model/basis_pair_diagnostics.json` contains a matching basis-generation fingerprint for the current parent structure, cutoff, `basis_ncell`, symmetry settings, power range, and strain-coupling setting. Python-fit diagnostics in manifests intentionally omit full coefficient arrays for large bases; fitted values remain in `model/fitted.xml`, while the manifest records coefficient counts and selected nonzero indices.

Validation can also compare representative metastable energies by setting `validation.metastable_energy_differences: true`. If `validation.metastable_records` points at a workflow manifest, successful metastable records are grouped by relaxed space group and an energy bucket (`validation.metastable_energy_tol`, default `1e-3 eV/FU`), one representative per group is converted to the validation supercell when compatible, and the report includes a source-vs-model energy table with the recorded source and validation supercell matrices.

The Markdown report also includes a full `Input Parameters` table generated from the sanitized workflow configuration, including nested `ddb`, `metastable.options`, `sampling.sources`, `training.options`, and `validation` values. Sensitive keys such as tokens, passwords, credentials, API keys, and private keys are redacted before writing artifacts. When metastable exploration is enabled, `report.md` lists every candidate record rather than only the lowest-energy subset, including status, attempt, amplitude, force-screening reductions, source-mode labels/frequencies, OPD labels, polarization direction, supercell matrices, grouping metadata, structure paths, and relaxation trajectory paths.

Contour exploration follows ASE's `ContourExploration` dynamics and is not useful exactly at a strict local minimum unless the structure is first perturbed. NPT sampling depends on stable stress predictions and conservative timestep/barostat settings.
