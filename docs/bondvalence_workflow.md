# Bond-Valence Model Workflow (`mlbvmodel`)

Fits conventional bond-valence parameters (R0, b per cation–anion pair) from a
parent structure using ML-potential-generated structures, with held-out
validation and a reproducible report. Chemistry and fitting live in the
separate `easybondvalence` package (install with `pip install
'atomchain[bondvalence]'`); atomchain provides the orchestration.

## What it does

1. **relax** — the parent structure is relaxed with the configured calculator
   (default MACE via `init_calc`).
2. **sample** — rattled frames per amplitude plus an isotropic strain grid,
   labeled by structure group (`relaxed`, `rattle-a{k}`, `strain`).
3. **evaluate** — single-point energies filter frames to an energy window per
   atom. Energies never enter the fit residual.
4. **fit** — frames become an `easybondvalence` fit dataset, split into
   fit/validation by whole structure groups; `fit_parameters` fits R0/b
   against |z| targets.
5. **validate** — unweighted held-out GII (RMS ΔV) against a configurable
   threshold (default 0.2 v.u.); identifiability diagnostics (Jacobian rank,
   conditioning, covariance, active bounds) are reported but never gate.
6. **report** — markdown/YAML report with fitted-vs-seed-vs-literature
   comparison and valence multipole descriptors on the relaxed parent
   (descriptors only — no energy model).

## Seed file

All chemistry comes from one required YAML file (no parameter database is
bundled). See `examples/08_bondvalence_workflow_batio3/seed.yaml`:

```yaml
provenance: "source citation and applicability"
oxidation_states: {Ba: 2, Ti: 4, O: -2}
contacts: {cutoff: 12.0}
pairs:
  - selector: [Ba, O]
    r0: {initial: 2.285, lower: 1.8, upper: 3.0}
    b: {initial: 0.37, lower: 0.2, upper: 0.6, transform: positive}
literature:            # optional, report-only comparison
  Ba-O: {r0: 2.285, b: 0.37}
```

Every schema violation is reported together before any expensive calculation
runs. Ties use easybondvalence's key convention `"Identifier:field"`
(e.g. `tie: "Ba-O:b"`). A large cutoff requires a parameter for **every**
within-cutoff pair family; represent like-charged pairs (O–O, cation–cation)
with fixed near-zero pad parameters — see the BaTiO3 example README for the
full reasoning and the b-fixing identifiability discussion.

## CLI

```bash
mlbvmodel POSCAR --seed-file seed.yaml --output-dir bv \
  --rattle-amplitudes 0.03 0.06 0.10 --frames-per-amplitude 20 \
  --strain-range -0.03 0.03 --strain-points 7 --energy-window 0.05 \
  --gii-threshold 0.2 --seed 1234
```

`--config workflow.yaml` accepts the same keys; explicit flags override YAML
values. `--dry-run`, `--force-stage <name>`, `--force-all`, and
`--stop-after <stage>` behave exactly as in `mlmbmodel`. Completed stages are
reused unless the configuration, the structure geometry, or the **content**
of the seed file changes.

## Minimal API (fit, save, reload, evaluate)

Three calls, everything else defaulted — see
`examples/09_bondvalence_minimal/`:

```python
from atomchain.bondvalence_workflow import (
    evaluate_bond_valence,
    load_bond_valence_model,
    run_bondvalence_workflow,
)

result = run_bondvalence_workflow(structure="POSCAR", seed_file="seed.yaml")

model = load_bond_valence_model(result.fitted_parameters)

evaluation = evaluate_bond_valence(model, atoms)
# evaluation: gii, site_sums, mismatches, per_site, dipoles, quadrupoles
# (v.u.), normalized coefficients and invariants (dimensionless).
```

The saved `fitted_parameters.yaml` (`atomchain-bv-model-v1`) is
self-contained: fitted pairs, the resolved contact cutoff, oxidation states,
dataset fingerprint, and provenance.

## Python API (full configuration)

```python
from atomchain.bondvalence_workflow import (
    BVWorkflowConfig, RelaxStageConfig, BVSamplingStageConfig,
    FitStageConfig, BVValidationStageConfig, run_bondvalence_workflow,
)

result = run_bondvalence_workflow(
    BVWorkflowConfig(
        structure="POSCAR",
        seed_file="seed.yaml",
        output_dir="bv",
        sampling=BVSamplingStageConfig(
            rattle_amplitudes=(0.03, 0.06, 0.10),
            frames_per_amplitude=20,
        ),
    )
)
print(result.passed, result.fitted_parameters)
```

## Choosing the contact cutoff

The cutoff is a formal device, not physics — the exponential decays fast
(b = 0.37 A: a bond 1 A beyond R0 contributes ~7e-3 v.u.). It must satisfy:

1. **Boundary stability**: keep every non-negligible shell away from the
   cutoff by more than the ensemble spread `m ~ 3*sqrt(2)*rattle_sigma +
   eps_max * d` (rattle displacements plus strain). Contacts flickering at
   the boundary across frames makes the effective model frame-dependent —
   the classic instability. Flicker is harmless only for shells with
   negligible valence (~1e-13 v.u.).
2. **Coverage**: every pair family inside the cutoff needs a parameter
   (`evaluate()` raises for uncovered contacts). Like-charged pairs get
   fixed near-zero pads.
3. **Degeneracy**: symmetry can force two families to the same distance
   (cubic perovskites: Ba-O and O-O both a/sqrt(2)); parameters, not the
   cutoff, must then arbitrate.
4. **Identifiability**: outer shells (~1e-5 v.u.) cannot pin b regardless
   of cutoff; b is constrained by the distance spread inside the active
   shell (hence the strain scan, or fix b at the tabulated value).

Practical rule: list shell distances and valences at the seed values, then
place the cutoff in a shell gap clearing `m` on both sides — either the
minimal first gap (fewest families) or a large ~3a cutoff (everything
negligible included, pads for the rest; most robust). Verify by refitting
at two cutoffs within the same gap: fitted values must not move.

## Interpretation

BVS and GII are structural diagnostics in valence units — not energies and
not a stability proof. A passing gate means the fitted parameters reproduce
|z| site sums on unseen structures within the stated contact policy and
oxidation assignment. Check the identifiability block before trusting the
numbers: parameters at active bounds or a large condition number mean the
sampled ensemble does not constrain them.

## Example

See `examples/08_bondvalence_workflow_batio3/` for a complete BaTiO3 run with
MACE, including the expected output report and caveats.
