# Metastable State Exploration

`mlmetastable` explores candidate metastable structures by following imaginary phonon modes of a high-symmetry parent structure.

The workflow is designed for systems such as cubic perovskites, where unstable phonon modes indicate symmetry-lowering distortions that may relax into local minima.

## Workflow

`mlmetastable` performs this sequence:

1. Compute phonons for the parent structure with a finite-displacement supercell.
2. Identify imaginary phonon modes and label them with symmetry information when available.
3. Generate single-mode and multi-mode distortions up to `--nmax` modes.
4. Enumerate order-parameter directions for each unstable mode combination.
5. Build commensurate supercells for finite-q modes, subject to `--max-cell-size`.
6. Run a single-point force calculation on each distorted structure. If the maximum force is above `10 eV/Å`, halve the distortion amplitude and repeat the single-point check until the force is acceptable.
7. Relax each screened distorted structure with the selected ML calculator, retrying failed relaxations at smaller amplitudes.
8. Record successful and failed candidates, including failed symmetry detection or relaxation details.
9. Group equivalent relaxed results by space group, atom count, and energy tolerance while keeping every generated candidate listed.
10. Write machine-readable and human-readable reports, relaxed structures, checkpoints, and summary plots.
11. Optionally compute phonons for each relaxed metastable structure with `--compute-phonons`.

## CLI Usage

Quick run with the default `chgnet` calculator:

```bash
mlmetastable POSCAR -o metastable_results
```

Run with MACE R2SCAN, pairwise mode combinations, and a 2x2x2 phonon supercell:

```bash
mlmetastable POSCAR \
  --model mace-r2scan \
  --nmax 2 \
  --phonon-ndim 2 2 2 \
  --amplitude 0.5 \
  --fmax 0.01 \
  --output-dir metastable_results
```

More exhaustive run with triplet combinations and phonon stability checks for the relaxed structures:

```bash
mlmetastable POSCAR \
  --model mace-r2scan \
  --nmax 3 \
  --phonon-ndim 2 2 2 \
  --max-cell-size 80 \
  --compute-phonons \
  --fmax 0.01 \
  --output-dir metastable_results
```

## Options

- `fname` - Input high-symmetry parent structure readable by ASE.
- `--model, -m` - Calculator name passed to `init_calc()`. Common choices are `chgnet`, `mace`, `mace-r2scan`, `matgl`, and `m3gnet`.
- `--model-path` - Optional custom model or config path.
- `--nmax` - Maximum number of imaginary modes to combine. Use `1` for single modes, `2` for pairs, or `3` for triplets.
- `--amplitude` - Initial distortion amplitude in Angstrom. Default: `0.5`.
- `--max-cell-size` - Maximum number of atoms allowed in generated commensurate supercells. Default: `64`.
- `--output-dir, -o` - Directory for final reports, structures, and plots. Default: `metastable_results`.
- `--phonon-ndim` - Parent phonon finite-displacement supercell dimensions. Default: `2 2 2`.
- `--fmax` - Force convergence threshold for relaxations in eV/Angstrom. Default: `0.001`.
- `--compute-phonons` - Also compute phonon band structures for each relaxed metastable structure. This is substantially slower.

## Outputs

The final output directory contains:

- `report.yaml` - Machine-readable summary with parameters, parent structure information, imaginary modes, relaxed space groups, energies, and source modes.
- `report.md` - Markdown report with tables and embedded plot references.
- `checkpoint.pkl` - Restart checkpoint loaded by default on later runs in the same output directory.
- `energy_bar_chart.png` - Energy ranking of relaxed metastable states as `delta_e_per_fu`.
- `phonon_band_structure.png` - Parent phonon band plot when plotting succeeds.
- `initial_structures/initial_NNN.vasp` - Initial distorted structures before relaxation.
- `initial_structures/README.md` - Mapping from initial structure files to report rows, mode labels, source modes, and supercells.
- `relaxed_structures/relaxed_NNN.vasp` - Relaxed metastable structures in VASP format.
- `relaxed_structures/README.md` - Mapping from relaxed structure files to report rows and relaxed equivalence groups.
- `metastable_NNN_phonon.png` - Optional phonon plots for relaxed structures when `--compute-phonons` is used.

Important fields in `report.yaml` include:

- `ase_kpoints` - Special q-points from the ASE band path used by the phonon-band plotting workflow.
- `bcs_kpoints` - Special q-points returned by the BCS/symphon labeling workflow, in the input reciprocal basis.
- `bcs_labeled_modes` - Full symphon mode-label table for every q-point in `bcs_kpoints`, including band indices, frequencies, BCS labels, and Mulliken labels when available.
- `imaginary_modes` - Unstable parent modes with `kpoint_label`, `band_index`, `frequency_THz`, `bcs_label`, and `degeneracy` when available.
- `results[].combined_label` - Mode and order-parameter label, such as `R5-(a,a,0)`.
- `results[].status` - `success` or `failed` for the candidate row.
- `results[].error_stage` and `results[].error_message` - Failure stage and details when `status` is `failed`.
- `results[].attempt` - Relaxation attempt that produced the recorded row; failed relaxations retry up to three times.
- `results[].amplitude` - Distortion amplitude used for the recorded attempt. Retries use half and quarter of the requested amplitude.
- `results[].pre_relax_max_force` - Maximum force from the single-point screening calculation before relaxation.
- `results[].force_screen_reductions` - Number of amplitude halvings applied before relaxation because the pre-relaxation force exceeded `10 eV/Å`.
- `results[].delta_e_per_fu` - Energy relative to the parent, in eV per formula unit.
- `results[].initial_spacegroup` - Space group of the distorted structure before relaxation.
- `results[].relaxed_spacegroup` - Space group after relaxation.
- `results[].initial_structure_file` - Relative path to the initial distorted structure file.
- `results[].relaxed_structure_file` - Relative path to the relaxed structure file.
- `results[].relaxed_group_id` - Group ID for equivalent relaxed structures.
- `results[].relaxed_group_size` - Number of generated candidates in the same relaxed group.
- `results[].relaxed_group_representative` - Whether this row is the representative of its relaxed group.
- `results[].supercell_matrix` - Commensurate supercell matrix used for the distortion.
- `relaxed_structure_groups` - Group summaries with representative IDs and member IDs. Equivalent structures are grouped in the Markdown report table, but all generated candidates remain listed.

The Markdown report places the ASE and BCS/symphon q-point lists near the beginning. This is intentional because the phonon plot and symmetry-labeling backend can use different special-point catalogs. For example, some path-specific ASE/SeeK-path points may have imaginary phonons even when they are absent from the BCS/symphon list.

## BaTiO3 Example

Create a cubic BaTiO3 parent structure as `BaTiO3_cubic.vasp`:

```text
BaTiO3 cubic Pm-3m
1.0
4.00 0.00 0.00
0.00 4.00 0.00
0.00 0.00 4.00
Ba Ti O
1 1 3
Direct
0.000000 0.000000 0.000000
0.500000 0.500000 0.500000
0.500000 0.500000 0.000000
0.500000 0.000000 0.500000
0.000000 0.500000 0.500000
```

Run a practical BaTiO3 search with MACE R2SCAN:

```bash
mlmetastable BaTiO3_cubic.vasp \
  --model mace-r2scan \
  --nmax 2 \
  --phonon-ndim 2 2 2 \
  --max-cell-size 64 \
  --amplitude 0.5 \
  --fmax 0.01 \
  --output-dir batio3_metastable
```

Inspect the ranked structures:

```bash
uv run python - <<'PY'
import yaml

with open("batio3_metastable/report.yaml") as handle:
    report = yaml.safe_load(handle)

successful = [row for row in report["results"] if row["status"] == "success"]
for row in sorted(successful, key=lambda r: r["delta_e_per_fu"]):
    print(
        f'{row["id"]:03d} '
        f'{row["delta_e_per_fu"]: .4f} eV/FU '
        f'{row["relaxed_spacegroup"]:16s} '
        f'{row["combined_label"]}'
    )
PY
```

Then open `batio3_metastable/report.md` and `batio3_metastable/energy_bar_chart.png` to review the symmetry labels, relaxed space groups, and energy ordering.

Use the structure directory READMEs to connect report rows to files:

```bash
less batio3_metastable/initial_structures/README.md
less batio3_metastable/relaxed_structures/README.md
```

For a stability check of the relaxed candidates, rerun with `--compute-phonons`:

```bash
mlmetastable BaTiO3_cubic.vasp \
  --model mace-r2scan \
  --nmax 2 \
  --phonon-ndim 2 2 2 \
  --max-cell-size 64 \
  --amplitude 0.5 \
  --fmax 0.01 \
  --compute-phonons \
  --output-dir batio3_metastable_phonons
```

## Python API

Use `explore_metastable_states()` directly when you need access to the raw result dictionaries:

```python
from ase.io import read

from atomchain import explore_metastable_states

atoms = read("BaTiO3_cubic.vasp")

data = explore_metastable_states(
    atoms,
    calc="mace-r2scan",
    nmax=2,
    amplitude=0.5,
    max_cell_size=64,
    phonon_ndim=[2, 2, 2],
    relax_kwargs={"fmax": 0.01},
    output_dir="batio3_metastable_api",
)

for result in data["results"]:
    print(
        result["combined_label"],
        result["spacegroup_name"],
        result["delta_e_per_fu"],
    )
```

To create the same report files as the CLI, call `generate_report()`:

```python
from atomchain.metastable_report import generate_report

generate_report(
    data,
    atoms,
    "mace-r2scan",
    {
        "nmax": 2,
        "amplitude": 0.5,
        "max_cell_size": 64,
        "fmax": 0.01,
        "phonon_ndim": [2, 2, 2],
    },
    "batio3_metastable_api",
)
```

## Practical Notes

- Start with `--nmax 1` or `--nmax 2`; `--nmax 3` can grow quickly because it enumerates more mode combinations and OPDs.
- Increase `--max-cell-size` only when important finite-q instabilities are skipped.
- `--amplitude 0.3` to `0.8` Angstrom is usually a reasonable scan range; too small can relax back to the parent, while too large may create unphysical initial structures.
- Interrupted runs continue from `checkpoint.pkl` in the output directory by default when the checkpoint metadata matches the current run parameters. Remove that file or use a new output directory to force a fresh candidate search.
- Failed rows remain in `report.yaml` and `report.md` with `status: failed` so a single bad candidate does not hide the rest of the search.
- `mace-r2scan` downloads the MACE-MH-1 model automatically to `~/.config/mace/mace-mh-1.model` when missing. If automatic download fails, download `https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model` manually and save it at that path.
- `--compute-phonons` is best used after a first pass identifies a smaller set of interesting low-energy candidates.
