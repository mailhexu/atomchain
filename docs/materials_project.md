# Materials Project Access

AtomChain can fetch structures from the Materials Project through the optional `mp-api` client and convert them to ASE `Atoms` for downstream workflows.

## Installation

Install the optional dependency for Materials Project support:

```bash
uv sync --extra materials-project
```

For package installs, use:

```bash
pip install 'atomchain[materials-project]'
```

## Authentication

Pass an API key explicitly or set `MP_API_KEY`:

```bash
export MP_API_KEY=your_key_here
```

## CLI

Fetch one material by ID:

```bash
mlmp fetch mp-149 --output Si.vasp
```

Fetch all structures matching a space group number or symbol:

```bash
mlmp spacegroup 141 --output-dir sg141 --limit 100
mlmp spacegroup I4_1/amd --output-dir sg141 --format cif
```

Run a generic Materials Project summary query:

```bash
mlmp query --elements Zr Si --energy-above-hull 0 0.05 --output-dir zr_si_stable
mlmp query --chemsys Zr-Si --band-gap 0 1.5 --filter is_stable=true --output-dir zr_si_gap
```

`mlmp query` supports first-class flags plus repeatable generic filters with `--filter key=value`. Comma-separated values are parsed as tuples, for example `--filter band_gap=0,1.5`.

Batch commands write structure files and `manifest.yaml` by default. Existing output files are not overwritten unless `--overwrite` is provided.

## Python API

```python
from atomchain.materials_project import (
    get_structure_by_id,
    get_structure_record_by_id,
    get_structures_by_spacegroup,
    query_structures,
    write_structure_records,
    write_manifest,
)

atoms = get_structure_by_id("mp-149")
record = get_structure_record_by_id("mp-149")

records = get_structures_by_spacegroup(141, limit=100)
records = query_structures(elements=["Zr", "Si"], energy_above_hull=(0, 0.05), limit=100)

written = write_structure_records(records, "zr_si_stable")
write_manifest(written, "zr_si_stable/manifest.yaml", {"command": "query"})
```

`get_structure_by_id()` returns bare ASE `Atoms` for convenient chaining. Use `get_structure_record_by_id()` when material ID, formula, space group, and source metadata are needed.
