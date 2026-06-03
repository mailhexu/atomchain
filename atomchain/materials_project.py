"""Materials Project structure access helpers."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from ase import Atoms
from ase.io import write

DEFAULT_FIELDS = ["material_id", "formula_pretty", "symmetry", "structure"]


@dataclass
class MaterialsProjectStructure:
    """ASE structure plus Materials Project provenance metadata."""

    material_id: str
    formula: str
    atoms: Atoms
    spacegroup_number: int | None = None
    spacegroup_symbol: str | None = None
    structure_file: str | None = None
    source: str = "materials_project"
    metadata: dict[str, Any] = field(default_factory=dict)


def _get_mprester_class():
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError(
            "Materials Project support requires the optional `mp-api` dependency. "
            "Install with `uv sync --extra materials-project` or "
            "`pip install atomchain[materials-project]`."
        ) from exc
    return MPRester


def _make_mprester(api_key: str | None = None):
    key = api_key if api_key is not None else os.environ.get("MP_API_KEY")
    MPRester = _get_mprester_class()
    if key:
        return MPRester(api_key=key)
    return MPRester()


def _to_ase_atoms(structure: Any) -> Atoms:
    if isinstance(structure, Atoms):
        return structure.copy()
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise RuntimeError(
            "Converting Materials Project structures requires pymatgen, normally "
            "installed with `mp-api`. Install with `uv sync --extra materials-project`."
        ) from exc
    return AseAtomsAdaptor.get_atoms(structure).copy()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _symmetry_field(symmetry: Any, name: str) -> Any:
    if symmetry is None:
        return None
    value = _field(symmetry, name)
    if value is not None:
        return value
    aliases = {"number": "space_group_number", "symbol": "space_group_symbol"}
    return _field(symmetry, aliases.get(name, name))


def _record_from_doc(doc: Any) -> MaterialsProjectStructure:
    material_id = str(_field(doc, "material_id", ""))
    formula = _field(doc, "formula_pretty") or _field(doc, "formula") or material_id
    symmetry = _field(doc, "symmetry")
    structure = _field(doc, "structure")
    if structure is None:
        raise ValueError(
            "Materials Project result does not include `structure`; request the structure field before writing atoms."
        )
    metadata = {}
    if not isinstance(doc, dict):
        metadata["document_type"] = type(doc).__name__
    return MaterialsProjectStructure(
        material_id=material_id,
        formula=str(formula),
        atoms=_to_ase_atoms(structure),
        spacegroup_number=_symmetry_field(symmetry, "number"),
        spacegroup_symbol=_symmetry_field(symmetry, "symbol"),
        metadata=metadata,
    )


def _apply_limit(kwargs: dict[str, Any], limit: int | None) -> dict[str, Any]:
    if limit is not None:
        kwargs["num_chunks"] = 1
        kwargs["chunk_size"] = limit
    return kwargs


def _structure_kwargs(structure_type: str) -> dict[str, bool]:
    if structure_type == "final":
        return {"final": True}
    if structure_type == "initial":
        return {"final": False}
    raise ValueError("structure_type must be 'final' or 'initial'.")


def get_structure_by_id(
    material_id: str, api_key: str | None = None, structure_type: str = "final"
) -> Atoms:
    """Return one Materials Project structure as bare ASE ``Atoms``."""

    rester = _make_mprester(api_key)
    structure = rester.get_structure_by_material_id(
        material_id, **_structure_kwargs(structure_type)
    )
    return _to_ase_atoms(structure)


def get_structure_record_by_id(
    material_id: str,
    api_key: str | None = None,
    structure_type: str = "final",
) -> MaterialsProjectStructure:
    """Return one Materials Project structure with provenance metadata."""

    rester = _make_mprester(api_key)
    structure = rester.get_structure_by_material_id(
        material_id, **_structure_kwargs(structure_type)
    )
    atoms = _to_ase_atoms(structure)
    docs = rester.materials.summary.search(
        material_ids=[material_id],
        fields=["material_id", "formula_pretty", "symmetry"],
        num_chunks=1,
        chunk_size=1,
    )
    doc = docs[0] if docs else {}
    symmetry = _field(doc, "symmetry")
    return MaterialsProjectStructure(
        material_id=str(_field(doc, "material_id", material_id)),
        formula=str(_field(doc, "formula_pretty", atoms.get_chemical_formula())),
        atoms=atoms,
        spacegroup_number=_symmetry_field(symmetry, "number"),
        spacegroup_symbol=_symmetry_field(symmetry, "symbol"),
        metadata={"structure_type": structure_type},
    )


def get_structures_by_spacegroup(
    spacegroup: int | str,
    api_key: str | None = None,
    limit: int | None = None,
    fields: list[str] | None = None,
    **kwargs: Any,
) -> list[MaterialsProjectStructure]:
    """Query Materials Project structures by space group number or symbol."""

    if isinstance(spacegroup, int) or str(spacegroup).isdigit():
        kwargs["spacegroup_number"] = int(spacegroup)
    else:
        kwargs["spacegroup_symbol"] = str(spacegroup)
    return query_structures(api_key=api_key, limit=limit, fields=fields, **kwargs)


def query_structures(
    api_key: str | None = None,
    limit: int | None = None,
    fields: list[str] | None = None,
    **filters: Any,
) -> list[MaterialsProjectStructure]:
    """Query Materials Project summary search and return provenance records."""

    rester = _make_mprester(api_key)
    query = dict(filters)
    query["fields"] = fields or list(DEFAULT_FIELDS)
    _apply_limit(query, limit)
    docs = rester.materials.summary.search(**query)
    return [_record_from_doc(doc) for doc in docs]


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return token or "structure"


def _filename_for_record(record: MaterialsProjectStructure, fmt: str) -> str:
    suffix = "vasp" if fmt == "vasp" else fmt
    return f"{_safe_token(record.material_id)}_{_safe_token(record.formula)}.{suffix}"


def write_structure_records(
    records: list[MaterialsProjectStructure],
    output_dir: str | Path,
    fmt: str = "vasp",
    overwrite: bool = False,
) -> list[MaterialsProjectStructure]:
    """Write records to structure files and return records with filenames filled."""

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for record in records:
        filename = _filename_for_record(record, fmt)
        path = outdir / filename
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {path}. Use overwrite=True or --overwrite to replace it."
            )
        write(path, record.atoms, format=fmt)
        record.structure_file = filename
        written.append(record)
    return written


def write_manifest(
    records: list[MaterialsProjectStructure],
    path: str | Path,
    query_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a YAML manifest for Materials Project query outputs."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source": "materials_project",
        "query": query_metadata or {},
        "results": [
            {
                "material_id": record.material_id,
                "formula": record.formula,
                "spacegroup_number": record.spacegroup_number,
                "spacegroup_symbol": record.spacegroup_symbol,
                "structure_file": record.structure_file,
            }
            for record in records
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return manifest_path


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_filter_pair(pair: str) -> tuple[str, Any]:
    """Parse a CLI ``key=value`` filter into a query keyword argument."""

    if "=" not in pair:
        raise ValueError("Generic filters must use key=value syntax.")
    key, value = pair.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Generic filters must include a non-empty key.")
    if "," in value:
        return key, tuple(_parse_scalar(part.strip()) for part in value.split(","))
    return key, _parse_scalar(value.strip())


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key",
        default=None,
        help="Materials Project API key; defaults to MP_API_KEY or MPRester config",
    )
    parser.add_argument(
        "--format", default="vasp", help="ASE output format (default: vasp)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )


def _ensure_structure_fields(fields: list[str] | None, output_requested: bool) -> None:
    if output_requested and fields is not None and "structure" not in fields:
        raise ValueError(
            "Output writing requires the `structure` field. Include `--fields structure` or omit --fields for defaults."
        )


def mlmp_cli() -> int:
    """CLI entry point for Materials Project structure access."""

    parser = argparse.ArgumentParser(
        description="Fetch structures from the Materials Project database."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch", help="Fetch one Materials Project structure by material ID"
    )
    fetch.add_argument("material_id")
    fetch.add_argument("--output", "-o", required=True, help="Output structure file")
    fetch.add_argument(
        "--structure-type",
        default="final",
        choices=["final", "initial"],
        help="Structure type passed to MPRester",
    )
    _add_common_options(fetch)

    spacegroup = subparsers.add_parser(
        "spacegroup", help="Fetch structures matching a space group number or symbol"
    )
    spacegroup.add_argument("spacegroup", help="Space group number or symbol")
    spacegroup.add_argument(
        "--output-dir", "-o", required=True, help="Directory for output structures"
    )
    spacegroup.add_argument(
        "--limit", type=int, default=None, help="Maximum number of structures"
    )
    spacegroup.add_argument(
        "--fields", nargs="+", default=None, help="MP summary fields to request"
    )
    spacegroup.add_argument(
        "--manifest",
        default="manifest.yaml",
        help="Manifest filename (default: manifest.yaml)",
    )
    _add_common_options(spacegroup)

    query = subparsers.add_parser(
        "query", help="Fetch structures using Materials Project summary filters"
    )
    query.add_argument(
        "--output-dir", "-o", required=True, help="Directory for output structures"
    )
    query.add_argument(
        "--limit", type=int, default=None, help="Maximum number of structures"
    )
    query.add_argument(
        "--fields", nargs="+", default=None, help="MP summary fields to request"
    )
    query.add_argument(
        "--manifest",
        default="manifest.yaml",
        help="Manifest filename (default: manifest.yaml)",
    )
    query.add_argument("--elements", nargs="+", default=None, help="Required elements")
    query.add_argument("--chemsys", default=None, help="Chemical system, e.g. Zr-Si")
    query.add_argument("--formula", default=None, help="Formula filter")
    query.add_argument(
        "--energy-above-hull", nargs=2, type=float, default=None, metavar=("MIN", "MAX")
    )
    query.add_argument(
        "--band-gap", nargs=2, type=float, default=None, metavar=("MIN", "MAX")
    )
    query.add_argument(
        "--num-sites", nargs=2, type=int, default=None, metavar=("MIN", "MAX")
    )
    query.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Repeatable generic MP filter as key=value",
    )
    _add_common_options(query)

    args = parser.parse_args()
    try:
        if args.command == "fetch":
            atoms = get_structure_by_id(
                args.material_id,
                api_key=args.api_key,
                structure_type=args.structure_type,
            )
            output = Path(args.output)
            if output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output file already exists: {output}. Use --overwrite to replace it."
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            write(output, atoms, format=args.format)
            return 0

        if args.command == "spacegroup":
            _ensure_structure_fields(args.fields, True)
            sg_value: int | str = (
                int(args.spacegroup) if args.spacegroup.isdigit() else args.spacegroup
            )
            records = get_structures_by_spacegroup(
                sg_value, api_key=args.api_key, limit=args.limit, fields=args.fields
            )
            written = write_structure_records(
                records, args.output_dir, fmt=args.format, overwrite=args.overwrite
            )
            write_manifest(
                written,
                Path(args.output_dir) / args.manifest,
                {
                    "command": "spacegroup",
                    "spacegroup": sg_value,
                    "limit": args.limit,
                    "fields": args.fields or DEFAULT_FIELDS,
                },
            )
            return 0

        if args.command == "query":
            _ensure_structure_fields(args.fields, True)
            filters: dict[str, Any] = {}
            for name in ("elements", "chemsys", "formula"):
                value = getattr(args, name)
                if value is not None:
                    filters[name] = value
            for source, dest in (
                ("energy_above_hull", "energy_above_hull"),
                ("band_gap", "band_gap"),
                ("num_sites", "num_sites"),
            ):
                value = getattr(args, source)
                if value is not None:
                    filters[dest] = tuple(value)
            for pair in args.filter:
                key, value = parse_filter_pair(pair)
                filters[key] = value
            records = query_structures(
                api_key=args.api_key, limit=args.limit, fields=args.fields, **filters
            )
            written = write_structure_records(
                records, args.output_dir, fmt=args.format, overwrite=args.overwrite
            )
            write_manifest(
                written,
                Path(args.output_dir) / args.manifest,
                {
                    "command": "query",
                    "filters": filters,
                    "limit": args.limit,
                    "fields": args.fields or DEFAULT_FIELDS,
                },
            )
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(mlmp_cli())
