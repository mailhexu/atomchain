"""Validation utilities for generated DDB sidecar metadata."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_metadata(metadata):
    if isinstance(metadata, (str, Path)):
        with Path(metadata).open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return metadata


def validate_ddb_metadata(metadata):
    """Validate required sidecar fields and return a list of errors."""
    data = load_metadata(metadata)
    errors = []
    if not isinstance(data, dict):
        return ["metadata must be a mapping"]
    for key in ("format", "header", "derivative_blocks"):
        if key not in data:
            errors.append(f"missing required key: {key}")
    header = data.get("header", {})
    for key in ("lattice_bohr", "atomic_numbers", "reduced_positions", "species"):
        if key not in header:
            errors.append(f"missing header key: {key}")
    natom = len(header.get("atomic_numbers", []))
    if natom and len(header.get("reduced_positions", [])) != natom:
        errors.append("reduced_positions length does not match atomic_numbers")
    for i, block in enumerate(data.get("derivative_blocks", []) or []):
        for key in (
            "qpoint",
            "perturbation_i",
            "perturbation_j",
            "value",
            "units",
            "source",
        ):
            if key not in block:
                errors.append(f"derivative_blocks[{i}] missing key: {key}")
    return errors


def compare_derivative_blocks(actual, reference, tolerance=1e-10):
    """Compare selected derivative block dictionaries with a tolerance."""
    errors = []
    actual_blocks = (
        load_metadata(actual).get("derivative_blocks", [])
        if isinstance(actual, (str, Path, dict))
        else actual
    )
    reference_blocks = (
        load_metadata(reference).get("derivative_blocks", [])
        if isinstance(reference, (str, Path, dict))
        else reference
    )
    if len(actual_blocks) != len(reference_blocks):
        errors.append(
            f"block count differs: {len(actual_blocks)} != {len(reference_blocks)}"
        )
        return errors
    for index, (a, r) in enumerate(zip(actual_blocks, reference_blocks)):
        for key in ("qpoint", "perturbation_i", "perturbation_j", "units", "source"):
            if a.get(key) != r.get(key):
                errors.append(f"block {index} key {key} differs")
        if (
            abs(_numeric_value(a.get("value")) - _numeric_value(r.get("value")))
            > tolerance
        ):
            errors.append(f"block {index} value differs")
    return errors


def _numeric_value(value):
    if isinstance(value, dict):
        return complex(value.get("real", 0.0), value.get("imag", 0.0))
    return complex(value)
