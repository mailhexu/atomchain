"""Tests for the bond-valence workflow report stage (story-008)."""

import json
import re

import numpy as np
import yaml

from atomchain import bondvalence_workflow as bvw
from tests.test_bondvalence_validate import _workflow_context


def _reported_context(tmp_path):
    context = _workflow_context(tmp_path)
    validate_metadata = bvw.StageMetadata(name="validate")
    validate_metadata.outputs = bvw.run_validate_stage(context, validate_metadata)
    context.manifest["stages"]["validate"] = validate_metadata.to_dict()
    return context


def test_report_contains_sections_and_resolving_links(tmp_path):
    context = _reported_context(tmp_path)
    outputs = bvw.run_report_stage(context, bvw.StageMetadata(name="report"))
    report = (tmp_path / "bv" / "report.md").read_text("utf-8")
    for section in (
        "## Procedure",
        "## Configuration",
        "## Stages",
        "## Artifact Index",
        "## Seed",
        "## Sampling",
        "## Fit",
        "## Validation",
        "## Parameter Comparison",
        "## Limitations",
    ):
        assert section in report, section
    assert "Gate PASSED" in report
    report_yaml = yaml.safe_load((tmp_path / "bv" / "report.yaml").read_text("utf-8"))
    assert report_yaml["format"] == "atomchain-bondvalence-workflow-v1"
    links = re.findall(r"\]\(([^)#]+)\)", report)
    assert links
    for link in links:
        assert (tmp_path / "bv" / link).exists(), link
    assert outputs["report_md"] == "report.md"
    assert outputs["report_yaml"] == "report.yaml"


def test_report_writes_multipole_descriptors(tmp_path):
    context = _reported_context(tmp_path)
    bvw.run_report_stage(context, bvw.StageMetadata(name="report"))
    data = json.loads(
        (tmp_path / "bv" / "multipoles" / "relaxed_parent.json").read_text("utf-8")
    )
    n_sites = len(context.parent_atoms)
    assert np.asarray(data["dipoles"]).shape == (n_sites, 3)
    assert np.asarray(data["quadrupoles"]).shape == (n_sites, 3, 3)
    assert "invariants" in data
    assert "max_rank" in data
    report = (tmp_path / "bv" / "report.md").read_text("utf-8")
    assert "descriptor" in report.lower()


def test_report_multipole_failure_is_warning(tmp_path, monkeypatch):
    import easybondvalence

    def broken(structural, max_rank):
        raise RuntimeError("zero bond-valence site")

    monkeypatch.setattr(easybondvalence, "expand_multipoles", broken)
    context = _reported_context(tmp_path)
    metadata = bvw.StageMetadata(name="report")
    bvw.run_report_stage(context, metadata)
    assert any("multipole" in warning.lower() for warning in metadata.warnings)
    assert (tmp_path / "bv" / "report.md").exists()
    assert not (tmp_path / "bv" / "multipoles" / "relaxed_parent.json").exists()


def test_report_lists_failed_stage_on_partial_failure(tmp_path):
    context = _reported_context(tmp_path)
    context.manifest["stages"]["fit"] = {
        "name": "fit",
        "status": "failed",
        "error": "fit exploded",
        "outputs": {},
    }
    bvw.run_report_stage(context, bvw.StageMetadata(name="report"))
    report = (tmp_path / "bv" / "report.md").read_text("utf-8")
    assert "fit exploded" in report
    assert "failed" in report


def test_report_yaml_is_complete_structured_payload(tmp_path):
    context = _reported_context(tmp_path)
    bvw.run_report_stage(context, bvw.StageMetadata(name="report"))
    payload = yaml.safe_load((tmp_path / "bv" / "report.yaml").read_text("utf-8"))
    for key in (
        "seed",
        "sampling",
        "fit",
        "validation",
        "comparison",
        "artifacts",
        "limitations",
        "stages",
    ):
        assert key in payload, key
    seed_pair = payload["seed"]["pairs"][0]
    assert "initial" in seed_pair["r0"] and "lower" in seed_pair["r0"]
    assert payload["fit"]["termination"]
    assert payload["fit"]["validation_data_cost"] is not None
    assert payload["comparison"]
    assert payload["artifacts"]


def test_report_omits_literature_column_without_literature(tmp_path):
    context = _reported_context(tmp_path)
    bvw.run_report_stage(context, bvw.StageMetadata(name="report"))
    report = (tmp_path / "bv" / "report.md").read_text("utf-8")
    assert "Literature" not in report
