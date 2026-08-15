"""Tests for the bond-valence workflow stage engine (story-001)."""

import json

import pytest
import yaml
from ase.build import bulk

from atomchain import bondvalence_workflow as bvw


def _fake_stages(calls, fail_stage=None, outputs=None):
    """Build a stage-function map that records calls like the multibinit tests."""

    def make(name):
        def run(context, metadata):
            calls.append(name)
            if name == fail_stage:
                raise RuntimeError(f"{name} failed")
            out = dict((outputs or {}).get(name, {"marker": name}))
            if name == "validate" and "metrics" in out:
                metrics_path = context.output_dir / out["metrics"]
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(
                    json.dumps({"passed": out.get("passed", False)}), encoding="utf-8"
                )
            if name == "report":
                report_md = context.output_dir / "report.md"
                report_yaml = context.output_dir / "report.yaml"
                errors = [
                    stage.get("error")
                    for stage in context.manifest["stages"].values()
                    if stage.get("error")
                ]
                report_md.write_text(
                    "report\n" + ("".join(f"{err}\n" for err in errors)),
                    encoding="utf-8",
                )
                report_yaml.write_text("{}", encoding="utf-8")
                return {"report_md": str(report_md), "report_yaml": str(report_yaml)}
            return out

        return run

    return {name: make(name) for name in bvw.STAGE_ORDER}


def _config(tmp_path):
    return bvw.BVWorkflowConfig(
        structure=bulk("NaCl", "rocksalt", a=5.6),
        seed_file="seed.yaml",
        output_dir=tmp_path / "bv",
    )


def test_dry_run_returns_stage_plan(tmp_path):
    result = bvw.run_bondvalence_workflow(
        _config(tmp_path), dry_run=True, stage_functions=_fake_stages([])
    )
    assert [item["name"] for item in result.manifest["stage_plan"]] == bvw.STAGE_ORDER
    assert result.passed is None


def test_manifest_format_and_stage_checkpoints(tmp_path):
    calls = []
    result = bvw.run_bondvalence_workflow(
        _config(tmp_path), stage_functions=_fake_stages(calls)
    )
    assert result.manifest["format"] == "atomchain-bondvalence-workflow-v1"
    for name in bvw.STAGE_ORDER:
        checkpoint = result.output_dir / name / "stage.yaml"
        assert checkpoint.exists(), name
        data = yaml.safe_load(checkpoint.read_text(encoding="utf-8"))
        assert data["status"] == "completed", name
        assert data["signature"], name
    assert calls == bvw.STAGE_ORDER


def test_matching_signature_reuses_completed_stages(tmp_path):
    calls = []
    stages = _fake_stages(calls)
    first = bvw.run_bondvalence_workflow(_config(tmp_path), stage_functions=stages)
    second = bvw.run_bondvalence_workflow(_config(tmp_path), stage_functions=stages)
    # report always re-executes (mlmbmodel contract); every other stage is reused.
    assert calls == bvw.STAGE_ORDER + ["report"]
    for name in bvw.STAGE_ORDER[:-1]:
        assert second.manifest["stages"][name]["status"] == "reused", name
    assert second.manifest["stages"]["report"]["status"] == "completed"
    assert first.manifest["stages"]["prepare"]["status"] == "completed"


def test_config_change_invalidates_signature(tmp_path):
    calls = []
    stages = _fake_stages(calls)
    config = _config(tmp_path)
    bvw.run_bondvalence_workflow(config, stage_functions=stages)
    config.validation.gii_threshold = 0.15
    bvw.run_bondvalence_workflow(config, stage_functions=stages)
    # Stage-specific signatures: only validate and report depend on the
    # threshold; the expensive stages are reused.
    assert calls.count("prepare") == 1
    assert calls.count("validate") == 2
    assert calls.count("report") == 2


def test_structure_change_invalidates_signature(tmp_path):
    calls = []
    stages = _fake_stages(calls)
    config = _config(tmp_path)
    bvw.run_bondvalence_workflow(config, stage_functions=stages)
    config.structure = bulk("Si", "diamond", a=5.4)
    bvw.run_bondvalence_workflow(config, stage_functions=stages)
    assert calls.count("prepare") == 2


def test_seed_file_content_change_invalidates_signature(tmp_path):
    seed = tmp_path / "seed.yaml"
    seed.write_text("provenance: v1\n", encoding="utf-8")
    calls = []
    stages = _fake_stages(calls)
    config = bvw.BVWorkflowConfig(
        structure=bulk("NaCl", "rocksalt", a=5.6),
        seed_file=seed,
        output_dir=tmp_path / "bv",
    )
    bvw.run_bondvalence_workflow(config, stage_functions=stages)
    seed.write_text("provenance: v2\n", encoding="utf-8")  # same path, new content
    bvw.run_bondvalence_workflow(config, stage_functions=stages)
    assert calls.count("fit") == 2
    # The expensive calculator stages are reused (story-006 contract).
    assert calls.count("relax") == 1
    assert calls.count("sample") == 1
    assert calls.count("evaluate") == 1


def test_passed_is_none_when_validation_lacks_pass_metric(tmp_path):
    calls = []
    result = bvw.run_bondvalence_workflow(
        _config(tmp_path),
        stage_functions=_fake_stages(calls, outputs={"validate": {"marker": "x"}}),
    )
    assert result.passed is None


def test_force_stage_reruns_stage_and_downstream(tmp_path):
    calls = []
    stages = _fake_stages(calls)
    bvw.run_bondvalence_workflow(_config(tmp_path), stage_functions=stages)
    calls.clear()
    bvw.run_bondvalence_workflow(
        _config(tmp_path), stage_functions=stages, force_stage=["fit"]
    )
    assert calls == ["fit", "validate", "report"]


def test_stop_after_persists_manifest_and_report(tmp_path):
    calls = []
    result = bvw.run_bondvalence_workflow(
        _config(tmp_path),
        stage_functions=_fake_stages(calls),
        stop_after="sample",
    )
    assert calls == ["prepare", "relax", "sample", "report"]
    manifest = yaml.safe_load(
        (result.output_dir / "manifest.yaml").read_text(encoding="utf-8")
    )
    for name in ("evaluate", "fit", "validate"):
        assert manifest["stages"][name]["status"] == "skipped", name
    assert manifest["stages"]["report"]["status"] == "completed"
    assert (result.output_dir / "report.md").exists()


def test_stage_failure_writes_partial_report_and_raises(tmp_path):
    calls = []
    with pytest.raises(RuntimeError, match="fit failed"):
        bvw.run_bondvalence_workflow(
            _config(tmp_path), stage_functions=_fake_stages(calls, fail_stage="fit")
        )
    output = _config(tmp_path).output_dir
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "fit failed" in report
    checkpoint = yaml.safe_load((output / "fit" / "stage.yaml").read_text("utf-8"))
    assert checkpoint["status"] == "failed"
    manifest = yaml.safe_load((output / "manifest.yaml").read_text("utf-8"))
    # Downstream stages are recorded as skipped before the partial report.
    assert manifest["stages"]["validate"]["status"] == "skipped"
    assert manifest["stages"]["report"]["status"] == "completed"


def test_unknown_forced_stage_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown workflow stage"):
        bvw.run_bondvalence_workflow(
            _config(tmp_path),
            stage_functions=_fake_stages([]),
            force_stage=["ddb"],
        )


def test_coerce_config_rejects_config_and_kwargs(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        bvw.run_bondvalence_workflow(
            _config(tmp_path), stage_functions=_fake_stages([]), seed_file="other.yaml"
        )


def test_workflow_requires_seed_file(tmp_path):
    with pytest.raises(ValueError):
        bvw.run_bondvalence_workflow(
            structure=bulk("NaCl", "rocksalt", a=5.6), output_dir=tmp_path / "bv2"
        )


def test_result_exposes_paths_and_passed(tmp_path):
    calls = []
    outputs = {
        "fit": {"fitted_parameters": "fit/fitted_parameters.yaml"},
        "validate": {
            "metrics": "validation/metrics.json",
            "passed": True,
        },
    }
    result = bvw.run_bondvalence_workflow(
        _config(tmp_path), stage_functions=_fake_stages(calls, outputs=outputs)
    )
    assert result.report_md.exists()
    assert result.report_yaml.exists()
    assert result.fitted_parameters is not None
    assert result.validation_metrics.get("passed") is True
    assert result.passed is True


def test_stage_outputs_are_relativized(tmp_path):
    calls = []
    result = bvw.run_bondvalence_workflow(
        _config(tmp_path),
        stage_functions=_fake_stages(
            calls, outputs={"prepare": {"some_path": str(tmp_path / "bv" / "x.yaml")}}
        ),
    )
    outputs = result.manifest["stages"]["prepare"]["outputs"]
    assert outputs["some_path"] == "x.yaml"


def test_report_stage_failure_raises(tmp_path):
    calls = []

    def fail_report(context, metadata):
        calls.append("report")
        raise RuntimeError("report failed")

    stages = _fake_stages([])
    stages["report"] = fail_report
    with pytest.raises(RuntimeError, match="report failed"):
        bvw.run_bondvalence_workflow(_config(tmp_path), stage_functions=stages)
    checkpoint = yaml.safe_load(
        (_config(tmp_path).output_dir / "report" / "stage.yaml").read_text("utf-8")
    )
    assert checkpoint["status"] == "failed"
