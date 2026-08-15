"""Integration test: full bond-valence workflow stage graph (story-010).

Runs every real stage (relax/sample/evaluate/fit/validate/report) with a
synthetic deterministic calculator and real easybondvalence. Only init_calc is
patched to inject the synthetic calculator.
"""

import re
import textwrap

import yaml
from ase.build import bulk

from atomchain import bondvalence_workflow as bvw
from tests.test_bondvalence_stages import SpringCalculator

SEED = """
provenance: "integration test tables"
oxidation_states: {Na: 1, Cl: -1}
contacts: {cutoff: 3.2}
pairs:
  - selector: [Na, Cl]
    r0: {initial: 3.2, lower: 2.0, upper: 4.2}
    b: {initial: 0.37, lower: 0.1, upper: 1.0, transform: positive}
"""


def test_full_workflow_stage_graph(tmp_path, monkeypatch):
    parent = bulk("NaCl", "rocksalt", a=5.6)
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(textwrap.dedent(SEED), encoding="utf-8")
    structure_path = tmp_path / "POSCAR"
    from ase.io import write as ase_write

    ase_write(structure_path, parent, format="vasp", vasp5=True, direct=True)

    monkeypatch.setattr(
        bvw,
        "init_calc",
        lambda model_type, model_path=None: SpringCalculator(parent.positions, k=5.0),
    )
    result = bvw.run_bondvalence_workflow(
        bvw.BVWorkflowConfig(
            structure=structure_path,
            seed_file=seed_path,
            output_dir=tmp_path / "bv",
            relax=bvw.RelaxStageConfig(fmax=0.05),
            sampling=bvw.BVSamplingStageConfig(
                rattle_amplitudes=(0.05,),
                frames_per_amplitude=2,
                strain_range=(-0.02, 0.02),
                strain_points=3,
                energy_window=10.0,
            ),
            validation=bvw.BVValidationStageConfig(gii_threshold=2.0),
        )
    )

    assert result.report_md.exists()
    assert result.report_yaml.exists()
    assert isinstance(result.passed, bool)
    assert result.fitted_parameters is not None

    report = result.report_md.read_text("utf-8")
    for section in ("## Stages", "## Fit", "## Validation", "## Limitations"):
        assert section in report
    for link in re.findall(r"\]\(([^)#]+)\)", report):
        assert (tmp_path / "bv" / link).exists(), link

    manifest = yaml.safe_load((tmp_path / "bv" / "manifest.yaml").read_text("utf-8"))
    for name in bvw.STAGE_ORDER:
        assert manifest["stages"][name]["status"] in ("completed", "reused"), name

    fitted = yaml.safe_load(result.fitted_parameters.read_text("utf-8"))
    assert "Na-Cl" in fitted["pairs"]
    # Rocksalt Na-Cl at 2.8 A with target 1 v.u. over 6 bonds pushes r0 well
    # above the 3.2 seed once b is free; assert the fit moved and stayed sane.
    pair = fitted["pairs"]["Na-Cl"]
    assert 2.0 < pair["r0"] < 4.2
    assert 0.1 < pair["b"] < 1.0

    metrics = result.validation_metrics
    assert metrics["passed"] is True
    assert metrics["aggregate_rms_dv"] < 2.0

    # Second run with unchanged config reuses every expensive stage.
    calls = []
    original_relax = bvw.run_relax_stage

    def counting_relax(context, metadata):
        calls.append("relax")
        return original_relax(context, metadata)

    monkeypatch.setattr(bvw, "run_relax_stage", counting_relax)
    second = bvw.run_bondvalence_workflow(
        bvw.BVWorkflowConfig(
            structure=structure_path,
            seed_file=seed_path,
            output_dir=tmp_path / "bv",
            relax=bvw.RelaxStageConfig(fmax=0.05),
            sampling=bvw.BVSamplingStageConfig(
                rattle_amplitudes=(0.05,),
                frames_per_amplitude=2,
                strain_range=(-0.02, 0.02),
                strain_points=3,
                energy_window=10.0,
            ),
            validation=bvw.BVValidationStageConfig(gii_threshold=2.0),
        )
    )
    assert calls == []
    for name in bvw.STAGE_ORDER[:-1]:
        assert second.manifest["stages"][name]["status"] == "reused", name
