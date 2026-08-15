"""Tests for smart contact-cutoff suggestion (story-011)."""

import textwrap

import numpy as np
import pytest
import yaml
from ase import Atoms

from atomchain import bondvalence_workflow as bvw

BATIO3_SEED = """
provenance: "cutoff test tables"
oxidation_states: {Ba: 2, Ti: 4, O: -2}
contacts: {cutoff: 12.0}
pairs:
  - selector: [Ba, O]
    r0: {initial: 2.285, lower: 1.8, upper: 3.7}
    b: {initial: 0.37, fixed: true}
  - selector: [Ti, O]
    r0: {initial: 1.815, lower: 1.4, upper: 2.6}
    b: {initial: 0.37, fixed: true}
  - selector: [O, O]
    r0: {initial: 1.0, fixed: true}
    b: {initial: 0.37, fixed: true}
  - selector: [Ba, Ti]
    r0: {initial: 1.0, fixed: true}
    b: {initial: 0.37, fixed: true}
  - selector: [Ba, Ba]
    r0: {initial: 1.0, fixed: true}
    b: {initial: 0.37, fixed: true}
  - selector: [Ti, Ti]
    r0: {initial: 1.0, fixed: true}
    b: {initial: 0.37, fixed: true}
"""


def _batio3():
    a = 4.0
    return Atoms(
        "BaTiO3",
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
        ],
        cell=np.eye(3) * a,
        pbc=True,
    )


def _seed(tmp_path, text=BATIO3_SEED):
    path = tmp_path / "seed.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_suggest_cutoffs_batio3_intervals(tmp_path):
    seed = bvw.load_seed_file(_seed(tmp_path))
    suggestion = bvw.suggest_cutoffs(_batio3(), seed, sigma_max=0.10, eps_max=0.03)
    # Live shells: Ti-O at 2.0, Ba-O at 2.83, and the Ti-O second shell
    # at 4.47 (12 bonds ~ 9e-3 v.u. total); the 6.32 A Ba-O shell is dead.
    live = {(s.family, round(s.distance, 2)): s for s in suggestion.live_shells}
    assert any(
        family == "Ti-O" and abs(distance - 2.0) < 0.02 for family, distance in live
    )
    assert any(
        family == "Ba-O" and abs(distance - 2.83) < 0.02 for family, distance in live
    )
    assert any(
        family == "Ti-O" and abs(distance - 4.47) < 0.03 for family, distance in live
    )
    assert not any(distance > 6.0 for _, distance in live)

    # The only safe region is beyond the last live shell (Ba-O second shell
    # at 4.899 A); one interval plus its minimal-mode variant.
    assert len(suggestion.intervals) == 2
    base = suggestion.intervals[-1]
    assert base.lower == pytest.approx(5.47, abs=0.15)
    assert base.upper >= 11.0
    assert base.lower < base.representative < base.upper
    minimal_variant = suggestion.intervals[0]
    assert minimal_variant.representative < base.representative
    # All six families are inside every interval from the first one on.
    assert set(base.families_inside) == {
        "Ba-Ba",
        "Ba-O",
        "Ba-Ti",
        "O-O",
        "Ti-O",
        "Ti-Ti",
    }
    assert base.missing_families == ()
    assert suggestion.minimal is minimal_variant
    robust = suggestion.robust
    assert robust is not None
    assert robust.representative > 4.9


def test_suggest_cutoffs_reports_missing_families(tmp_path):
    # Drop the pad families: only Ba-O and Ti-O remain.
    text = BATIO3_SEED
    for pair in ("O, O", "Ba, Ti", "Ba, Ba", "Ti, Ti"):
        block_start = text.index(f"  - selector: [{pair}]")
        block_end = text.index("    b: {initial: 0.37, fixed: true}", block_start)
        block_end = text.index("\n", block_end) + 1
        text = text[:block_start] + text[block_end:]
    seed = bvw.load_seed_file(_seed(tmp_path, text))
    suggestion = bvw.suggest_cutoffs(_batio3(), seed, sigma_max=0.10, eps_max=0.03)
    assert suggestion.intervals
    missing = set(suggestion.intervals[0].missing_families)
    assert {"O-O", "Ba-Ti", "Ba-Ba", "Ti-Ti"} <= missing


def test_suggest_cutoffs_no_safe_interval_warns(tmp_path):
    seed = bvw.load_seed_file(_seed(tmp_path))
    suggestion = bvw.suggest_cutoffs(_batio3(), seed, sigma_max=3.0, eps_max=0.0)
    assert suggestion.intervals == ()
    assert any("no safe" in warning for warning in suggestion.warnings)


def test_loader_accepts_auto_cutoff(tmp_path):
    text = BATIO3_SEED.replace(
        "contacts: {cutoff: 12.0}", "contacts:\n  cutoff: auto\n  cutoff_mode: minimal"
    )
    seed = bvw.load_seed_file(_seed(tmp_path, text))
    assert seed.contact_policy is None
    assert seed.cutoff_auto is True
    assert seed.cutoff_mode == "minimal"


def test_loader_rejects_bad_cutoff_mode(tmp_path):
    text = BATIO3_SEED.replace(
        "contacts: {cutoff: 12.0}", "contacts:\n  cutoff: auto\n  cutoff_mode: huge"
    )
    with pytest.raises(ValueError, match="cutoff_mode"):
        bvw.load_seed_file(_seed(tmp_path, text))


def test_relax_resolves_auto_cutoff(tmp_path):
    from tests.test_bondvalence_stages import SpringCalculator

    text = BATIO3_SEED.replace("cutoff: 12.0", "cutoff: auto")
    config = bvw.BVWorkflowConfig(
        structure=_batio3(),
        seed_file=_seed(tmp_path, text),
        output_dir=tmp_path / "bv",
    )
    context = bvw.BVWorkflowContext(config, _batio3())
    context.reference_calculator = SpringCalculator(context.parent_atoms.positions)
    metadata = bvw.StageMetadata(name="relax")
    outputs = bvw.run_relax_stage(context, metadata)
    assert context.contact_policy is not None
    assert 4.47 < context.contact_policy.cutoff  # inside a safe interval
    assert outputs["cutoff"] == pytest.approx(context.contact_policy.cutoff)
    assert outputs["cutoff_suggestion"]
    # Robust mode default: representative beyond the last live shell.
    assert context.contact_policy.cutoff > 6.32


def test_relax_warns_on_unsafe_numeric_cutoff(tmp_path):
    from tests.test_bondvalence_stages import SpringCalculator

    # 3.0 A is inside the flicker margin of the 2.83 A Ba-O/O-O shell.
    text = BATIO3_SEED.replace("cutoff: 12.0", "cutoff: 3.0")
    config = bvw.BVWorkflowConfig(
        structure=_batio3(),
        seed_file=_seed(tmp_path, text),
        output_dir=tmp_path / "bv",
    )
    context = bvw.BVWorkflowContext(config, _batio3())
    context.reference_calculator = SpringCalculator(context.parent_atoms.positions)
    metadata = bvw.StageMetadata(name="relax")
    bvw.run_relax_stage(context, metadata)
    assert any(
        "flicker" in warning or "margin" in warning for warning in metadata.warnings
    )


def test_fit_and_validate_use_auto_policy(tmp_path):
    from tests.test_bondvalence_fit import TRUE_B, TRUE_R0
    from tests.test_bondvalence_stages import SpringCalculator

    d1 = TRUE_R0
    d2 = TRUE_R0 + TRUE_B * np.log(2.0)
    d4 = TRUE_R0 + TRUE_B * np.log(4.0)
    vacuum = 12.0
    parent = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [d2, 0, 0]],
        cell=[[2 * d2, 0, 0], [0, vacuum, 0], [0, 0, vacuum]],
        pbc=(True, False, False),
    )
    seed_text = """
    provenance: "auto policy test"
    oxidationation_states: {Na: 1, Cl: -1}
    contacts: {cutoff: auto}
    pairs:
      - selector: [Na, Cl]
        r0: {initial: 2.30, lower: 2.0, upper: 2.6}
        b: {initial: 0.37, lower: 0.2, upper: 0.6, transform: positive}
      - selector: [Na, Na]
        r0: {initial: 1.0, fixed: true}
        b: {initial: 0.37, fixed: true}
      - selector: [Cl, Cl]
        r0: {initial: 1.0, fixed: true}
        b: {initial: 0.37, fixed: true}
    """
    seed_text = seed_text.replace("oxidationation_states", "oxidation_states")
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(textwrap.dedent(seed_text), encoding="utf-8")
    config = bvw.BVWorkflowConfig(
        structure=parent,
        seed_file=seed_path,
        output_dir=tmp_path / "bv",
    )
    context = bvw.BVWorkflowContext(config, parent)
    context.reference_calculator = SpringCalculator(parent.positions)
    relax_metadata = bvw.StageMetadata(name="relax")
    relax_metadata.outputs = bvw.run_relax_stage(context, relax_metadata)
    context.manifest["stages"]["relax"] = relax_metadata.to_dict()
    assert context.contact_policy is not None

    # Na-Cl shells: 2.30 (live), 3*d2 (next Na-Cl at 7.67 -> dead).
    from ase.io import Trajectory

    sampling_dir = context.output_dir / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    chain = parent.copy()
    chain.info = {"frame_id": "chain", "group": "chain", "kind": "chain"}
    dimer = Atoms("NaCl", positions=[[0, 0, 0], [d1, 0, 0]])
    dimer.info = {"frame_id": "dimer", "group": "dimer", "kind": "dimer"}
    lattice = d4 * np.sqrt(2.0)
    layer = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [lattice / 2, lattice / 2, 0]],
        cell=[[lattice, 0, 0], [0, lattice, 0], [0, 0, vacuum]],
        pbc=(True, True, False),
    )
    layer.info = {"frame_id": "layer", "group": "layer", "kind": "layer"}
    with Trajectory(sampling_dir / "evaluated.traj", "w") as trajectory:
        trajectory.write(dimer)
        trajectory.write(chain)
        trajectory.write(layer)
    evaluate_metadata = bvw.StageMetadata(name="evaluate")
    evaluate_metadata.outputs = {"evaluated": "sampling/evaluated.traj"}
    context.manifest["stages"]["evaluate"] = evaluate_metadata.to_dict()
    fit_metadata = bvw.StageMetadata(name="fit")
    fit_metadata.outputs = bvw.run_fit_stage(context, fit_metadata)
    context.manifest["stages"]["fit"] = fit_metadata.to_dict()
    fitted = yaml.safe_load(
        (tmp_path / "bv" / "fit" / "fitted_parameters.yaml").read_text("utf-8")
    )
    assert fitted["pairs"]["Na-Cl"]["r0"] == pytest.approx(TRUE_R0, abs=1e-3)
    validate_metadata = bvw.StageMetadata(name="validate")
    bvw.run_validate_stage(context, validate_metadata)
    context.manifest["stages"]["validate"] = validate_metadata.to_dict()
    bvw.run_report_stage(context, bvw.StageMetadata(name="report"))
    report = (tmp_path / "bv" / "report.md").read_text("utf-8")
    assert "## Contact Cutoff" in report
    assert f"{context.contact_policy.cutoff:g}" in report


def test_live_shell_weight_formula_exact(tmp_path):
    seed = bvw.load_seed_file(_seed(tmp_path))
    suggestion = bvw.suggest_cutoffs(_batio3(), seed, sigma_max=0.10, eps_max=0.03)
    ti_o_first = next(
        shell
        for shell in suggestion.live_shells
        if shell.family == "Ti-O" and abs(shell.distance - 2.0) < 0.02
    )
    assert ti_o_first.multiplicity == 6
    expected = 6 * np.exp((1.815 - ti_o_first.distance) / 0.37)
    assert ti_o_first.weight == pytest.approx(expected, rel=1e-6)


def test_numeric_cutoff_warns_when_no_interval_exists(tmp_path):
    from tests.test_bondvalence_stages import SpringCalculator

    config = bvw.BVWorkflowConfig(
        structure=_batio3(),
        seed_file=_seed(tmp_path),
        output_dir=tmp_path / "bv",
        sampling=bvw.BVSamplingStageConfig(
            rattle_amplitudes=(3.0,), frames_per_amplitude=1, strain_points=2
        ),
    )
    context = bvw.BVWorkflowContext(config, _batio3())
    context.reference_calculator = SpringCalculator(context.parent_atoms.positions)
    metadata = bvw.StageMetadata(name="relax")
    bvw.run_relax_stage(context, metadata)
    assert any("no safe interval" in w for w in metadata.warnings)


def test_numeric_cutoff_warns_on_missing_families(tmp_path):
    from tests.test_bondvalence_stages import SpringCalculator

    # Seed without pads and a cutoff that sweeps past the 3.46/4.0 A shells.
    text = BATIO3_SEED.replace("cutoff: 12.0", "cutoff: 5.0")
    for pair in ("O, O", "Ba, Ti", "Ba, Ba", "Ti, Ti"):
        block_start = text.index(f"  - selector: [{pair}]")
        block_end = text.index("    b: {initial: 0.37, fixed: true}", block_start)
        block_end = text.index("\n", block_end) + 1
        text = text[:block_start] + text[block_end:]
    config = bvw.BVWorkflowConfig(
        structure=_batio3(),
        seed_file=_seed(tmp_path, text),
        output_dir=tmp_path / "bv",
    )
    context = bvw.BVWorkflowContext(config, _batio3())
    context.reference_calculator = SpringCalculator(context.parent_atoms.positions)
    metadata = bvw.StageMetadata(name="relax")
    bvw.run_relax_stage(context, metadata)
    missing_warnings = [w for w in metadata.warnings if "without parameters" in w]
    assert missing_warnings
    for family in ("O-O", "Ba-Ti"):
        assert any(family in w for w in missing_warnings)


def test_workflow_reuse_hydrates_auto_policy(tmp_path):
    """Full workflow with an auto seed; the second run must hydrate the
    reused relax checkpoint and still produce multipoles."""
    from tests.test_bondvalence_stages import SpringCalculator

    text = BATIO3_SEED.replace("cutoff: 12.0", "cutoff: auto")
    parent = _batio3()
    config = bvw.BVWorkflowConfig(
        structure=parent,
        seed_file=_seed(tmp_path, text),
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
    calls = []

    def make_calc(model_type, model_path=None):
        calls.append("init")
        return SpringCalculator(parent.positions, k=5.0)

    import atomchain.bondvalence_workflow as module

    original_relax = module.run_relax_stage

    def counting_relax(context, metadata):
        calls.append("relax")
        return original_relax(context, metadata)

    monkey_targets = {"init_calc": make_calc, "run_relax_stage": counting_relax}
    originals = {k: getattr(module, k) for k in monkey_targets}
    for key, value in monkey_targets.items():
        setattr(module, key, value)
    try:
        first = module.run_bondvalence_workflow(config)
        assert first.passed in (True, False)
        assert (tmp_path / "bv" / "multipoles" / "relaxed_parent.json").exists()
        second = module.run_bondvalence_workflow(config)
    finally:
        for key, value in originals.items():
            setattr(module, key, value)
    # Second run reuses the relax checkpoint (hydrated, not re-executed) and
    # still produces the multipole descriptors.
    assert calls.count("relax") == 1
    assert second.manifest["stages"]["relax"]["status"] == "reused"
    assert (tmp_path / "bv" / "multipoles" / "relaxed_parent.json").exists()
    report = (tmp_path / "bv" / "report.md").read_text("utf-8")
    assert "## Contact Cutoff" in report
