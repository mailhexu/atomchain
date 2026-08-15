"""Tests for the bond-valence seed file loader (story-002)."""

import textwrap

import pytest
from ase import Atoms
from ase.build import bulk

from atomchain import bondvalence_workflow as bvw

VALID_NACL = """
provenance: "demo tables, transcribed for tests"
oxidation_states: {Na: 1, Cl: -1}
contacts: {cutoff: 3.2}
pairs:
  - selector: [Na, Cl]
    r0: {initial: 2.30, lower: 1.0, upper: 3.0}
    b: {initial: 0.37, lower: 0.1, upper: 1.0, transform: positive}
"""

VALID_BATIO3_WITH_TIE = """
provenance: "demo tables"
oxidation_states: {Ba: 2, Ti: 4, O: -2}
contacts: {cutoff: 3.2}
pairs:
  - selector: [Ba, O]
    r0: {initial: 2.285, lower: 1.8, upper: 3.0}
    b: {initial: 0.37, lower: 0.2, upper: 0.6, transform: positive}
  - selector: [Ti, O]
    r0: {initial: 1.815, lower: 1.4, upper: 2.4}
    b: {initial: 0.37, lower: 0.2, upper: 0.6, transform: positive, tie: "Ba-O:b"}
"""


def _write(tmp_path, text, name="seed.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_valid_seed_loads_all_objects(tmp_path):
    seed = bvw.load_seed_file(_write(tmp_path, VALID_NACL), species=("Na", "Cl"))
    assert [p.identifier for p in seed.parameter_set.parameters] == ["Na-Cl"]
    param = seed.parameter_set.parameters[0]
    assert param.selector.center_element == "Na"
    assert param.selector.neighbor_element == "Cl"
    assert param.r0 == pytest.approx(2.30)
    assert param.b == pytest.approx(0.37)
    assert param.source == "demo tables, transcribed for tests"
    assert seed.contact_policy.cutoff == pytest.approx(3.2)
    assert seed.oxidation_states == {"Na": 1.0, "Cl": -1.0}
    keys = sorted(p.key for p in seed.fit_parameters)
    assert keys == ["Na-Cl:b", "Na-Cl:r0"]
    by_key = {p.key: p for p in seed.fit_parameters}
    assert by_key["Na-Cl:b"].transform == "positive"
    valences = seed.valences_for(("Na", "Cl", "Cl"))
    assert list(valences.target_values) == [1.0, 1.0, 1.0]
    assert valences.source == seed.provenance


def test_tied_parameter_maps_to_easybondvalence_tie(tmp_path):
    seed = bvw.load_seed_file(_write(tmp_path, VALID_BATIO3_WITH_TIE))
    by_key = {p.key: p for p in seed.fit_parameters}
    assert by_key["Ti-O:b"].status == "tied"
    assert by_key["Ti-O:b"].tie_to == "Ba-O:b"
    assert by_key["Ba-O:b"].status == "free"


def test_missing_oxidation_state_is_aggregated_error(tmp_path):
    bad = VALID_NACL.replace(
        "oxidation_states: {Na: 1, Cl: -1}", "oxidation_states: {Na: 1}"
    )
    with pytest.raises(ValueError) as excinfo:
        bvw.load_seed_file(_write(tmp_path, bad), species=("Na", "Cl"))
    assert "Cl" in str(excinfo.value)


def test_all_schema_violations_reported_together(tmp_path):
    bad = """
    pairs:
      - selector: [Na, Cl]
        r0: {initial: 2.30}
        b: {initial: 0.37}
    """
    with pytest.raises(ValueError) as excinfo:
        bvw.load_seed_file(_write(tmp_path, bad), species=("Na", "Cl"))
    message = str(excinfo.value)
    assert "provenance" in message
    assert "cutoff" in message
    assert "Cl" in message  # missing oxidation state
    assert "b" in message  # free b without positive transform/bound


def test_free_b_requires_positive_lower_or_transform(tmp_path):
    bad = VALID_NACL.replace(
        "b: {initial: 0.37, lower: 0.1, upper: 1.0, transform: positive}",
        "b: {initial: 0.37}",
    )
    with pytest.raises(ValueError, match="b"):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_broken_tie_target_is_aggregated_error(tmp_path):
    bad = VALID_BATIO3_WITH_TIE.replace('tie: "Ba-O:b"', 'tie: "Sr-O:b"')
    with pytest.raises(ValueError) as excinfo:
        bvw.load_seed_file(_write(tmp_path, bad))
    assert "Sr-O:b" in str(excinfo.value)


def test_tie_to_itself_is_rejected(tmp_path):
    bad = VALID_BATIO3_WITH_TIE.replace('tie: "Ba-O:b"', 'tie: "Ti-O:b"')
    with pytest.raises(ValueError, match="Ti-O:b"):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_missing_seed_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bvw.load_seed_file(tmp_path / "absent.yaml")


def test_structure_species_pair_cross_check_warns(tmp_path):
    # Ti present in the structure but no Ti pair in the seed.
    bad = VALID_NACL.replace(
        "oxidation_states: {Na: 1, Cl: -1}",
        "oxidation_states: {Na: 1, Cl: -1, Ti: 4, O: -2}",
    )
    seed = bvw.load_seed_file(_write(tmp_path, bad), species=("Na", "Cl", "Ti", "O"))
    warnings_text = " ".join(seed.warnings)
    assert "Ti-O" in warnings_text


def test_require_easybondvalence_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("easybondvalence"):
            raise ImportError("No module named easybondvalence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="atomchain\\[bondvalence\\]"):
        bvw._require_easybondvalence()


def test_contact_audit_warns_on_missing_pair_contacts(tmp_path):
    seed = bvw.load_seed_file(_write(tmp_path, VALID_NACL))
    # NaCl pair far beyond the 3.2 A cutoff: zero Na-Cl contacts expected.
    atoms = Atoms("NaCl", positions=[[0, 0, 0], [5.0, 0, 0]])
    warnings = bvw.audit_contacts(seed, atoms)
    assert any("Na-Cl" in warning for warning in warnings)


def test_contact_audit_clean_for_rocksalt(tmp_path):
    seed = bvw.load_seed_file(_write(tmp_path, VALID_NACL))
    atoms = bulk("NaCl", "rocksalt", a=5.6)
    warnings = bvw.audit_contacts(seed, atoms)
    assert warnings == []


def test_r0_tie_is_honored(tmp_path):
    seed_text = """
    provenance: "tie demo"
    oxidation_states: {Na: 1, Cl: -1, K: 1}
    contacts: {cutoff: 3.6}
    pairs:
      - selector: [Na, Cl]
        r0: {initial: 2.30, lower: 2.0, upper: 2.6}
        b: {initial: 0.37, lower: 0.2, upper: 0.6, transform: positive}
      - selector: [K, Cl]
        r0: {initial: 2.30, lower: 2.0, upper: 2.6, tie: "Na-Cl:r0"}
        b: {initial: 0.37, lower: 0.2, upper: 0.6, transform: positive}
    """
    path = tmp_path / "seed.yaml"
    path.write_text(textwrap.dedent(seed_text), encoding="utf-8")
    seed = bvw.load_seed_file(path)
    by_key = {p.key: p for p in seed.fit_parameters}
    assert by_key["K-Cl:r0"].status == "tied"
    assert by_key["K-Cl:r0"].tie_to == "Na-Cl:r0"


def test_fixed_and_tie_conflict_is_error(tmp_path):
    bad = VALID_NACL.replace(
        "r0: {initial: 2.30, lower: 1.0, upper: 3.0}",
        'r0: {initial: 2.30, lower: 1.0, upper: 3.0, fixed: true, tie: "X-O:r0"}',
    )
    with pytest.raises(ValueError, match="fixed and tied"):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_typo_transform_is_rejected(tmp_path):
    bad = VALID_NACL.replace("transform: positive}", "transform: positve}")
    with pytest.raises(ValueError, match="positve"):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_nan_oxidation_state_is_error(tmp_path):
    bad = VALID_NACL.replace(
        "oxidation_states: {Na: 1, Cl: -1}",
        "oxidation_states: {Na: .nan, Cl: -1}",
    )
    with pytest.raises(ValueError, match="finite"):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_missing_oxidation_states_block_is_error(tmp_path):
    bad = VALID_NACL.replace("oxidation_states: {Na: 1, Cl: -1}\n", "")
    with pytest.raises(ValueError, match="oxidation_states"):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_overflowing_oxidation_state_is_aggregated(tmp_path):
    bad = VALID_NACL.replace(
        "oxidation_states: {Na: 1, Cl: -1}",
        "oxidation_states: {Na: " + "9" * 400 + ", Cl: -1}",
    )
    with pytest.raises(ValueError):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_reversed_duplicate_pair_is_error(tmp_path):
    bad = (
        VALID_NACL
        + """  - selector: [Cl, Na]
    r0: {initial: 2.30, lower: 1.0, upper: 3.0}
    b: {initial: 0.37, lower: 0.1, upper: 1.0, transform: positive}
"""
    )
    with pytest.raises(ValueError, match="duplicate"):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_malformed_literature_is_aggregated(tmp_path):
    bad = VALID_NACL.replace(
        'provenance: "demo tables, transcribed for tests"',
        'provenance: "demo tables, transcribed for tests"\nliterature: {Na-Cl: 3.0}',
    )
    with pytest.raises(ValueError, match="literature"):
        bvw.load_seed_file(_write(tmp_path, bad))


def test_valid_literature_loads(tmp_path):
    good = VALID_NACL.replace(
        'provenance: "demo tables, transcribed for tests"',
        'provenance: "demo tables, transcribed for tests"\n'
        "literature:\n  Na-Cl: {r0: 2.28, b: 0.36}",
    )
    seed = bvw.load_seed_file(_write(tmp_path, good))
    assert seed.literature == {"Na-Cl": {"r0": 2.28, "b": 0.36}}


def test_audit_contacts_orientation_insensitive(tmp_path):
    seed = bvw.load_seed_file(_write(tmp_path, VALID_NACL))
    rocksalt = bulk("NaCl", "rocksalt", a=5.6)
    reversed_order = rocksalt[[1, 0]]
    assert bvw.audit_contacts(seed, reversed_order) == []
