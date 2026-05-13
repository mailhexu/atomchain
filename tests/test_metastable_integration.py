import os

import numpy as np
import pytest
import yaml
from ase.io import read


def _get_calculator():
    from atomchain.init_model import init_calc

    for model in ["chgnet", "mace"]:
        try:
            calc = init_calc(model_type=model)
            return calc, model
        except Exception:
            continue
    return None, None


HAS_CALC, CALC_MODEL = _get_calculator()


@pytest.fixture
def catio3_cubic(fixtures_dir):
    return read(fixtures_dir / "CaTiO3.vasp")


@pytest.mark.skipif(
    not HAS_CALC, reason="No ML calculator available (tried chgnet, mace)"
)
@pytest.mark.slow
def test_metastable_catio3_pipeline(catio3_cubic, tmp_path):
    from atomchain.metastable import explore_metastable_states

    output_dir = str(tmp_path / "metastable_results")

    results = explore_metastable_states(
        catio3_cubic,
        calc=CALC_MODEL,
        nmax=1,
        amplitude=0.5,
        max_cell_size=64,
        output_dir=output_dir,
        phonon_ndim=np.diag([2, 2, 2]),
        relax_kwargs={"fmax": 0.1},
    )

    assert isinstance(results, dict)
    assert "results" in results
    assert (
        len(results["results"]) > 0
    ), "Expected at least one metastable state from CaTiO3"
    for r in results["results"]:
        assert "energy" in r
        assert "spacegroup_name" in r
        assert "source_modes" in r
        assert len(r["source_modes"]) > 0
        assert "structure_file" in r
        assert os.path.exists(os.path.join(output_dir, r["structure_file"]))


@pytest.mark.skipif(
    not HAS_CALC, reason="No ML calculator available (tried chgnet, mace)"
)
@pytest.mark.slow
def test_metastable_catio3_report(catio3_cubic, tmp_path):
    from atomchain.metastable import explore_metastable_states
    from atomchain.metastable_cli import generate_report

    output_dir = str(tmp_path / "metastable_results")
    results = explore_metastable_states(
        catio3_cubic,
        calc=CALC_MODEL,
        nmax=1,
        amplitude=0.5,
        output_dir=output_dir,
        phonon_ndim=np.diag([2, 2, 2]),
        relax_kwargs={"fmax": 0.1},
    )

    params = {"nmax": 1, "amplitude": 0.5, "max_cell_size": 64}
    generate_report(results, catio3_cubic, CALC_MODEL, params, output_dir)

    report_path = os.path.join(output_dir, "report.yaml")
    assert os.path.exists(report_path)

    with open(report_path) as f:
        report = yaml.safe_load(f)

    assert "parent_structure" in report
    assert report["parent_structure"]["formula"] == "CaTiO3"
    assert "method" in report
    assert "results" in report
