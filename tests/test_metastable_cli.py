import numpy as np
import yaml
from ase import Atoms


def test_generate_report(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    results = [
        {
            "id": 1,
            "energy": -10.5,
            "energy_per_fu": -10.5,
            "delta_e_per_fu": 0.0,
            "spacegroup_number": 225,
            "spacegroup_name": "Fm-3m",
            "n_atoms": 4,
            "source_modes": [
                {
                    "kpoint": [0.5, 0, 0],
                    "kpoint_label": "X",
                    "band_index": 3,
                    "frequency": -1.2,
                    "bcs_label": "X2",
                }
            ],
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "structure_file": "metastable_001.vasp",
            "is_still_imaginary": False,
            "atoms": atoms,
        }
    ]
    params = {"nmax": 1, "amplitude": 0.5, "max_cell_size": 64}

    generate_report(results, atoms, "chgnet", params, str(tmp_path))

    report_path = tmp_path / "report.yaml"
    assert report_path.exists()

    with open(report_path) as f:
        report = yaml.safe_load(f)

    assert "parent_structure" in report
    assert "method" in report
    assert "results" in report
    assert len(report["results"]) == 1
    assert report["results"][0]["relaxed_spacegroup"] == "Fm-3m (#225)"


def test_generate_report_creates_vasp_files(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    results = [
        {
            "id": 1,
            "energy": -5.0,
            "spacegroup_number": 225,
            "spacegroup_name": "Fm-3m",
            "n_atoms": 1,
            "source_modes": [],
            "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "structure_file": "metastable_001.vasp",
            "is_still_imaginary": False,
            "atoms": atoms,
        }
    ]
    params = {"nmax": 1, "amplitude": 0.3, "max_cell_size": 32}

    generate_report(results, atoms, "mace", params, str(tmp_path))

    vasp_path = tmp_path / "metastable_001.vasp"
    assert vasp_path.exists()


def test_generate_report_multiple_results(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    results = [
        {
            "id": 1,
            "energy": -10.0,
            "energy_per_fu": -10.0,
            "delta_e_per_fu": -1.0,
            "spacegroup_number": 225,
            "spacegroup_name": "Fm-3m",
            "n_atoms": 4,
            "source_modes": [],
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "structure_file": "metastable_001.vasp",
            "is_still_imaginary": False,
            "atoms": atoms,
        },
        {
            "id": 2,
            "energy": -9.5,
            "energy_per_fu": -9.5,
            "delta_e_per_fu": -0.5,
            "spacegroup_number": 221,
            "spacegroup_name": "Pm-3m",
            "n_atoms": 4,
            "source_modes": [],
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "structure_file": "metastable_002.vasp",
            "is_still_imaginary": True,
            "atoms": atoms,
        },
    ]
    params = {"nmax": 2, "amplitude": 0.5, "max_cell_size": 64}

    generate_report(results, atoms, "chgnet", params, str(tmp_path))

    with open(tmp_path / "report.yaml") as f:
        report = yaml.safe_load(f)

    assert len(report["results"]) == 2
    assert report["results"][0]["delta_e_per_fu"] == -1.0
    assert report["results"][1]["delta_e_per_fu"] == -0.5
    assert report["method"]["calculator"] == "chgnet"


def test_generate_report_parent_structure_info(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    results = []
    params = {"nmax": 1, "amplitude": 0.5, "max_cell_size": 64}

    generate_report(results, atoms, "chgnet", params, str(tmp_path))

    with open(tmp_path / "report.yaml") as f:
        report = yaml.safe_load(f)

    assert report["parent_structure"]["formula"] == "Al"
    assert report["parent_structure"]["n_atoms"] == 1


def test_generate_report_source_modes_in_output(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    results = [
        {
            "id": 1,
            "energy": -10.5,
            "spacegroup_number": 225,
            "spacegroup_name": "Fm-3m",
            "n_atoms": 4,
            "source_modes": [
                {
                    "kpoint": [0.5, 0.5, 0.5],
                    "kpoint_label": "R",
                    "band_index": 7,
                    "frequency": -3.45,
                    "bcs_label": "R4-",
                },
                {
                    "kpoint": [0.0, 0.0, 0.0],
                    "kpoint_label": "G",
                    "band_index": 2,
                    "frequency": -1.0,
                },
            ],
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "structure_file": "metastable_001.vasp",
            "is_still_imaginary": False,
            "atoms": atoms,
        }
    ]
    params = {"nmax": 2, "amplitude": 0.5, "max_cell_size": 64}

    generate_report(results, atoms, "mace", params, str(tmp_path))

    with open(tmp_path / "report.yaml") as f:
        report = yaml.safe_load(f)

    modes = report["results"][0]["source_modes"]
    assert len(modes) == 2
    assert modes[0]["bcs_label"] == "R4-"
    assert modes[0]["frequency_THz"] == -3.45
    assert modes[1]["kpoint_label"] == "G"


def test_print_summary(capsys):
    from atomchain.metastable_report import print_summary as _print_summary

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    report = {
        "parent_structure": {
            "formula": "Al",
            "n_atoms": 1,
            "spacegroup": "Pm-3m",
            "spacegroup_number": 221,
        },
        "method": {"calculator": "chgnet", "nmax": 1, "amplitude": 0.5},
        "results": [
            {
                "id": 1,
                "delta_e_per_fu": -0.5,
                "initial_spacegroup": "Pm-3m (#221)",
                "relaxed_spacegroup": "Fm-3m (#225)",
                "n_atoms": 4,
                "combined_label": "X2-(a,0,0)",
                "source_modes": [
                    {
                        "kpoint": [0.5, 0, 0],
                        "kpoint_label": "X",
                        "band_index": 3,
                        "frequency_THz": -1.2,
                    }
                ],
            }
        ],
    }

    _print_summary(report, atoms)
    captured = capsys.readouterr()
    assert "Metastable States" in captured.out
    assert "Al" in captured.out
    assert "chgnet" in captured.out
    assert "Fm-3m" in captured.out
