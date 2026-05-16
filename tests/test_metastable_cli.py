import numpy as np
import yaml
from ase import Atoms


def test_mlmetastable_cli_uses_same_output_dir_for_explore_and_report(
    monkeypatch, tmp_path
):
    from atomchain import metastable_cli

    atoms = Atoms(
        "ZrSi", positions=[[0, 0, 0], [1, 1, 1]], cell=np.eye(3) * 4, pbc=True
    )
    output_dir = tmp_path / "zrsi_results"
    captured = {}

    def mock_explore(input_atoms, **kwargs):
        captured["explore_atoms"] = input_atoms
        captured["explore_kwargs"] = kwargs
        return {
            "results": [],
            "imaginary_modes": [],
            "phonon_dir": str(output_dir / "parent_phonon_save"),
        }

    def mock_report(
        exploration_data, parent_atoms, calc_name, params, report_output_dir
    ):
        captured["report_exploration_data"] = exploration_data
        captured["report_parent_atoms"] = parent_atoms
        captured["report_output_dir"] = report_output_dir
        return {"parent_structure": {"formula": "ZrSi", "n_atoms": 2}, "results": []}

    monkeypatch.setattr("sys.argv", ["mlmetastable", "ZrSi.cif", "-o", str(output_dir)])
    monkeypatch.setattr(metastable_cli, "read", lambda fname: atoms)
    monkeypatch.setattr(metastable_cli, "init_calc", lambda **kwargs: object())
    monkeypatch.setattr(metastable_cli, "explore_metastable_states", mock_explore)
    monkeypatch.setattr(metastable_cli, "generate_report", mock_report)
    monkeypatch.setattr(metastable_cli, "print_summary", lambda report, atoms: None)

    metastable_cli.mlmetastable_cli()

    assert captured["explore_kwargs"]["output_dir"] == str(output_dir)
    assert captured["report_output_dir"] == str(output_dir)
    assert captured["report_exploration_data"]["phonon_dir"] == str(
        output_dir / "parent_phonon_save"
    )
    assert captured["report_parent_atoms"] is atoms


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
            "initial_atoms": atoms,
            "is_still_imaginary": False,
            "atoms": atoms,
        }
    ]
    params = {"nmax": 1, "amplitude": 0.3, "max_cell_size": 32}

    generate_report(results, atoms, "mace", params, str(tmp_path))

    assert (tmp_path / "initial_structures" / "initial_001.vasp").exists()
    assert (tmp_path / "relaxed_structures" / "relaxed_001.vasp").exists()
    assert (tmp_path / "initial_structures" / "README.md").exists()
    assert (tmp_path / "relaxed_structures" / "README.md").exists()

    with open(tmp_path / "report.yaml") as f:
        report = yaml.safe_load(f)
    result = report["results"][0]
    assert result["initial_structure_file"] == "initial_structures/initial_001.vasp"
    assert result["relaxed_structure_file"] == "relaxed_structures/relaxed_001.vasp"


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


def test_markdown_energy_table_shows_top_10_lowest_energy_structures(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    results = []
    for i in range(12):
        results.append(
            {
                "id": i + 1,
                "status": "success",
                "energy": -100.0 + i,
                "energy_per_fu": -100.0 + i,
                "delta_e_per_fu": float(i),
                "spacegroup_number": 225,
                "spacegroup_name": "Fm-3m",
                "n_atoms": 1,
                "source_modes": [],
                "combined_label": f"mode-{i + 1}",
                "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "atoms": atoms,
            }
        )

    generate_report(results, atoms, "chgnet", {"nmax": 1}, str(tmp_path))

    markdown = (tmp_path / "report.md").read_text()
    assert "Lowest-Energy Structures" in markdown
    assert "Showing the 10 lowest-energy relaxed candidates" in markdown
    assert "mode-10" in markdown
    assert "mode-11" not in markdown
    assert "mode-12" not in markdown


def test_generate_report_groups_equivalent_relaxed_structures(tmp_path):
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
            "n_atoms": 1,
            "source_modes": [],
            "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "atoms": atoms,
            "initial_atoms": atoms,
        },
        {
            "id": 2,
            "energy": -10.0004,
            "energy_per_fu": -10.0004,
            "delta_e_per_fu": -1.0004,
            "spacegroup_number": 225,
            "spacegroup_name": "Fm-3m",
            "n_atoms": 1,
            "source_modes": [],
            "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "atoms": atoms,
            "initial_atoms": atoms,
        },
    ]

    report = generate_report(results, atoms, "chgnet", {"nmax": 1}, str(tmp_path))

    assert len(report["results"]) == 2
    assert (
        report["results"][0]["relaxed_group_id"]
        == report["results"][1]["relaxed_group_id"]
    )
    assert report["results"][0]["relaxed_group_size"] == 2
    assert report["relaxed_structure_groups"][0]["member_ids"] == [1, 2]

    markdown = (tmp_path / "report.md").read_text()
    assert "same relaxed group" in markdown
    assert "Relaxed Structure Groups" in markdown


def test_generate_report_includes_kpoint_and_symphon_tables(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    exploration_data = {
        "results": [],
        "imaginary_modes": [],
        "ase_kpoints": [
            {"label": "G", "qpoint": [0.0, 0.0, 0.0]},
            {"label": "X", "qpoint": [0.0, 0.0, 0.5]},
        ],
        "bcs_kpoints": [
            {"label": "GM", "qpoint": [0.0, 0.0, 0.0]},
            {"label": "Gamma", "qpoint": [0.0, 0.0, 0.0]},
            {"label": "X", "qpoint": [0.0, 0.0, 0.5]},
        ],
        "bcs_labeled_modes": {
            "GM": {
                "frequencies": np.array([0.0, 1.0]),
                "modes": [
                    {
                        "band_index": 0,
                        "frequency": 0.0,
                        "bcs_label": "GM1+",
                        "mulliken_label": "A1g",
                    },
                    {
                        "band_index": 1,
                        "frequency": 1.0,
                        "bcs_label": "GM2+",
                        "mulliken_label": None,
                    },
                ],
            }
        },
    }

    report = generate_report(
        exploration_data, atoms, "chgnet", {"nmax": 1}, str(tmp_path)
    )

    assert report["ase_kpoints"][0]["label"] == "G"
    assert report["bcs_kpoints"][0]["label"] == "GM"
    assert [item["label"] for item in report["bcs_kpoints"]] == ["GM", "X"]
    assert report["bcs_labeled_modes"][0]["modes"][0]["bcs_label"] == "GM1+"

    markdown = (tmp_path / "report.md").read_text()
    assert "High-Symmetry Q-Point Coverage" in markdown
    assert "ASE Band-Path Special Points" in markdown
    assert "BCS/Symphon Special Points" in markdown
    assert "Symphon Mode Labels At BCS Points" in markdown


def test_generate_report_includes_failed_candidate_status(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    results = [
        {
            "id": 1,
            "status": "failed",
            "error_stage": "relaxation",
            "error_message": "optimizer failed",
            "attempt": 3,
            "amplitude": 0.125,
            "energy": None,
            "energy_per_fu": None,
            "delta_e_per_fu": None,
            "spacegroup_number": None,
            "spacegroup_name": None,
            "n_atoms": 1,
            "source_modes": [],
            "combined_label": "X1(a)",
            "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "initial_structure_file": "initial_structures/initial_001.vasp",
            "initial_atoms": atoms,
        }
    ]

    report = generate_report(results, atoms, "chgnet", {"nmax": 1}, str(tmp_path))

    row = report["results"][0]
    assert row["status"] == "failed"
    assert row["error_stage"] == "relaxation"
    assert row["attempt"] == 3
    assert row["amplitude"] == 0.125
    assert row["relaxed_structure_file"] is None

    markdown = (tmp_path / "report.md").read_text()
    assert "failed" in markdown
    assert "optimizer failed" in markdown


def test_generate_report_handles_mixed_failed_and_success_spacegroups(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    results = [
        {
            "id": 1,
            "status": "success",
            "energy": -10.0,
            "energy_per_fu": -10.0,
            "delta_e_per_fu": 0.0,
            "spacegroup_number": 225,
            "spacegroup_name": "Fm-3m",
            "n_atoms": 1,
            "source_modes": [],
            "combined_label": "X1(a)",
            "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "atoms": atoms,
        },
        {
            "id": 2,
            "status": "failed",
            "error_stage": "relaxation",
            "error_message": "optimizer failed",
            "attempt": 3,
            "amplitude": 0.025,
            "delta_e_per_fu": None,
            "spacegroup_number": None,
            "spacegroup_name": None,
            "n_atoms": 1,
            "source_modes": [],
            "combined_label": "M1(a)",
            "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "initial_atoms": atoms,
        },
    ]

    generate_report(results, atoms, "chgnet", {"nmax": 1}, str(tmp_path))

    markdown = (tmp_path / "report.md").read_text()
    assert "Space Groups Found" in markdown
    assert "Fm-3m" in markdown
    assert "optimizer failed" in markdown


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


def test_report_uses_formula_specific_figure_names(tmp_path, monkeypatch):
    from atomchain import metastable_report
    from atomchain.metastable_cli import generate_report

    atoms = Atoms(
        "ZrSi", positions=[[0, 0, 0], [1, 1, 1]], cell=np.eye(3) * 4, pbc=True
    )
    phonon_dir = tmp_path / "parent_phonon_save"
    phonon_dir.mkdir()
    (phonon_dir / "phonopy_params.yaml").write_text("phonopy:\n")

    def mock_plot_phonon(path, figname, show, units):
        figure_path = tmp_path / figname
        if path == str(phonon_dir):
            figure_path = phonon_dir / figname
        figure_path.write_bytes(b"png")

    monkeypatch.setattr("atomchain.phonon.plotphonopy.plot_phonon", mock_plot_phonon)
    monkeypatch.setattr(metastable_report, "_plot_energy_bar_chart", lambda *args: None)

    report = generate_report(
        {"results": [], "imaginary_modes": [], "phonon_dir": str(phonon_dir)},
        atoms,
        "chgnet",
        {"nmax": 1},
        str(tmp_path),
    )

    assert report["phonon_band_structure"] == "ZrSi_phonon_band_structure.png"
    markdown = (tmp_path / "report.md").read_text()
    assert "ZrSi_phonon_band_structure.png" in markdown
    assert "](phonon_band_structure.png)" not in markdown


def test_generate_report_source_modes_in_output(tmp_path):
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
                    "kpoint": [0.5, 0.5, 0.5],
                    "kpoint_label": "R",
                    "band_index": 7,
                    "frequency": -3.45,
                    "bcs_label": "R4-",
                    "mulliken_label": "T2g",
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
    assert modes[0]["mulliken_label"] == "T2g"
    assert modes[0]["frequency_THz"] == -3.45
    assert modes[1]["kpoint_label"] == "G"
    assert report["results"][0]["mode_indices"] == "R:7, G:2"
    assert report["results"][0]["mode_labels"] == "R4-/T2g, G"

    markdown = (tmp_path / "report.md").read_text()
    assert "Mode Index" in markdown
    assert "Mode Labels" in markdown
    assert "R:7" in markdown


def test_generate_report_imaginary_modes_include_qpoint_index_and_labels(tmp_path):
    from atomchain.metastable_cli import generate_report

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    exploration_data = {
        "results": [],
        "imaginary_modes": [
            {
                "kpoint": [0.0, 0.0, 0.5],
                "kpoint_label": "X",
                "band_index": 4,
                "frequency": -1.25,
                "bcs_label": "X3-",
                "mulliken_label": "B2u",
                "degeneracy": 1,
            }
        ],
    }

    report = generate_report(
        exploration_data, atoms, "chgnet", {"nmax": 1}, str(tmp_path)
    )

    mode = report["imaginary_modes"][0]
    assert mode["kpoint"] == [0.0, 0.0, 0.5]
    assert mode["band_index"] == 4
    assert mode["bcs_label"] == "X3-"
    assert mode["mulliken_label"] == "B2u"

    markdown = (tmp_path / "report.md").read_text()
    assert "Band index" in markdown
    assert "[0, 0, 0.5]" in markdown
    assert "X3-" in markdown
    assert "B2u" in markdown


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
