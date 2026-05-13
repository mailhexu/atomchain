from atomchain.ddb.cli import build_parser


def test_mlddb_help_documents_key_options():
    help_text = build_parser().format_help()
    assert "--phonopy-yaml" in help_text
    assert "--model" in help_text
    assert "--include-stress" in help_text
    assert "--include-strain-phonon" in help_text
    assert "--strain-amplitude" in help_text


def test_mlddb_finite_difference_flags_parse():
    args = build_parser().parse_args(
        [
            "BaTiO3.vasp",
            "--model",
            "mace-r2scan",
            "--include-stress",
            "--include-strain-phonon",
            "--phonon-ndim",
            "2",
            "2",
            "2",
        ]
    )
    assert args.model == "mace-r2scan"
    assert args.include_stress
    assert args.include_strain_phonon
    assert args.phonon_ndim == [2, 2, 2]
