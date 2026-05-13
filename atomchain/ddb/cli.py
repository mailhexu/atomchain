"""Command-line interface for DDB generation."""

from __future__ import annotations

import argparse

from ase.io import read

from atomchain.init_model import init_calc

from .finite_difference import write_ddb_from_finite_difference
from .phonopy_converter import write_ddb_from_phonopy
from .validate import validate_ddb_metadata


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate ABINIT-style DDB files from atomchain/phonopy data."
    )
    parser.add_argument("structure", help="Input structure readable by ASE")
    parser.add_argument(
        "--phonopy-yaml", help="Existing phonopy_params.yaml for phonon-only DDB"
    )
    parser.add_argument("--output", default="out.ddb", help="Output DDB filename")
    parser.add_argument("--metadata", help="Output YAML metadata filename")
    parser.add_argument(
        "--model",
        default="mace-r2scan",
        help="ML calculator model for finite-difference workflows",
    )
    parser.add_argument("--model-path", help="Optional model/config path")
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Include finite-difference stress/elastic blocks",
    )
    parser.add_argument(
        "--include-strain-phonon",
        action="store_true",
        help="Include expensive strain-phonon finite differences",
    )
    parser.add_argument(
        "--strain-amplitude",
        type=float,
        default=1e-3,
        help="Central finite-difference strain amplitude",
    )
    parser.add_argument(
        "--phonon-ndim",
        nargs=3,
        type=int,
        default=(2, 2, 2),
        help="Diagonal phonon supercell dimensions",
    )
    parser.add_argument(
        "--qgrid",
        nargs=3,
        type=int,
        help="Reduced q-point grid to include in DDB, default follows phonon supercell",
    )
    parser.add_argument(
        "--cache", default="ddb_fd_cache", help="Finite-difference cache directory"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate generated sidecar metadata"
    )
    return parser


def mlddb_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    atoms = read(args.structure)
    if args.phonopy_yaml and not (args.include_stress or args.include_strain_phonon):
        write_ddb_from_phonopy(
            atoms=atoms,
            phonopy_yaml=args.phonopy_yaml,
            filename=args.output,
            qgrid=tuple(args.qgrid) if args.qgrid else None,
            metadata_filename=args.metadata,
        )
    else:
        calc = init_calc(args.model, model_path=args.model_path)
        write_ddb_from_finite_difference(
            atoms,
            calc,
            filename=args.output,
            phonon_ndim=tuple(args.phonon_ndim),
            qgrid=tuple(args.qgrid) if args.qgrid else None,
            strain_amplitude=args.strain_amplitude,
            include_stress=args.include_stress,
            include_strain_phonon=args.include_strain_phonon,
            cache_dir=args.cache,
        )
    if args.validate:
        metadata = args.metadata or f"{args.output}.yaml"
        errors = validate_ddb_metadata(metadata)
        if errors:
            parser.exit(2, "\n".join(errors) + "\n")


if __name__ == "__main__":
    mlddb_cli()
