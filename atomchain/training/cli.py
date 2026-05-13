"""CLI for HIST conversion and MULTIBINIT training artifact workflows."""

from __future__ import annotations

import argparse

from atomchain.io.hist import hist_to_traj, traj_to_hist
from atomchain.multibinit import train_multibinit_model
from atomchain.training.artifacts import generate_multibinit_training_artifacts
from atomchain.training.samplers import generate_training_trajectory


def mlhist_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert between ABINIT HIST.nc and ASE trajectory files."
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--to", choices=["hist", "traj"], required=True)
    args = parser.parse_args(argv)
    if args.to == "hist":
        traj_to_hist(args.input, args.output)
    else:
        hist_to_traj(args.input, args.output)


def mltraining_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate MULTIBINIT training artifacts from atomchain workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser(
        "generate", help="Generate an evaluated training trajectory"
    )
    gen.add_argument("structure")
    gen.add_argument("--sources", nargs="+", default=["phonon_modes"])
    gen.add_argument("--model", default="mace-r2scan")
    gen.add_argument("--output", default="training.traj")

    art = subparsers.add_parser("artifacts", help="Generate DDB + HIST artifact bundle")
    art.add_argument("structure")
    art.add_argument("--trajectory", required=True)
    art.add_argument("--model", default="mace-r2scan")
    art.add_argument("--ddb", default="out.ddb")
    art.add_argument("--hist", default="out_HIST.nc")
    art.add_argument("--phonopy-yaml")
    art.add_argument("--output-dir", default=".")
    art.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate raw trajectory frames with --model before writing HIST",
    )

    train = subparsers.add_parser(
        "train", help="Delegate MULTIBINIT training to pymultibinit"
    )
    train.add_argument("--ddb", required=True)
    train.add_argument("--hist", required=True)
    train.add_argument("--config")
    train.add_argument("--output-dir", default="multibinit_training")

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_training_trajectory(
            args.structure, model=args.model, sources=args.sources, output=args.output
        )
    elif args.command == "artifacts":
        generate_multibinit_training_artifacts(
            args.structure,
            args.trajectory,
            model=args.model,
            ddb=args.ddb,
            hist=args.hist,
            phonopy_yaml=args.phonopy_yaml,
            output_dir=args.output_dir,
            evaluate=args.evaluate,
        )
    elif args.command == "train":
        train_multibinit_model(
            args.ddb, args.hist, config=args.config, output_dir=args.output_dir
        )
