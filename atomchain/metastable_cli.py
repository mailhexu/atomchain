"""CLI for metastable states exploration.

Entry point: ``mlmetastable`` (installed via pyproject.toml console_scripts).

Usage::

    mlmetastable POSCAR -m mace-r2scan --nmax 2 --compute-phonons -o results/

See :func:`mlmetastable_cli` for full option reference.
"""

import argparse

from ase.io import read

from atomchain.init_model import init_calc
from atomchain.metastable import explore_metastable_states
from atomchain.metastable_report import generate_report, print_summary


def mlmetastable_cli():
    """CLI entry point for the ``mlmetastable`` command.

    Explores metastable structures starting from a high-symmetry parent
    structure by following imaginary phonon mode instabilities.

    Usage examples::

        # Quick run with CHGNet (default)
        mlmetastable POSCAR -o results/

        # MACE r2scan with mode pairs
        mlmetastable POSCAR -m mace-r2scan --nmax 2 -o results/

        # Full run: mode triplets + phonons for each metastable structure
        mlmetastable POSCAR -m mace-r2scan --nmax 3 --compute-phonons \\
            --fmax 0.01 -o results/

    Output files in *output_dir*:

    - ``report.yaml`` — machine-readable results
    - ``report.md`` — markdown report with embedded plots
    - ``phonon_band_structure.png`` — parent phonon dispersion
    - ``energy_bar_chart.png`` — ΔE bar chart
    - ``metastable_NNN.vasp`` — relaxed structures (VASP format)
    - ``metastable_NNN_phonon.png`` — phonon bands per structure
      (with ``--compute-phonons``)
    """
    p = argparse.ArgumentParser(
        description=(
            "Explore metastable structures from imaginary phonon modes. "
            "Given a high-symmetry parent structure, computes phonons, "
            "identifies unstable modes, generates displacements along all "
            "order parameter directions, relaxes each structure, and "
            "reports the resulting metastable states with energies and "
            "space groups."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("fname", help="input structure file (e.g. POSCAR, .cif)")
    p.add_argument(
        "--model",
        "-m",
        help="ML potential: mace-r2scan, mace, chgnet, matgl, m3gnet (default: chgnet)",
        default="chgnet",
    )
    p.add_argument(
        "--model-path",
        help="path to custom model file",
        default=None,
    )
    p.add_argument(
        "--nmax",
        type=int,
        default=2,
        help="max number of modes to combine: 1=single, 2=pairs, 3=triplets (default: 2)",
    )
    p.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="initial displacement amplitude in Angstrom (default: 0.5)",
    )
    p.add_argument(
        "--max-cell-size",
        type=int,
        default=64,
        help="max atoms in commensurate supercell (default: 64)",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        default="metastable_results",
        help="output directory (default: metastable_results)",
    )
    p.add_argument(
        "--phonon-ndim",
        nargs=3,
        type=int,
        default=[2, 2, 2],
        help="phonon supercell dimensions, e.g. 2 2 2 (default: 2 2 2)",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=0.001,
        help="relaxation force convergence in eV/Angstrom (default: 0.001)",
    )
    p.add_argument(
        "--compute-phonons",
        action="store_true",
        help="compute phonon band structures for each metastable structure (slower)",
    )
    args = p.parse_args()

    atoms = read(args.fname)
    calc = init_calc(model_type=args.model, model_path=args.model_path)

    params = {
        "nmax": args.nmax,
        "amplitude": args.amplitude,
        "max_cell_size": args.max_cell_size,
        "fmax": args.fmax,
        "phonon_ndim": args.phonon_ndim,
    }

    results = explore_metastable_states(
        atoms,
        calc=calc,
        nmax=args.nmax,
        amplitude=args.amplitude,
        max_cell_size=args.max_cell_size,
        output_dir=args.output_dir,
        phonon_ndim=args.phonon_ndim,
        relax_kwargs={"fmax": args.fmax},
        compute_phonons=args.compute_phonons,
    )

    if isinstance(results, dict):
        exploration_data = results
    else:
        exploration_data = {
            "results": results,
            "imaginary_modes": [],
            "phonon_dir": None,
        }

    parent_atoms = exploration_data.get("parent_atoms", atoms)
    report = generate_report(
        exploration_data, parent_atoms, args.model, params, args.output_dir
    )

    print_summary(report, parent_atoms)


if __name__ == "__main__":
    mlmetastable_cli()
