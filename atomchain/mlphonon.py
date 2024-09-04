#!/usr/bin/env python
"""
A simple script to relax a structure with matgl.
"""

import argparse

import numpy as np
from ase.io import read
from phonopy.units import VaspToTHz

from atomchain.frozenphonon import calculate_phonon
from atomchain.init_model import init_calc
from atomchain.relax import relax_with_ml


def phonon_with_ml(
    atoms,
    calc=None,
    relax=False,
    plot=True,
    knames=None,
    kvectors=None,
    npoints=100,
    figname="phonon.pdf",
    **kwargs,
):
    """
    Perform phonon calculation using the given calculator object.

    Args:
        atoms (ase.Atoms): The atoms object to calculate the phonons for.
        calc (ase.Calculator): The calculator object to be used for energy and force calculations.
        relax (bool, optional): Whether to relax the atomic positions and cell shape before calculating the phonons. Defaults to False.

    Returns:
        ase.Atoms: The relaxed atoms object.
    """
    if isinstance(calc, str):
        calc = init_calc(model_type=calc)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    else:
        pass
    if relax:
        atoms = relax_with_ml(atoms, calc)
    phon_args = dict(
        forces_set_file=None,
        ndim=np.diag([2, 2, 2]),
        primitive_matrix=np.eye(3),
        distance=0.05,
        factor=VaspToTHz,
        is_plusminus="auto",
        is_symmetry=True,
        symprec=1e-3,
        func=None,
        prepare_initial_wavecar=False,
        skip=None,
        restart=False,
        parallel=False,
        sc_mag=None,
        mask_force=[1, 1, 1],
    )
    phon_args.update(kwargs)
    calculate_phonon(atoms, calc=calc, **phon_args)

    if plot:
        from pyDFTutils.phonon.plotphonopy import plot_phonon

        plot_phonon(
            path="./",
            knames=knames,
            kvectors=kvectors,
            npoints=npoints,
            figname=figname,
            show=True,
        )


def mlphonon_cli():
    p = argparse.ArgumentParser(
        description="Compute the phonon of a structure with machine learning potential."
    )
    p.add_argument("fname", help="input file name which contains the structure.")
    p.add_argument(
        "--model",
        "-m",
        help="type of model: m3gnet|chgnet|matgl. Default is chgnet",
        default="chgnet",
    )
    p.add_argument(
        "--relax",
        "-r",
        help="relax the structure before computing the phonon.",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--ndim",
        "-n",
        help="number of repetitions of the structure in each direction.",
        nargs=3,
        type=int,
        default=[2, 2, 2],
    )
    p.add_argument(
        "--knames",
        "-k",
        help="special kpoints names for the band structure plot.",
        default=None,
    )
    p.add_argument(
        "--npoints",
        "-p",
        help="number of points in the band structure plot.",
        type=int,
        default=100,
    )
    p.add_argument(
        "--figname", "-f", help="name of the band structure plot.", default="phonon.pdf"
    )
    args = p.parse_args()
    atoms = read(args.fname)
    atoms = phonon_with_ml(
        atoms,
        calc=args.model,
        relax=args.relax,
        ndim=np.diag(args.ndim),
        knames=args.knames,
        npoints=args.npoints,
        figname=args.figname,
    )


if __name__ == "__main__":
    mlphonon_cli()
