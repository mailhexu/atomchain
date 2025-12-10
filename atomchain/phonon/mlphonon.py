#!/usr/bin/env python
"""
A simple script to relax a structure with matgl.
"""

import argparse

import numpy as np
from ase.io import read
from phonopy.units import VaspToTHz

from atomchain.phonon.frozenphonon import calculate_phonon
from atomchain.init_model import init_calc
from atomchain.relax import relax_with_ml


def phonon_with_ml(
    atoms,
    calc=None,
    model_path=None,
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
        calc (ase.Calculator or str): The calculator object to be used for energy and force calculations,
            or a string specifying the model type (e.g., 'chgnet', 'multibinit').
        model_path (str, optional): Path to the model file/configuration. Required for some models
            like 'multibinit'. Defaults to None.
        relax (bool, optional): Whether to relax the atomic positions and cell shape before calculating the phonons. Defaults to False.
        plot (bool, optional): Whether to plot the phonon band structure. Defaults to True.
        knames (str or list, optional): K-point path specification for band structure. Defaults to None.
        kvectors (array-like, optional): Custom k-vectors for band structure. Defaults to None.
        npoints (int, optional): Number of points in band structure plot. Defaults to 100.
        figname (str, optional): Filename for phonon plot. Defaults to "phonon.pdf".
        **kwargs: Additional arguments passed to calculate_phonon.

    Returns:
        ase.Atoms: The (potentially relaxed) atoms object.
    """
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
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

    # Special handling for MultibinitPotential to fix supercell reference structure
    # This fixes the issue where Phonopy generates element-grouped supercells
    # while MULTIBINIT expects unit-cell-grouped supercells (ASE-style).
    potential = None
    if hasattr(calc, "potential"):
         potential = calc.potential
    elif hasattr(calc, "calc") and hasattr(calc.calc, "potential"):
         potential = calc.calc.potential
    elif hasattr(calc, "__class__") and "MultibinitPotential" in calc.__class__.__name__:
         potential = calc

    if potential is not None and "MultibinitPotential" in potential.__class__.__name__:
        try:
            # Extract supercell dimensions
            ndim_arg = phon_args.get("ndim", np.diag([2, 2, 2]))
            
            # Convert ndim to tuple of integers for repeat()
            repeat_tuple = None
            if isinstance(ndim_arg, np.ndarray):
                if ndim_arg.shape == (3, 3):
                    # Extract diagonal elements for supercell size
                    nx = int(np.round(np.linalg.norm(ndim_arg[:, 0])))
                    ny = int(np.round(np.linalg.norm(ndim_arg[:, 1])))
                    nz = int(np.round(np.linalg.norm(ndim_arg[:, 2])))
                    repeat_tuple = (nx, ny, nz)
                else:
                    repeat_tuple = tuple(int(x) for x in ndim_arg.flatten()[:3])
            elif isinstance(ndim_arg, (list, tuple)):
                if len(ndim_arg) == 3:
                     repeat_tuple = tuple(int(x) for x in ndim_arg)
                
            if repeat_tuple:
                # Create ASE supercell reference (has correct ordering for MULTIBINIT)
                # atoms is the unit cell here
                print(f"PyMultibinit: Setting reference structure for {repeat_tuple} supercell")
                ase_supercell = atoms.repeat(repeat_tuple)
                
                potential.set_reference_structure(
                    positions=ase_supercell.positions,
                    lattice=ase_supercell.cell.array
                )
        except Exception as e:
            print(f"PyMultibinit WARNING: Failed to set reference structure: {e}")

    calculate_phonon(atoms, calc=calc, **phon_args)

    if plot:
        from atomchain.phonon.plotphonopy import plot_phonon

        # Construct kpath string from knames if provided
        kpath = knames if isinstance(knames, str) else None

        plot_phonon(
            path="phonon_save",
            kpath=kpath,
            npoints=npoints,
            figname=figname,
            show=True,
        )
    
    return atoms


def mlphonon_cli():
    p = argparse.ArgumentParser(
        description="Compute the phonon of a structure with machine learning potential."
    )
    p.add_argument("fname", help="input file name which contains the structure.")
    p.add_argument(
        "--model",
        "-m",
        help="type of model: m3gnet|chgnet|matgl|multibinit. Default is chgnet",
        default="chgnet",
    )
    p.add_argument(
        "--model_path",
        help="path to model file/configuration (required for some models like multibinit)",
        default=None,
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
        "--kpath",
        "-k",
        help="k-path string for band structure plot (e.g., 'GXMG'). If not specified, automatic detection is used.",
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
        model_path=args.model_path,
        relax=args.relax,
        ndim=np.diag(args.ndim),
        knames=args.kpath,
        npoints=args.npoints,
        figname=args.figname,
    )


if __name__ == "__main__":
    mlphonon_cli()
