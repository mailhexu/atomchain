#!/usr/bin/env python
"""
A simple script to relax a structure with matgl.
"""

from __future__ import annotations

import argparse

from ase.constraints import UnitCellFilter
from ase.io import Trajectory, read, write
from ase.optimize import BFGS, FIRE

try:
    from ase.constraints import FixSymmetry
except ImportError:
    from ase.spacegroup.symmetries import FixSymmetry
from atomchain.init_model import init_calc


def relax_with_ml(
    atoms,
    calc=None,
    relax_cell=True,
    sym=True,
    traj_file="relax.traj",
    model_path=None,
    fmax=0.001,
    cell_factor=1000,
    rattle=None,
    **ucf_kwargs,
):
    """
    Perform relaxation of atomic positions and/or cell shape using the FIRE algorithm.

    Args:
        atoms (ase.Atoms): The atoms object to be relaxed.
        calc (ase.Calculator): The calculator object to be used for energy and force calculations.
        relax_cell (bool, optional): Whether to relax the cell shape as well. Defaults to True.
        sym (bool, optional): Whether to impose symmetry constraints on the atoms. Defaults to True.
        traj_file (str, optional): The name of the file to write the trajectory to. Defaults to "relax.traj".
        fmax (float, optional): The maximum force allowed on each atom. Defaults to 0.001.
        cell_factor (float, optional): The factor by which to scale the unit cell when relaxing the cell shape. Defaults to 1000.
        **ucf_kwargs (dict, optional): Additional keyword arguments to pass to the UnitCellFilter constructor.

    Returns:
        ase.Atoms: The relaxed atoms object.
    """
    catoms = atoms.copy()
    if rattle is not None:
        catoms.rattle(rattle)
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    catoms.calc = calc
    if sym:
        catoms.set_constraint(FixSymmetry(catoms))
    if relax_cell:
        ecf = UnitCellFilter(catoms, cell_factor=cell_factor, **ucf_kwargs)
        opt = FIRE(ecf)
        opt.run(fmax=fmax * 10, steps=3500)
        opt = BFGS(ecf)
        opt.run(fmax=fmax, steps=5000)
    else:
        opt = FIRE(catoms)
        traj = Trajectory(traj_file, "w", catoms)
        opt.attach(traj)
        opt.run(fmax=fmax)
    return catoms


def mlrelax_cli():
    p = argparse.ArgumentParser(
        description="Relax a structure with Machine learning potentials."
    )
    p.add_argument("fname", help="input file name which contains the structure.")
    p.add_argument(
        "--model",
        "-m",
        help="type of model: m3gnet|chgnet|matgl. Default is chgnet",
        default="chgnet",
    )
    p.add_argument(
        "--sym",
        "-s",
        help="whether to impose symmetry constraints on the atoms. Default is False",
        default=False,
        action="store_true",
    )
    p.add_argument(
        "--relax_cell",
        "-r",
        help="whether to relax the cell shape as well. Default is False",
        default=False,
        action="store_true",
    )
    p.add_argument(
        "--fmax",
        "-f",
        help="The maximum force allowed on each atom. Default is 0.001",
        default=0.001,
        type=float,
    )
    p.add_argument(
        "--cell_factor",
        "-c",
        help="The factor by which to scale the unit cell when relaxing the cell shape. Default is 1000",
        default=1000,
        type=float,
    )
    p.add_argument(
        "--output_file",
        "-o",
        help="The name of the file to write the relaxed structure to. Default is POSCAR_relax.vasp",
        default="POSCAR_relax.vasp",
    )
    p.add_argument(
        "--model_path",
        "-p",
        help="The path of the model file for deepmd",
        default="model.dp",
    )
    args = p.parse_args()
    atoms = read(args.fname)
    atoms = relax_with_ml(
        atoms,
        calc=args.model,
        sym=args.sym,
        relax_cell=args.relax_cell,
        fmax=args.fmax,
        cell_factor=args.cell_factor,
        model_path=args.model_path,
    )
    write(args.output_file, atoms)


if __name__ == "__main__":
    mlrelax_cli()
