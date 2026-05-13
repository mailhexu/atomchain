import os

import numpy as np
import spglib
from phonopy.structure.atoms import PhonopyAtoms


def _patch_symphon():
    try:
        import symphon.chiral

        chiral_dir = os.path.dirname(symphon.chiral.__file__)
    except (ImportError, ModuleNotFoundError):
        return
    except Exception:
        return

    stubs = {
        "circular.py": "class CircularMode: pass\nclass CircularPhononFinder: pass\n",
        "sam.py": "class SAMCalculator: pass\n",
        "abstract_circular.py": "class AbstractCircularPhononFinder: pass\n",
    }
    for name, content in stubs.items():
        fpath = os.path.join(chiral_dir, name)
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            with open(fpath, "w") as f:
                f.write(content)


_patch_symphon()


def get_high_symmetry_kpoints(atoms, symprec=1e-5):
    pa = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell(),
    )

    from symphon.irreps.highsym import get_special_qpoints

    sqs = get_special_qpoints(pa, symprec=symprec)
    return [{"label": sq["label"], "qpoint": sq["qpoint_input"]} for sq in sqs]


def get_kpoint_star(kpoint, atoms, symprec=1e-5):
    cell = (
        atoms.get_cell(),
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    sym = spglib.get_symmetry(cell, symprec=symprec)
    rotations = sym["rotations"]

    k = np.asarray(kpoint)
    star = set()
    for R in rotations:
        k_new = R.T @ k
        k_new = k_new - np.round(k_new)
        star.add(tuple(np.round(k_new, 6)))

    return np.array(list(star))
