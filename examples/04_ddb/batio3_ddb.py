"""BaTiO3 DDB writer example using MACE-r2scan.

Run from the repository root after installing MACE and placing the MACE-r2scan
model at ``~/.config/mace/mace-mh-1.model``.
"""

from pathlib import Path

from ase.io import read

from atomchain.ddb import write_ddb_from_finite_difference, write_ddb_from_phonopy
from atomchain.init_model import init_calc
from atomchain.phonon.frozenphonon import calculate_phonon


def main():
    root = Path(__file__).resolve().parents[2]
    atoms = read(root / "tests" / "fixtures" / "BaTiO3.vasp")
    output_dir = root / ".tmp" / "batio3_ddb_example"
    output_dir.mkdir(parents=True, exist_ok=True)

    calc = init_calc("mace-r2scan")
    atoms.calc = calc
    phonon_dir = output_dir / "phonon_save"
    phonon = calculate_phonon(
        atoms,
        calc=calc,
        ndim=(2, 2, 2),
        phonon_save_dir=str(phonon_dir),
        parallel=False,
    )

    write_ddb_from_phonopy(
        atoms=atoms,
        phonon=phonon,
        filename=output_dir / "BaTiO3_phonon.ddb",
        qgrid=(2, 2, 2),
    )

    write_ddb_from_finite_difference(
        atoms,
        calc,
        filename=output_dir / "BaTiO3_stress.ddb",
        phonon_ndim=(2, 2, 2),
        qgrid=(2, 2, 2),
        include_stress=True,
        include_strain_phonon=True,
        cache_dir=output_dir / "fd_cache",
        phonon_kwargs={"parallel": False},
    )


if __name__ == "__main__":
    main()
