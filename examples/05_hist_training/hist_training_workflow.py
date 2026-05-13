"""Small ABINIT HIST training-artifact workflow example.

This example uses a synthetic calculator so it is fast and dependency-light.
Replace ``ConstantCalculator`` with ``init_calc("mace-r2scan")`` for production
data generation.
"""

from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from atomchain.io.hist import read_abinit_hist, write_abinit_hist
from atomchain.training import evaluate_training_frames, generate_training_trajectory


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = -float(len(atoms))
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = np.zeros(6)


def main():
    output = Path("hist_training_example")
    output.mkdir(exist_ok=True)
    parent = bulk("Al", "fcc", a=4.05, cubic=True)
    frames = generate_training_trajectory(
        parent,
        sources=["md", "phonon_modes"],
        evaluate=False,
        output=output / "raw_training.traj",
    )
    evaluated = evaluate_training_frames(
        frames, calculator=ConstantCalculator(), output=output / "training.traj"
    )
    write_abinit_hist(
        evaluated, output / "training_HIST.nc", metadata=output / "training_HIST.yaml"
    )
    loaded = read_abinit_hist(output / "training_HIST.nc")
    print(f"Wrote {len(loaded)} HIST frames to {output}")


if __name__ == "__main__":
    main()
