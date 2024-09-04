import argparse

import matgl
import torch
from ase.io import read


def ase_to_pymatgen(atoms):
    """
    ase atoms transform into pymatgen structure
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(atoms)


XCdict = {"PBE": 0, "GLLB-SC": 1, "HSE": 2, "SCAN": 3}


class MatGLGapPredictor:
    def __init__(self):
        self._model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

    def predict_gap(self, atoms, xc):
        struct = ase_to_pymatgen(atoms)
        graph_attrs = torch.tensor([XCdict[xc]])
        # For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
        bandgap = self._model.predict_structure(
            structure=struct, state_feats=graph_attrs
        )
        # print(f"The predicted {xc} band gap for CsCl is {float(bandgap):.3f} eV.")
        return float(bandgap)


def predict_gap(atoms, xc="PBE"):
    p = MatGLGapPredictor()
    return p.predict_gap(atoms, xc)


def mlgap_cli():
    parser = argparse.ArgumentParser(description="Predict band gap using MatGL")
    parser.add_argument("filename", type=str, help="The filename of the structure")
    parser.add_argument(
        "--xc",
        type=str,
        default="PBE",
        help="The exchange-correlation functional to use",
    )
    args = parser.parse_args()
    atoms = read(args.filename)
    gap = predict_gap(atoms, args.xc)
    print(f"The predicted {args.xc} band gap for {args.filename} is {gap:.3f} eV.")


if __name__ == "__main__":
    mlgap_cli()
