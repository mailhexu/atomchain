"""
Machine learning potentials
"""

from atomchain.init_model import init_calc
from atomchain.phonon.mlphonon import phonon_with_ml
from atomchain.relax import relax_with_ml
from atomchain.gap import predict_gap
from atomchain.singlepoint import calculate_single_point
from atomchain.supercell import make_supercell_structure
from atomchain.rattle import generate_rattle_dataset
from atomchain.batch import calculate_trajectory_batch
from atomchain.compare import compare_trajectories
from atomchain.neb import calculate_neb

__all__ = [
    "init_calc",
    "phonon_with_ml",
    "relax_with_ml",
    "predict_gap",
    "calculate_single_point",
    "make_supercell_structure",
    "generate_rattle_dataset",
    "calculate_trajectory_batch",
    "compare_trajectories",
    "calculate_neb",
]
