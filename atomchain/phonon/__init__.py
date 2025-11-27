"""
Phonon calculation module for AtomChain.

This module provides functionality for phonon calculations using machine learning
potentials and the frozen phonon method.
"""

from atomchain.phonon.frozenphonon import calculate_phonon, compute_phonon_at_qpoints
from atomchain.phonon.mlphonon import phonon_with_ml, mlphonon_cli
from atomchain.phonon.plotphonopy import plot_phonon

__all__ = [
    "calculate_phonon",
    "compute_phonon_at_qpoints",
    "phonon_with_ml",
    "mlphonon_cli",
    "plot_phonon",
]
