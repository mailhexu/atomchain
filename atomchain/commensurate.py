from __future__ import annotations

from typing import Optional

import numpy as np


def find_commensurate_matrix(kpoints, max_size: int = 64) -> Optional[np.ndarray]:
    kpoints = np.atleast_2d(kpoints)
    kpoints = kpoints.astype(float)

    for d in range(1, max_size + 1):
        result = _search_det(kpoints, d)
        if result is not None:
            return result

    return None


def _search_det(kpoints: np.ndarray, d: int) -> Optional[np.ndarray]:
    for a in range(1, d + 1):
        if d % a != 0:
            continue
        remainder_a = d // a
        for e in range(1, remainder_a + 1):
            if remainder_a % e != 0:
                continue
            g = remainder_a // e

            for b in range(a):
                for c in range(e):
                    for f in range(e):
                        S = np.array([[a, b, c], [0, e, f], [0, 0, g]], dtype=int)
                        if _check_commensurate(kpoints, S):
                            return S

    return None


def _check_commensurate(kpoints: np.ndarray, S: np.ndarray) -> bool:
    product = (S.T @ kpoints.T).T
    return np.allclose(product, np.round(product), atol=1e-8)
