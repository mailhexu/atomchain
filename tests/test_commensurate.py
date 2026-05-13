import numpy as np

from atomchain.commensurate import find_commensurate_matrix


def test_gamma_returns_identity():
    S = find_commensurate_matrix([[0, 0, 0]])
    assert S is not None
    np.testing.assert_array_equal(S, np.eye(3, dtype=int))


def test_x_point_returns_diag_2_1_1():
    S = find_commensurate_matrix([[0.5, 0, 0]])
    assert S is not None
    assert abs(np.linalg.det(S)) == 2
    k = np.array([0.5, 0, 0])
    result = S.T @ k
    np.testing.assert_allclose(result, np.round(result), atol=1e-8)


def test_r_point_minimal():
    S = find_commensurate_matrix([[0.5, 0.5, 0.5]])
    assert S is not None
    # Non-diagonal HNF can give det=4 (smaller than diag(2,2,2)=8)
    # e.g. [[2,1,0],[0,1,0],[0,0,2]] -> S^T @ [0.5,0.5,0.5] = [1,1,1]
    assert abs(np.linalg.det(S)) <= 8
    k = np.array([0.5, 0.5, 0.5])
    result = S.T @ k
    np.testing.assert_allclose(result, np.round(result), atol=1e-8)


def test_two_kpoints():
    kpoints = [[0.5, 0, 0], [0, 0.5, 0]]
    S = find_commensurate_matrix(kpoints)
    assert S is not None
    for k in kpoints:
        result = S.T @ np.array(k)
        np.testing.assert_allclose(result, np.round(result), atol=1e-8)


def test_no_commensurate_returns_none():
    S = find_commensurate_matrix([[0.123456789, 0, 0]], max_size=10)
    assert S is None


def test_r_and_m():
    kpoints = [[0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    S = find_commensurate_matrix(kpoints)
    assert S is not None
    for k in kpoints:
        result = S.T @ np.array(k)
        np.testing.assert_allclose(result, np.round(result), atol=1e-8)
