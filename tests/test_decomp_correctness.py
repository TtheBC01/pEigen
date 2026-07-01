import numpy as np
import numpy.testing as npt
import pytest

from peigen import decomp

sp = pytest.importorskip("scipy.sparse")


def test_svd_object_solves_least_squares():
    rng = np.random.default_rng(20)
    a = rng.standard_normal((30, 12))
    b = rng.standard_normal((30,))

    svd = decomp.SVD(a)
    x = svd.solve(b)
    x_ref, *_ = np.linalg.lstsq(a, b, rcond=None)

    npt.assert_allclose(x, x_ref, rtol=1e-9, atol=1e-9)


def test_qr_object_solves_square_system():
    rng = np.random.default_rng(21)
    a = rng.standard_normal((18, 18)) + 5 * np.eye(18)
    b = rng.standard_normal((18,))

    qr = decomp.QR(a)
    x = qr.solve(b)
    npt.assert_allclose(a @ x, b, rtol=1e-9, atol=1e-9)


def test_sparse_factorized_object():
    rng = np.random.default_rng(22)
    a = sp.random(18, 18, density=0.15, format="csc", random_state=22)
    a = a + 7 * sp.eye(18, format="csc")
    b = rng.standard_normal((18, 2))

    fac = decomp.SparseFactorized(a)
    x = fac.solve(b)
    residual = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert residual < 1e-9
