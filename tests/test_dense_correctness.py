import numpy as np
import numpy.testing as npt

from peigen import linalg


def test_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((16, 8))
    npt.assert_allclose(linalg.matmul(a, b), a @ b, rtol=1e-11, atol=1e-12)


def test_solve_matches_numpy():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 16))
    a = a + 5 * np.eye(16)
    b = rng.standard_normal((16, 4))
    for method in ("eigen", "auto"):
        npt.assert_allclose(linalg.solve(a, b, method=method), np.linalg.solve(a, b), rtol=1e-10, atol=1e-11)


def test_solve_lapack_if_available():
    from peigen import build_config

    if not build_config().get("lapack_enabled"):
        return

    rng = np.random.default_rng(11)
    a = rng.standard_normal((20, 20))
    a = a + 5 * np.eye(20)
    b = rng.standard_normal((20, 3))
    npt.assert_allclose(linalg.solve(a, b, method="lapack"), np.linalg.solve(a, b), rtol=1e-10, atol=1e-11)


def test_qr_reconstructs_matrix():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((24, 10))
    q, r = linalg.qr(a, mode="reduced")
    npt.assert_allclose(q @ r, a, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(q.T @ q, np.eye(q.shape[1]), rtol=1e-10, atol=1e-10)


def test_qr_tall_skinny_matches_numpy():
    rng = np.random.default_rng(21)
    a = rng.standard_normal((128, 32))
    q, r = linalg.qr(a, mode="reduced")
    q_ref, r_ref = np.linalg.qr(a, mode="reduced")
    npt.assert_allclose(q @ r, a, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(q @ r, q_ref @ r_ref, rtol=1e-9, atol=1e-9)
    npt.assert_allclose(q.T @ q, np.eye(q.shape[1]), rtol=1e-10, atol=1e-10)
    assert q.shape == q_ref.shape
    assert r.shape == r_ref.shape


def test_svd_reconstructs_matrix():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((20, 12))
    for method in ("bdcsvd", "auto"):
        u, s, vt = linalg.svd(a, full_matrices=False, method=method)
        npt.assert_allclose(u @ np.diag(s) @ vt, a, rtol=1e-10, atol=1e-10)


def test_svd_lapack_if_available():
    from peigen import build_config

    if not build_config().get("lapack_enabled"):
        return

    rng = np.random.default_rng(31)
    a = rng.standard_normal((24, 16))
    u, s, vt = linalg.svd(a, full_matrices=False, method="lapack")
    u_ref, s_ref, vt_ref = np.linalg.svd(a, full_matrices=False)
    npt.assert_allclose(s, s_ref, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(u @ np.diag(s) @ vt, u_ref @ np.diag(s_ref) @ vt_ref, rtol=1e-9, atol=1e-9)


def test_eigh_matches_numpy():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((14, 14))
    a = (a + a.T) / 2.0
    for method in ("eigen", "auto"):
        w, v = linalg.eigh(a, method=method)
        w_ref, v_ref = np.linalg.eigh(a)
        npt.assert_allclose(w, w_ref, rtol=1e-11, atol=1e-11)
        npt.assert_allclose(v @ v.T, v_ref @ v_ref.T, rtol=1e-10, atol=1e-10)


def test_eigh_lapack_if_available():
    from peigen import build_config

    if not build_config().get("lapack_enabled"):
        return

    rng = np.random.default_rng(41)
    a = rng.standard_normal((16, 16))
    a = (a + a.T) / 2.0
    w, v = linalg.eigh(a, method="lapack")
    w_ref, v_ref = np.linalg.eigh(a)
    npt.assert_allclose(w, w_ref, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(v @ np.diag(w) @ v.T, v_ref @ np.diag(w_ref) @ v_ref.T, rtol=1e-9, atol=1e-9)


def test_eighvals_matches_numpy():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((18, 18))
    a = (a + a.T) / 2.0
    for method in ("eigen", "auto"):
        w = linalg.eighvals(a, method=method)
        npt.assert_allclose(w, np.linalg.eigvalsh(a), rtol=1e-11, atol=1e-11)


def test_eighvals_lapack_if_available():
    from peigen import build_config

    if not build_config().get("lapack_enabled"):
        return

    rng = np.random.default_rng(44)
    a = rng.standard_normal((20, 20))
    a = (a + a.T) / 2.0
    w = linalg.eighvals(a, method="lapack")
    npt.assert_allclose(w, np.linalg.eigvalsh(a), rtol=1e-10, atol=1e-10)


def test_eigh_compute_matches_eighvals_sum():
    rng = np.random.default_rng(45)
    a = rng.standard_normal((16, 16))
    a = (a + a.T) / 2.0
    for method in ("eigen", "auto"):
        assert np.isclose(linalg.eigh_compute(a, method=method), linalg.eighvals(a, method=method).sum())


def test_eigh_upper_triangle():
    rng = np.random.default_rng(43)
    a = rng.standard_normal((12, 12))
    a = (a + a.T) / 2.0
    w, v = linalg.eigh(a, lower=False)
    w_ref, v_ref = np.linalg.eigh(a)
    npt.assert_allclose(w, w_ref, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(v @ np.diag(w) @ v.T, v_ref @ np.diag(w_ref) @ v_ref.T, rtol=1e-9, atol=1e-9)


def test_norm_matches_numpy_default():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((40, 12))
    assert np.isclose(linalg.norm(a), np.linalg.norm(a), rtol=1e-12, atol=1e-12)


def test_norm_matches_numpy_non_contiguous():
    rng = np.random.default_rng(52)
    base = rng.standard_normal((40, 12))
    a = base[::2, ::2]
    assert not a.flags.c_contiguous
    assert np.isclose(linalg.norm(a), np.linalg.norm(a), rtol=1e-12, atol=1e-12)
