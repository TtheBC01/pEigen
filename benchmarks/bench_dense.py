"""Optional dense benchmarks comparing pEigen and NumPy."""

from __future__ import annotations

import time

import numpy as np

from peigen import build_config, linalg

from _table import compare_row, print_compare_section


def _timed(fn, *args, warmup: int = 3, runs: int = 10):
    for _ in range(warmup):
        fn(*args)

    durations = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        durations.append((time.perf_counter() - t0) * 1000.0)

    durations.sort()
    p50 = durations[len(durations) // 2]
    p90 = durations[int(len(durations) * 0.9) - 1]
    return p50, p90


def _svd_methods():
    cfg = build_config()
    methods = ["bdcsvd"]
    if cfg.get("lapack_enabled"):
        methods.extend(["lapack", "auto"])
    return methods


def _eigh_methods():
    cfg = build_config()
    methods = ["eigen"]
    if cfg.get("lapack_enabled"):
        methods.extend(["lapack", "auto"])
    return methods


def _lapack_eigen_methods():
    cfg = build_config()
    methods = ["eigen"]
    if cfg.get("lapack_enabled"):
        methods.extend(["lapack", "auto"])
    return methods


# (label, a_shape, b_shape) for matmul; b_shape omitted for SVD entries
MATMUL_CASES = [
    ("256x256", (256, 256), (256, 256)),
    ("512x512", (512, 512), (512, 512)),
    ("1024x1024", (1024, 1024), (1024, 1024)),
    ("1024x256@256x128", (1024, 256), (256, 128)),
    ("512x128@128x64", (512, 128), (128, 64)),
    ("256x1024@1024x512", (256, 1024), (1024, 512)),
]

SVD_CASES = [
    ("256x256", (256, 256)),
    ("512x512", (512, 512)),
    ("1024x1024", (1024, 1024)),
    ("1024x256", (1024, 256)),
    ("512x128", (512, 128)),
    ("256x1024", (256, 1024)),
]

QR_CASES = [
    ("256x256", (256, 256)),
    ("512x512", (512, 512)),
    ("1024x1024", (1024, 1024)),
    ("2048x2048", (2048, 2048)),
    ("2048x1024", (2048, 1024)),
    ("1024x256", (1024, 256)),
    ("512x128", (512, 128)),
]

EIGH_CASES = [
    ("256x256", (256, 256)),
    ("512x512", (512, 512)),
    ("1024x1024", (1024, 1024)),
]

SOLVE_CASES = [
    ("256x256", (256, 256), (256, 1)),
    ("512x512", (512, 512), (512, 1)),
    ("1024x1024", (1024, 1024), (1024, 1)),
    ("2048x2048", (2048, 2048), (2048, 1)),
    ("256x256@4rhs", (256, 256), (256, 4)),
    ("512x512@4rhs", (512, 512), (512, 4)),
    ("1024x1024@4rhs", (1024, 1024), (1024, 4)),
    ("2048x2048@4rhs", (2048, 2048), (2048, 4)),
]

NORM_CASES = [
    ("256x256", (256, 256)),
    ("512x512", (512, 512)),
    ("1024x1024", (1024, 1024)),
    ("1024x256", (1024, 256)),
    ("512x128", (512, 128)),
    ("256x1024", (256, 1024)),
]


def _symmetric(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    a = rng.standard_normal(shape)
    return (a + a.T) / 2.0


def _solvable(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.standard_normal((n, n))
    return a + 5.0 * np.eye(n)


def run():
    rng = np.random.default_rng(123)

    matmul_rows: list[list[str]] = []
    for label, a_shape, b_shape in MATMUL_CASES:
        a = rng.standard_normal(a_shape)
        b = rng.standard_normal(b_shape)

        np_p50, _ = _timed(lambda x, y: x @ y, a, b)
        pg_p50, _ = _timed(linalg.matmul, a, b)
        matmul_rows.append(compare_row("matmul", label, np_p50, pg_p50))
    print_compare_section("Matmul (with copy-out)", matmul_rows, ref_label="numpy")

    solve_rows: list[list[str]] = []
    for label, a_shape, b_shape in SOLVE_CASES:
        n = a_shape[0]
        a = _solvable(rng, n)
        b = rng.standard_normal(b_shape)

        np_p50, _ = _timed(lambda x, y: np.linalg.solve(x, y), a, b)
        for method in _lapack_eigen_methods():
            pg_p50, _ = _timed(lambda x, y, m=method: linalg.solve(x, y, method=m), a, b)
            solve_rows.append(compare_row(f"solve[{method}]", label, np_p50, pg_p50))
    print_compare_section("Solve (with copy-out)", solve_rows, ref_label="numpy")

    norm_rows: list[list[str]] = []
    for label, shape in NORM_CASES:
        a = rng.standard_normal(shape)

        np_p50, _ = _timed(lambda x: np.linalg.norm(x), a)
        pg_p50, _ = _timed(linalg.norm, a)
        norm_rows.append(compare_row("norm", label, np_p50, pg_p50))
    print_compare_section("Norm Frobenius (2D default)", norm_rows, ref_label="numpy")

    svd_rows: list[list[str]] = []
    for label, shape in SVD_CASES:
        a = rng.standard_normal(shape)
        np_p50, _ = _timed(lambda x: np.linalg.svd(x, full_matrices=False), a)

        for method in _svd_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.svd(x, full_matrices=False, method=m),
                a,
            )
            svd_rows.append(compare_row(f"svd[{method}]", label, np_p50, pg_p50))
    print_compare_section("SVD with copy-out (U, s, Vt returned)", svd_rows, ref_label="numpy")

    svd_compute_rows: list[list[str]] = []
    for label, shape in SVD_CASES:
        a = rng.standard_normal(shape)
        np_p50, _ = _timed(lambda x: np.linalg.svd(x, full_matrices=False), a)

        for method in _svd_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.svd_compute(x, full_matrices=False, method=m),
                a,
            )
            svd_compute_rows.append(compare_row(f"svd_compute[{method}]", label, np_p50, pg_p50))
    print_compare_section(
        "SVD compute-only (minimal return; excludes U/V copy-out)",
        svd_compute_rows,
        ref_label="numpy",
    )

    qr_rows: list[list[str]] = []
    for label, shape in QR_CASES:
        a = rng.standard_normal(shape)
        np_p50, _ = _timed(lambda x: np.linalg.qr(x, mode="reduced"), a)
        pg_p50, _ = _timed(lambda x: linalg.qr(x, mode="reduced"), a)
        qr_rows.append(compare_row("qr", label, np_p50, pg_p50))
    print_compare_section("QR with copy-out (Q, R returned, mode='reduced')", qr_rows, ref_label="numpy")

    eigh_rows: list[list[str]] = []
    for label, shape in EIGH_CASES:
        a = _symmetric(rng, shape)
        np_p50, _ = _timed(lambda x: np.linalg.eigh(x), a)

        for method in _eigh_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.eigh(x, method=m),
                a,
            )
            eigh_rows.append(compare_row(f"eigh[{method}]", label, np_p50, pg_p50))
    print_compare_section(
        "Eigh with copy-out (eigenvalues and eigenvectors returned)",
        eigh_rows,
        ref_label="numpy",
    )

    eighvals_rows: list[list[str]] = []
    for label, shape in EIGH_CASES:
        a = _symmetric(rng, shape)
        np_p50, _ = _timed(lambda x: np.linalg.eigvalsh(x), a)

        for method in _eigh_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.eighvals(x, method=m),
                a,
            )
            eighvals_rows.append(compare_row(f"eighvals[{method}]", label, np_p50, pg_p50))
    print_compare_section("Eigh eigenvalues only (no eigenvector copy-out)", eighvals_rows, ref_label="numpy")

    eigh_compute_rows: list[list[str]] = []
    for label, shape in EIGH_CASES:
        a = _symmetric(rng, shape)
        np_p50, _ = _timed(lambda x: np.linalg.eigvalsh(x), a)

        for method in _eigh_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.eigh_compute(x, method=m),
                a,
            )
            eigh_compute_rows.append(compare_row(f"eigh_compute[{method}]", label, np_p50, pg_p50))
    print_compare_section(
        "Eigh eigenvalues compute-only (minimal return; excludes eigenvector copy-out)",
        eigh_compute_rows,
        ref_label="numpy",
    )


if __name__ == "__main__":
    run()
