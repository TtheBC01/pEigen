"""Optional dense benchmarks comparing pEigen and NumPy."""

from __future__ import annotations

import time

import numpy as np

from peigen import build_config, linalg


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


def _report_line(op: str, size: str, numpy_ms: float, peigen_ms: float):
    speedup = numpy_ms / peigen_ms if peigen_ms > 0 else float("inf")
    print(f"{op:<18} {size:<14} {numpy_ms:>14.3f} {peigen_ms:>15.3f} {speedup:>8.2f}x")


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

    print(f"{'op':<18} {'size':<14} {'numpy_p50(ms)':>14} {'peigen_p50(ms)':>15} {'speedup':>8}")
    print("-" * 76)
    print("Matmul (with copy-out)")
    for label, a_shape, b_shape in MATMUL_CASES:
        a = rng.standard_normal(a_shape)
        b = rng.standard_normal(b_shape)

        np_p50, _ = _timed(lambda x, y: x @ y, a, b)
        pg_p50, _ = _timed(linalg.matmul, a, b)
        _report_line("matmul", label, np_p50, pg_p50)

    print("\nSolve (with copy-out)")
    for label, a_shape, b_shape in SOLVE_CASES:
        n = a_shape[0]
        a = _solvable(rng, n)
        b = rng.standard_normal(b_shape)

        np_p50, _ = _timed(lambda x, y: np.linalg.solve(x, y), a, b)
        for method in _lapack_eigen_methods():
            pg_p50, _ = _timed(lambda x, y, m=method: linalg.solve(x, y, method=m), a, b)
            _report_line(f"solve[{method}]", label, np_p50, pg_p50)

    print("\nNorm Frobenius (2D default)")
    for label, shape in NORM_CASES:
        a = rng.standard_normal(shape)

        np_p50, _ = _timed(lambda x: np.linalg.norm(x), a)
        pg_p50, _ = _timed(linalg.norm, a)
        _report_line("norm", label, np_p50, pg_p50)

    print("\nSVD with copy-out (U, s, Vt returned)")
    for label, shape in SVD_CASES:
        a = rng.standard_normal(shape)
        np_p50, _ = _timed(lambda x: np.linalg.svd(x, full_matrices=False), a)

        for method in _svd_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.svd(x, full_matrices=False, method=m),
                a,
            )
            _report_line(f"svd[{method}]", label, np_p50, pg_p50)

    print("\nSVD compute-only (minimal return; excludes U/V copy-out)")
    for label, shape in SVD_CASES:
        a = rng.standard_normal(shape)
        np_p50, _ = _timed(lambda x: np.linalg.svd(x, full_matrices=False), a)

        for method in _svd_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.svd_compute(x, full_matrices=False, method=m),
                a,
            )
            _report_line(f"svd_compute[{method}]", label, np_p50, pg_p50)

    print("\nQR with copy-out (Q, R returned, mode='reduced')")
    for label, shape in QR_CASES:
        a = rng.standard_normal(shape)
        np_p50, _ = _timed(lambda x: np.linalg.qr(x, mode="reduced"), a)
        pg_p50, _ = _timed(lambda x: linalg.qr(x, mode="reduced"), a)
        _report_line("qr", label, np_p50, pg_p50)

    print("\nEigh with copy-out (eigenvalues and eigenvectors returned)")
    for label, shape in EIGH_CASES:
        a = _symmetric(rng, shape)
        np_p50, _ = _timed(lambda x: np.linalg.eigh(x), a)

        for method in _eigh_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.eigh(x, method=m),
                a,
            )
            _report_line(f"eigh[{method}]", label, np_p50, pg_p50)

    print("\nEigh eigenvalues only (no eigenvector copy-out)")
    for label, shape in EIGH_CASES:
        a = _symmetric(rng, shape)
        np_p50, _ = _timed(lambda x: np.linalg.eigvalsh(x), a)

        for method in _eigh_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.eighvals(x, method=m),
                a,
            )
            _report_line(f"eighvals[{method}]", label, np_p50, pg_p50)

    print("\nEigh eigenvalues compute-only (minimal return; excludes eigenvector copy-out)")
    for label, shape in EIGH_CASES:
        a = _symmetric(rng, shape)
        np_p50, _ = _timed(lambda x: np.linalg.eigvalsh(x), a)

        for method in _eigh_methods():
            pg_p50, _ = _timed(
                lambda x, m=method: linalg.eigh_compute(x, method=m),
                a,
            )
            _report_line(f"eigh_compute[{method}]", label, np_p50, pg_p50)


if __name__ == "__main__":
    run()
