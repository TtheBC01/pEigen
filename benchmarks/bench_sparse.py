"""Optional sparse benchmarks comparing pEigen and SciPy."""

from __future__ import annotations

import time

import numpy as np

try:
    import scipy.sparse as sp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("SciPy is required for sparse benchmarks") from exc

from peigen import sparse


def _timed(fn, *args, warmup: int = 3, runs: int = 10):
    for _ in range(warmup):
        fn(*args)

    durations = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        durations.append((time.perf_counter() - t0) * 1000.0)

    durations.sort()
    return durations[len(durations) // 2]


def run():
    rng = np.random.default_rng(321)
    print(f"{'op':<10} {'size':<10} {'scipy_p50(ms)':>14} {'peigen_p50(ms)':>15} {'speedup':>8}")
    print("-" * 62)

    for n in (1000, 2000, 2048):
        a = sp.random(n, n, density=0.001, format="csc", random_state=42) + 10.0 * sp.eye(n, format="csc")
        b = rng.standard_normal((n, 8))

        scipy_ms = _timed(lambda x, y: x @ y, a, b)
        peigen_ms = _timed(sparse.spmm, a, b)
        speedup = scipy_ms / peigen_ms if peigen_ms > 0 else float("inf")
        print(f"spmm       {n}x{n:<6} {scipy_ms:>10.3f} {peigen_ms:>10.3f} {speedup:>8.2f}x")


if __name__ == "__main__":
    run()
