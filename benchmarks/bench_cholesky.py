"""Sparse SPD direct-solver benchmark: pEigen SimplicialLDLT/LLT vs SciPy SuperLU vs CHOLMOD.

For each test matrix and solver, measures:

* factor    - symbolic analysis + numeric factorization (what a one-off solve pays)
* refactor  - numeric factorization only, reusing the symbolic analysis for new values with the
              same sparsity pattern (SuperLU cannot do this; it pays a full factor again)
* solve     - triangular solves with an existing factorization, 1 and 32 right-hand sides
* fill      - stored nonzeros in the factor(s): L for Cholesky, L+U for LU

CHOLMOD (scikit-sparse) is optional and skipped if not installed. It uses OpenMP and a
multithreaded BLAS for its supernodal kernels, while Eigen's simplicial Cholesky is
single-threaded; run with ``OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1``
for a single-core comparison.

Usage: python benchmarks/bench_cholesky.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    import scipy
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover
    raise SystemExit("SciPy is required for this benchmark") from exc

try:
    import sksparse
    from sksparse import cholmod
except ImportError:  # pragma: no cover - optional
    cholmod = None

from tabulate import tabulate

from peigen import sparse

from _table import TABLEFMT

MANY_RHS = 32
TIME_BUDGET_S = 1.5  # per (solver, matrix, metric); at least MIN_RUNS runs unless a single run is slow
MIN_RUNS, MAX_RUNS = 3, 15
SLOW_RUN_S = 8.0  # a single run longer than this is not repeated


# --------------------------------------------------------------------------------------------
# Test matrices (all symmetric positive definite)
# --------------------------------------------------------------------------------------------


def _tridiag(n: int) -> sp.csc_matrix:
    return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")


def laplacian_2d(m: int) -> sp.csc_matrix:
    """5-point Dirichlet Laplacian on an m x m grid."""
    t, i = _tridiag(m), sp.eye(m, format="csc")
    return (sp.kron(i, t) + sp.kron(t, i)).tocsc()


def laplacian_3d(m: int) -> sp.csc_matrix:
    """7-point Dirichlet Laplacian on an m x m x m grid (much more fill than 2D)."""
    t, i = _tridiag(m), sp.eye(m, format="csc")
    return (sp.kron(sp.kron(i, i), t) + sp.kron(sp.kron(i, t), i) + sp.kron(sp.kron(t, i), i)).tocsc()


def knn_graph_laplacian(n: int, k: int = 8, seed: int = 0) -> sp.csc_matrix:
    """Unstructured: graph Laplacian of a symmetric k-NN graph on random 2D points, plus I.

    Resembles an unstructured mesh / point-cloud / GP-precision matrix.
    """
    pts = np.random.default_rng(seed).random((n, 2))
    _, nbrs = cKDTree(pts).query(pts, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = nbrs[:, 1:].ravel()
    w = sp.csc_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    w = ((w + w.T) > 0).astype(np.float64)
    lap = sp.diags(np.asarray(w.sum(axis=1)).ravel()) - w
    return (lap + sp.eye(n)).tocsc()


def problems(quick: bool):
    if quick:
        return [
            ("2D Laplacian", "128^2", lambda: laplacian_2d(128)),
            ("3D Laplacian", "20^3", lambda: laplacian_3d(20)),
            ("kNN graph", "20k", lambda: knn_graph_laplacian(20_000)),
        ]
    return [
        ("2D Laplacian", "64^2", lambda: laplacian_2d(64)),
        ("2D Laplacian", "128^2", lambda: laplacian_2d(128)),
        ("2D Laplacian", "256^2", lambda: laplacian_2d(256)),
        ("2D Laplacian", "512^2", lambda: laplacian_2d(512)),
        ("3D Laplacian", "16^3", lambda: laplacian_3d(16)),
        ("3D Laplacian", "24^3", lambda: laplacian_3d(24)),
        ("3D Laplacian", "32^3", lambda: laplacian_3d(32)),
        ("3D Laplacian", "40^3", lambda: laplacian_3d(40)),
        ("kNN graph", "10k", lambda: knn_graph_laplacian(10_000)),
        ("kNN graph", "50k", lambda: knn_graph_laplacian(50_000)),
        ("kNN graph", "200k", lambda: knn_graph_laplacian(200_000)),
    ]


# --------------------------------------------------------------------------------------------
# Solvers: a uniform adapter per contender
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Solver:
    name: str
    factor: Callable  # A -> factor object
    solve: Callable  # (factor, b) -> x
    refactor: Callable | None  # (factor, A_new) -> None, same pattern; None = unsupported
    fill: Callable  # factor -> int


def _splu_fill(lu) -> int:
    return int(lu.L.nnz + lu.U.nnz)


def solvers() -> list[Solver]:
    out = [
        Solver(
            "scipy splu",
            lambda a: spla.splu(a),
            lambda f, b: f.solve(b),
            None,
            _splu_fill,
        ),
        Solver(
            # SuperLU tuned for SPD: symmetric ordering on A+A^T, no off-diagonal pivoting.
            "scipy splu(sym)",
            lambda a: spla.splu(
                a, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True}
            ),
            lambda f, b: f.solve(b),
            None,
            _splu_fill,
        ),
        Solver(
            "peigen lu",
            lambda a: sparse.factorize(a),
            lambda f, b: f.solve(b),
            None,
            lambda f: -1,
        ),
        Solver(
            "peigen ldlt",
            lambda a: sparse.cholesky(a, mode="ldlt"),
            lambda f, b: f.solve(b),
            lambda f, a: f.refactorize(a),
            lambda f: f.factor_nnz,
        ),
        Solver(
            "peigen llt",
            lambda a: sparse.cholesky(a, mode="llt"),
            lambda f, b: f.solve(b),
            lambda f, a: f.refactorize(a),
            lambda f: f.factor_nnz,
        ),
    ]
    if cholmod is not None:
        for label, order, mode in (
            ("cholmod amd/simpl", "amd", "simplicial"),
            ("cholmod amd", "amd", "auto"),
            ("cholmod best", "default", "auto"),
        ):
            out.append(
                Solver(
                    label,
                    lambda a, order=order, mode=mode: cholmod.cho_factor(
                        a, lower=True, order=order, supernodal_mode=mode
                    ),
                    lambda f, b: f.solve(b),
                    lambda f, a: f.factorize(a),
                    lambda f: int(f.nnz),
                )
            )
    return out


# --------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------


def _p50_ms(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > SLOW_RUN_S:
        return first * 1e3
    times = [first]
    start = time.perf_counter()
    while len(times) < MAX_RUNS and (len(times) < MIN_RUNS or time.perf_counter() - start < TIME_BUDGET_S):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    # drop the first (cold) run when we have enough samples
    samples = times[1:] if len(times) > MIN_RUNS else times
    return float(np.median(samples)) * 1e3


def _rel_residual(a, x, b) -> float:
    return float(np.linalg.norm(a @ x - b) / np.linalg.norm(b))


@dataclass
class Result:
    factor_ms: float = float("nan")
    refactor_ms: float = float("nan")
    solve1_ms: float = float("nan")
    solve_many_ms: float = float("nan")
    fill: int = -1
    resid: float = float("nan")
    error: str | None = None


def bench_one(solver: Solver, a, a_refactor, b1, bmany) -> Result:
    r = Result()
    try:
        r.factor_ms = _p50_ms(lambda: solver.factor(a))
        fac = solver.factor(a)
        r.fill = solver.fill(fac)
        x = solver.solve(fac, b1)
        r.resid = _rel_residual(a, x, b1)
        r.solve1_ms = _p50_ms(lambda: solver.solve(fac, b1))
        r.solve_many_ms = _p50_ms(lambda: solver.solve(fac, bmany))
        if solver.refactor is not None:
            r.refactor_ms = _p50_ms(lambda: solver.refactor(fac, a_refactor))
            x2 = solver.solve(fac, b1)
            resid2 = _rel_residual(a_refactor, x2, b1)
            if not resid2 < 1e-8:
                r.error = f"refactor residual {resid2:.1e}"
        if not r.resid < 1e-8:
            r.error = f"residual {r.resid:.1e}"
    except Exception as exc:  # noqa: BLE001 - report and keep going
        r.error = f"{type(exc).__name__}: {exc}"[:60]
    return r


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def _fmt_ms(v: float) -> str:
    if v != v:
        return "-"
    return f"{v:,.1f}" if v >= 10 else f"{v:.2f}"


def _fmt_fill(v: int) -> str:
    return "-" if v < 0 else f"{v / 1e6:.2f}M"


def _print_metric(title: str, rows_meta, names, results, attr: str, fmt, notes: str | None = None):
    headers = ["problem", "n", "nnz(A)"] + names
    rows = []
    for (family, size, n, nnz), res in zip(rows_meta, results):
        cells = []
        for name in names:
            r = res[name]
            cells.append("err" if r.error and attr == "factor_ms" else fmt(getattr(r, attr)))
        rows.append([f"{family} {size}", f"{n:,}", f"{nnz / 1e6:.2f}M"] + cells)
    print()
    print(title)
    if notes:
        print(notes)
    print(tabulate(rows, headers=headers, tablefmt=TABLEFMT, disable_numparse=True))


def _ratio(num: float, den: float) -> str:
    if num != num or den != den or den <= 0:
        return "-"
    return f"{num / den:.2f}x"


def _print_summary(rows_meta, results, names):
    """pEigen LDLT relative to the alternatives (>1x means pEigen is faster)."""
    best_scipy = ("scipy splu", "scipy splu(sym)")
    cm = [n for n in ("cholmod amd/simpl", "cholmod best") if n in names]
    headers = ["problem", "factor vs best scipy", "factor vs peigen lu"]
    headers += [f"factor vs {n}" for n in cm]
    headers += [f"refactor vs {n}" for n in cm]
    headers += ["refactor vs own factor"]
    rows = []
    for (family, size, _, _), res in zip(rows_meta, results):
        ldlt = res["peigen ldlt"]
        scipy_best = min(res[n].factor_ms for n in best_scipy)
        row = [f"{family} {size}", _ratio(scipy_best, ldlt.factor_ms), _ratio(res["peigen lu"].factor_ms, ldlt.factor_ms)]
        row += [_ratio(res[n].factor_ms, ldlt.factor_ms) for n in cm]
        row += [_ratio(res[n].refactor_ms, ldlt.refactor_ms) for n in cm]
        row += [_ratio(ldlt.factor_ms, ldlt.refactor_ms)]
        rows.append(row)
    print()
    print("Summary: peigen ldlt speed relative to alternatives (>1x = peigen faster)")
    print("'refactor vs own factor' = how much a same-pattern refactorize saves over a fresh cholesky().")
    print(tabulate(rows, headers=headers, tablefmt=TABLEFMT, disable_numparse=True))


def _env_report():
    threads = {k: os.environ.get(k, "unset") for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")}
    print("Sparse SPD direct solvers: factor / refactor / solve")
    print(f"platform: {platform.platform()}  cpu: {platform.processor() or platform.machine()}  cores: {os.cpu_count()}")
    print(f"python {platform.python_version()}  numpy {np.__version__}  scipy {scipy.__version__}  "
          f"scikit-sparse {getattr(sksparse, '__version__', 'n/a') if cholmod else 'not installed'}")
    print(f"threads: {threads}")


def run(quick: bool = False):
    _env_report()
    slv = solvers()
    names = [s.name for s in slv]
    rows_meta, results = [], []
    rng = np.random.default_rng(0)
    for family, size, make in problems(quick):
        a = make()
        n = a.shape[0]
        # Same pattern, new values: scale and perturb the diagonal (stays SPD).
        a_re = a.copy()
        a_re.data = a_re.data * 1.5
        a_re.setdiag(a_re.diagonal() + rng.uniform(0.1, 1.0, n))
        a_re.sort_indices()
        b1 = rng.standard_normal(n)
        bmany = rng.standard_normal((n, MANY_RHS))
        print(f"  {family} {size}: n={n:,} nnz={a.nnz:,} ...", flush=True)
        res = {}
        for s in slv:
            res[s.name] = bench_one(s, a, a_re, b1, bmany)
            if res[s.name].error:
                print(f"    {s.name}: {res[s.name].error}", flush=True)
        rows_meta.append((family, size, n, a.nnz))
        results.append(res)

    _print_metric("Factor (analyze + numeric), p50 ms", rows_meta, names, results, "factor_ms", _fmt_ms)
    refac = [n for n in names if not n.startswith("scipy") and n != "peigen lu"]
    _print_metric(
        "Refactor (numeric only, same pattern), p50 ms",
        rows_meta,
        refac,
        results,
        "refactor_ms",
        _fmt_ms,
        notes="SuperLU and Eigen SparseLU (as bound) have no pattern-reuse path; they pay a full factor.",
    )
    _print_metric("Solve, 1 RHS, p50 ms", rows_meta, names, results, "solve1_ms", _fmt_ms)
    _print_metric(f"Solve, {MANY_RHS} RHS, p50 ms", rows_meta, names, results, "solve_many_ms", _fmt_ms)
    _print_metric(
        "Factor fill (stored nonzeros; L+U for LU, L for Cholesky)",
        [m for m in rows_meta],
        [n for n in names if n != "peigen lu"],
        results,
        "fill",
        _fmt_fill,
    )
    _print_summary(rows_meta, results, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="three small problems only")
    run(quick=parser.parse_args().quick)
