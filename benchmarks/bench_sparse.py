"""Optional sparse benchmarks comparing pEigen and SciPy.

Matrices are 2D finite-difference operators on structured grids (Poisson Laplacian,
advection–diffusion, lumped mass) rather than unstructured random sparse patterns.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("SciPy is required for sparse benchmarks") from exc

from peigen import sparse


# (nx, ny) structured grids; n = nx * ny is the global matrix dimension.
BENCH_GRIDS = (
    (32, 32),  # n = 1024
    (45, 45),  # n = 2025
    (64, 32),  # n = 2048
    (64, 64),  # n = 4096
    (128, 64),  # n = 8192
    (128, 128),  # n = 16384
)
RHS_COLS = 8
CG_RHS_COLS = 1
CG_PRECONDITIONERS = ("none", "jacobi", "ilu")
SOLVE_RTOL = 1e-8

# ILU parameters calibrated on BENCH_GRIDS Laplacian (seed 321): both sides converge at
# rtol=1e-8 with similar CG iteration counts (typically 3–8 iterations).
ILU_DROP_TOL = 1e-4
SCIPY_ILU_FILL_FACTOR = 20
EIGEN_ILU_FILL_FACTOR = 40

# Advection–diffusion: -epsilon * Laplacian + v · grad (upwind), nonsymmetric.
ADV_DIFF_EPSILON = 1.0
ADV_DIFF_VX = 1.0
ADV_DIFF_VY = 0.5


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


def _report_line(op: str, size: str, scipy_ms: float, peigen_ms: float):
    speedup = scipy_ms / peigen_ms if peigen_ms > 0 else float("inf")
    print(f"{op:<30} {size:<12} {scipy_ms:>14.3f} {peigen_ms:>15.3f} {speedup:>8.2f}x")


def _report_cg_line(
    op: str,
    size: str,
    scipy_ms: float,
    peigen_ms: float,
    *,
    scipy_iters: int,
    peigen_iters: int,
    scipy_resid: float,
    peigen_resid: float,
    scipy_ok: bool,
    peigen_ok: bool,
):
    if not scipy_ok or not peigen_ok:
        speedup_str = "     n/a"
    elif peigen_ms != peigen_ms:
        speedup_str = "     n/a"
    elif peigen_ms > 0:
        speedup_str = f"{scipy_ms / peigen_ms:>8.2f}x"
    else:
        speedup_str = f"{'inf':>8}x"

    if peigen_ms != peigen_ms:
        peigen_str = "           fail"
    else:
        peigen_str = f"{peigen_ms:>15.3f}"

    scipy_resid_str = f"{scipy_resid:.1e}" if scipy_ok else "fail"
    peigen_resid_str = f"{peigen_resid:.1e}" if peigen_ok else "fail"
    print(
        f"{op:<30} {size:<12} {scipy_ms:>14.3f} {peigen_str} {speedup_str}"
        f"  iters {scipy_iters}/{peigen_iters}"
        f"  resid {scipy_resid_str}/{peigen_resid_str}"
    )


def _grid_label(nx: int, ny: int) -> str:
    n = nx * ny
    return f"{nx}x{ny} n={n}"


def _flat_index(i: int, j: int, nx: int) -> int:
    return i * nx + j


def laplacian_2d(nx: int, ny: int) -> sp.csc_matrix:
    """5-point Laplacian (-∇²) with homogeneous Dirichlet boundaries on an nx×ny grid.

    Each node uses a fixed diagonal of 4; missing neighbors at the boundary represent
    Dirichlet zero values (standard finite-difference Poisson benchmark).
    """
    n = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(ny):
        for j in range(nx):
            k = _flat_index(i, j, nx)
            rows.append(k)
            cols.append(k)
            data.append(4.0)
            if i > 0:
                rows.append(k)
                cols.append(_flat_index(i - 1, j, nx))
                data.append(-1.0)
            if i < ny - 1:
                rows.append(k)
                cols.append(_flat_index(i + 1, j, nx))
                data.append(-1.0)
            if j > 0:
                rows.append(k)
                cols.append(_flat_index(i, j - 1, nx))
                data.append(-1.0)
            if j < nx - 1:
                rows.append(k)
                cols.append(_flat_index(i, j + 1, nx))
                data.append(-1.0)

    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()


def advection_diffusion_2d(
    nx: int,
    ny: int,
    *,
    epsilon: float = ADV_DIFF_EPSILON,
    vx: float = ADV_DIFF_VX,
    vy: float = ADV_DIFF_VY,
) -> sp.csc_matrix:
    """Nonsymmetric advection–diffusion operator: -epsilon * Laplacian + v · ∇ (upwind)."""
    lap = laplacian_2d(nx, ny)
    n = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    hx = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)

    for i in range(ny):
        for j in range(nx):
            k = _flat_index(i, j, nx)
            diag = 0.0

            if vx > 0.0:
                coeff = vx / hx
                if j > 0:
                    rows.append(k)
                    cols.append(_flat_index(i, j - 1, nx))
                    data.append(-coeff)
                diag += coeff
            elif vx < 0.0:
                coeff = -vx / hx
                if j < nx - 1:
                    rows.append(k)
                    cols.append(_flat_index(i, j + 1, nx))
                    data.append(-coeff)
                diag += coeff

            if vy > 0.0:
                coeff = vy / hy
                if i > 0:
                    rows.append(k)
                    cols.append(_flat_index(i - 1, j, nx))
                    data.append(-coeff)
                diag += coeff
            elif vy < 0.0:
                coeff = -vy / hy
                if i < ny - 1:
                    rows.append(k)
                    cols.append(_flat_index(i + 1, j, nx))
                    data.append(-coeff)
                diag += coeff

            if diag != 0.0:
                rows.append(k)
                cols.append(k)
                data.append(diag)

    adv = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    return (epsilon * lap + adv).tocsc()


def lumped_mass_2d(nx: int, ny: int) -> sp.csc_matrix:
    """Lumped mass matrix (cell area on the diagonal) for an nx×ny grid."""
    n = nx * ny
    hx = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)
    cell_area = hx * hy
    return sp.diags(np.full(n, cell_area), format="csc")


def _maxiter(n: int, *, method: str) -> int:
    """Iteration cap aligned with pEigen defaults; CG on Laplacian needs more steps."""
    if method == "cg":
        return max(10 * n, 5000)
    return 2 * n


def _relative_residual(a, x: np.ndarray, b: np.ndarray) -> float:
    b_vec = np.asarray(b, dtype=np.float64).reshape(-1)
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(b_vec)
    if denom == 0.0:
        return float(np.linalg.norm(a @ x_vec - b_vec))
    return float(np.linalg.norm(a @ x_vec - b_vec) / denom)


def _scipy_cg_preconditioner(a, preconditioner: str):
    if preconditioner == "ilu":
        ilu = spla.spilu(
            a.tocsc(),
            drop_tol=ILU_DROP_TOL,
            fill_factor=SCIPY_ILU_FILL_FACTOR,
        )
        return spla.LinearOperator(a.shape, matvec=ilu.solve)
    if preconditioner == "jacobi":
        return sp.diags(1.0 / a.diagonal())
    return None


def _scipy_cg_solve(a, b: np.ndarray, *, rtol: float, maxiter: int, preconditioner: str) -> None:
    """SciPy CG with optional preconditioner (setup included for ILU)."""
    b_vec = b if b.ndim == 1 else b[:, 0]
    kwargs: dict = {"rtol": rtol, "maxiter": maxiter}
    precond = _scipy_cg_preconditioner(a, preconditioner)
    if precond is not None:
        kwargs["M"] = precond
    spla.cg(a, b_vec, **kwargs)


def _scipy_cg_profile(
    a,
    b: np.ndarray,
    *,
    rtol: float,
    maxiter: int,
    preconditioner: str,
) -> dict:
    b_vec = b if b.ndim == 1 else b[:, 0]
    iters = [0]

    def _callback(_xk):
        iters[0] += 1

    kwargs: dict = {"rtol": rtol, "maxiter": maxiter, "callback": _callback}
    precond = _scipy_cg_preconditioner(a, preconditioner)
    if precond is not None:
        kwargs["M"] = precond
    x, info = spla.cg(a, b_vec, **kwargs)
    residual = _relative_residual(a, x, b_vec)
    converged = info == 0 and residual <= rtol
    return {
        "iters": iters[0] if info == 0 else maxiter,
        "residual": residual,
        "converged": converged,
    }


def _timed_cg_solve(fn, a, b) -> float:
    try:
        return _timed(fn, a, b)
    except RuntimeError:
        return float("nan")


def _peigen_cg_profile(
    a,
    b: np.ndarray,
    *,
    rtol: float,
    maxiter: int,
    preconditioner: str,
) -> dict:
    solve_kwargs: dict = {
        "method": "cg",
        "tol": rtol,
        "maxiter": maxiter,
        "preconditioner": preconditioner,
    }
    if preconditioner == "ilu":
        solve_kwargs["ilu_fill_factor"] = EIGEN_ILU_FILL_FACTOR
        solve_kwargs["ilu_drop_tol"] = ILU_DROP_TOL
    try:
        x = sparse.solve(a, b, **solve_kwargs)
        stats = sparse.solve_stats(a, b, **solve_kwargs)
        residual = _relative_residual(a, x, b)
        converged = residual <= rtol
        return {
            "iters": int(stats["iterations"]),
            "residual": residual,
            "converged": converged,
        }
    except RuntimeError:
        return {"iters": maxiter, "residual": float("nan"), "converged": False}


def _peigen_cg_solve(
    a,
    b: np.ndarray,
    *,
    rtol: float,
    maxiter: int,
    preconditioner: str,
):
    solve_kwargs: dict = {
        "method": "cg",
        "tol": rtol,
        "maxiter": maxiter,
        "preconditioner": preconditioner,
    }
    if preconditioner == "ilu":
        solve_kwargs["ilu_fill_factor"] = EIGEN_ILU_FILL_FACTOR
        solve_kwargs["ilu_drop_tol"] = ILU_DROP_TOL
    return sparse.solve(a, b, **solve_kwargs)


def _scipy_iterative_solve(
    solver: Callable,
    a,
    b: np.ndarray,
    *,
    rtol: float,
    maxiter: int,
) -> None:
    """Run SciPy iterative solver on each RHS column (mirrors pEigen column loop)."""
    if b.ndim == 1:
        solver(a, b, rtol=rtol, maxiter=maxiter)
        return
    for col in range(b.shape[1]):
        solver(a, b[:, col], rtol=rtol, maxiter=maxiter)


def _peigen_iterative_solve(method: str, a, b: np.ndarray, *, rtol: float, maxiter: int):
    return sparse.solve(a, b, method=method, tol=rtol, maxiter=maxiter)


def run():
    rng = np.random.default_rng(321)

    print(f"{'op':<30} {'size':<12} {'scipy_p50(ms)':>14} {'peigen_p50(ms)':>15} {'speedup':>8}")
    print("-" * 80)
    print(
        "Grid operators: 5-point Laplacian (SPD/CG), upwind advection–diffusion "
        "(LU/BiCGSTAB), stiffness @ lumped mass (spspmm)"
    )
    print("-" * 80)

    print("\nSpmm (sparse @ dense; Laplacian operator)")
    for nx, ny in BENCH_GRIDS:
        a = laplacian_2d(nx, ny)
        n = nx * ny
        b = rng.standard_normal((n, RHS_COLS))
        label = _grid_label(nx, ny)

        scipy_ms = _timed(lambda x, y: x @ y, a, b)
        peigen_ms = _timed(sparse.spmm, a, b)
        _report_line("spmm", label, scipy_ms, peigen_ms)

    print("\nSpspmm (stiffness @ lumped mass, copy-out)")
    for nx, ny in BENCH_GRIDS:
        stiffness = laplacian_2d(nx, ny)
        mass = lumped_mass_2d(nx, ny)
        label = _grid_label(nx, ny)

        scipy_ms = _timed(lambda k, m: k @ m, stiffness, mass)
        peigen_ms = _timed(sparse.spspmm, stiffness, mass)
        _report_line("spspmm", label, scipy_ms, peigen_ms)

    print("\nSparse solve [CG] (setup + solve; 2D Laplacian, SPD; single RHS)")
    print(
        "CG lines: speedup only when both sides converge; same (A, b) per grid across "
        "preconditioners."
    )
    print(
        f"ILU tuning: SciPy spilu(fill={SCIPY_ILU_FILL_FACTOR}, drop_tol={ILU_DROP_TOL}); "
        f"pEigen IncompleteLUT(fill={EIGEN_ILU_FILL_FACTOR}, drop_tol={ILU_DROP_TOL})."
    )
    for nx, ny in BENCH_GRIDS:
        a = laplacian_2d(nx, ny)
        n = nx * ny
        b = rng.standard_normal(n)
        label = _grid_label(nx, ny)
        maxiter = _maxiter(n, method="cg")

        for preconditioner in CG_PRECONDITIONERS:
            op = f"sparse_solve[cg,{preconditioner}]"

            scipy_ms = _timed(
                lambda x, y, pre=preconditioner: _scipy_cg_solve(
                    x, y, rtol=SOLVE_RTOL, maxiter=maxiter, preconditioner=pre
                ),
                a,
                b,
            )
            peigen_ms = _timed_cg_solve(
                lambda x, y, pre=preconditioner: _peigen_cg_solve(
                    x, y, rtol=SOLVE_RTOL, maxiter=maxiter, preconditioner=pre
                ),
                a,
                b,
            )
            scipy_prof = _scipy_cg_profile(
                a, b, rtol=SOLVE_RTOL, maxiter=maxiter, preconditioner=preconditioner
            )
            peigen_prof = _peigen_cg_profile(
                a, b, rtol=SOLVE_RTOL, maxiter=maxiter, preconditioner=preconditioner
            )
            _report_cg_line(
                op,
                label,
                scipy_ms,
                peigen_ms,
                scipy_iters=scipy_prof["iters"],
                peigen_iters=peigen_prof["iters"],
                scipy_resid=scipy_prof["residual"],
                peigen_resid=peigen_prof["residual"],
                scipy_ok=scipy_prof["converged"],
                peigen_ok=peigen_prof["converged"],
            )

    print("\nSparse solve [BiCGSTAB] (setup + solve; advection–diffusion)")
    for nx, ny in BENCH_GRIDS:
        a = advection_diffusion_2d(nx, ny)
        n = nx * ny
        b = rng.standard_normal((n, RHS_COLS))
        label = _grid_label(nx, ny)
        maxiter = _maxiter(n, method="bicgstab")

        scipy_ms = _timed(
            lambda x, y: _scipy_iterative_solve(
                spla.bicgstab, x, y, rtol=SOLVE_RTOL, maxiter=maxiter
            ),
            a,
            b,
        )
        peigen_ms = _timed(
            lambda x, y: _peigen_iterative_solve(
                "bicgstab", x, y, rtol=SOLVE_RTOL, maxiter=maxiter
            ),
            a,
            b,
        )
        _report_line("sparse_solve[bicgstab]", label, scipy_ms, peigen_ms)

    print("\nSparse factorize [LU] (pattern + numeric factorization; advection–diffusion)")
    for nx, ny in BENCH_GRIDS:
        a = advection_diffusion_2d(nx, ny)
        label = _grid_label(nx, ny)

        scipy_ms = _timed(lambda x: spla.splu(x.tocsc()), a)
        peigen_ms = _timed(sparse.factorize, a)
        _report_line("sparse_factorize[lu]", label, scipy_ms, peigen_ms)

    print("\nSparse solve [LU] (factorize + solve; advection–diffusion)")
    for nx, ny in BENCH_GRIDS:
        a = advection_diffusion_2d(nx, ny)
        n = nx * ny
        b = rng.standard_normal((n, RHS_COLS))
        label = _grid_label(nx, ny)

        scipy_ms = _timed(spla.spsolve, a, b)
        peigen_ms = _timed(lambda x, y: sparse.solve(x, y, method="lu"), a, b)
        _report_line("sparse_solve[lu]", label, scipy_ms, peigen_ms)


if __name__ == "__main__":
    run()
