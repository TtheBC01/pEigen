# pEigen

pEigen is a lightweight Python package that exposes Eigen-powered dense and sparse linear algebra routines using a NumPy/SciPy-native API.

## Highlights

- `pybind11` bindings over Eigen kernels
- Dense routines in `peigen.linalg`
- Sparse routines in `peigen.sparse` (optional SciPy dependency)
- Stateful decomposition helpers in `peigen.decomp`
- Wheel-first distribution for macOS (arm64 + x86_64) and Linux

## Installation

Dense-only install:

```bash
pip install peigen
```

Dense + sparse install:

```bash
pip install peigen[sparse]
```

## Quick start

```python
import numpy as np
from peigen import linalg, sparse, decomp

A = np.random.randn(100, 100)
B = np.random.randn(100, 40)

C = linalg.matmul(A, B)
X = linalg.solve(A + 10 * np.eye(100), B)
Q, R = linalg.qr(A)
U, s, Vt = linalg.svd(A)
w, V = linalg.eigh(A + 10 * np.eye(100))

svd = decomp.SVD(A)
x_ls = svd.solve(np.random.randn(100))
```

Sparse example:

```python
import scipy.sparse as sp

K = sp.random(200, 200, density=0.02, format="csc") + 5 * sp.eye(200, format="csc")
M = sp.diags(np.full(200, 0.1), format="csc")
rhs = np.random.randn(200, 8)
b = np.random.randn(200)

Y = sparse.spmm(K, rhs)                              # sparse @ dense
C = sparse.spspmm(K, M)                              # sparse @ sparse
x = sparse.solve(K, b, method="cg", preconditioner="jacobi")
x = sparse.solve(K, rhs, method="lu")                # direct LU
fac = sparse.factorize(K)
x2 = fac.solve(rhs)
```

## API overview

### `peigen.linalg`

- `matmul(a, b)`
- `solve(a, b, assume_a="gen", method="auto")`
- `qr(a, mode="reduced")`
- `svd(a, full_matrices=False, method="auto")`
- `eigh(a, lower=True, eigenvectors=True, method="auto")`
- `eighvals(a, lower=True, method="auto")`
- `norm(a, ord=None, axis=None, keepdims=False)`

#### Linear solve (`solve`)

Solve `A x = b` for a square matrix `A`. Matches `numpy.linalg.solve` for general dense systems: `b` may be a 1D vector or a 2D array with multiple right-hand sides (columns of `x` correspond to columns of `b`).

```python
# Single right-hand side (1D b returns 1D x)
x = linalg.solve(A, b)

# Multiple right-hand sides (2D b returns 2D x)
X = linalg.solve(A, B)

# Explicit backend selection
x = linalg.solve(A, b, method="lapack")   # LAPACK dgesv
x = linalg.solve(A, b, method="eigen")    # Eigen PartialPivLU
```

| Parameter | Description |
|-----------|-------------|
| `assume_a="gen"` | General square matrix (only option today). Structured solvers (e.g. symmetric positive-definite) may be added later. |
| `method="auto"` | Use LAPACK when compiled in (default on release wheels); otherwise Eigen. |
| `method="lapack"` | Force LAPACK `dgesv` (LU factorization + pivoting, overwrites internal copies of `A` and `b` during the call). |
| `method="eigen"` | Force Eigen `PartialPivLU`. |

**Requirements and behavior:**

- `A` must be square; `b` must be 1D with length `n` or 2D with shape `(n, k)`.
- Raises `ValueError` if dimensions mismatch or if `A` is singular / numerically ill-conditioned (LAPACK `info > 0` or near-zero LU diagonal).
- Column-major staging for `A` is handled once inside the extension; callers do not need to pass Fortran-ordered arrays.

On macOS and Linux release wheels with LAPACK linked, `method="auto"` typically matches or exceeds NumPy performance for moderate and large systems.

#### QR decomposition

Compute `A = Q @ R` for a dense matrix `A` using Eigen's Householder QR.

```python
Q, R = linalg.qr(A)                    # reduced (economy) decomposition
Q, R = linalg.qr(A, mode="complete")   # full Q
```

| Parameter | Description |
|-----------|-------------|
| `mode="reduced"` | Returns `Q` with shape `(m, min(m, n))` and upper-triangular `R` with shape `(min(m, n), n)`. Default; matches `numpy.linalg.qr(..., mode="reduced")`. |
| `mode="complete"` | Returns full `Q` with shape `(m, m)`. |

For tall-skinny matrices (`m > n`), reduced mode uses Eigen's thin-`Q` path (`householderQ() * I_{m×k}`) rather than forming a full `m×m` factor explicitly.

#### Symmetric eigenproblems (`eigh`)

Compute eigenvalues and optionally eigenvectors of a real symmetric matrix. These routines mirror `numpy.linalg.eigh` / `numpy.linalg.eigvalsh`.

```python
# Eigenvalues and orthonormal eigenvectors (columns of V)
w, V = linalg.eigh(A)

# Eigenvalues only (faster when vectors are not needed)
w = linalg.eighvals(A)

# Explicit backend selection
w, V = linalg.eigh(A, method="lapack")   # LAPACK dsyevd
w, V = linalg.eigh(A, method="eigen")    # Eigen SelfAdjointEigenSolver
```

| Parameter | Description |
|-----------|-------------|
| `lower=True` | Use the lower triangle of `A` (LAPACK `uplo='L'`). Set `lower=False` to use the upper triangle instead. |
| `eigenvectors=True` | Return `(w, V)`. Set `eigenvectors=False` to return only `w`. |
| `method="auto"` | Use LAPACK when compiled in (default on release wheels); otherwise fall back to Eigen. |
| `method="lapack"` | Force LAPACK `dsyevd` (divide-and-conquer). |
| `method="eigen"` | Force Eigen's `SelfAdjointEigenSolver`. |

`eighvals(a, ...)` is a convenience wrapper equivalent to `eigh(a, eigenvectors=False, ...)`.

**Note:** Input matrices are treated as symmetric; only the selected triangle (`lower` or upper) is referenced, consistent with NumPy and LAPACK.

#### Matrix norm (`norm`)

The 2D default Frobenius fast path uses a zero-copy reduction over contiguous storage (BLAS-accelerated when BLAS is linked). Non-contiguous inputs are copied once in the extension. Other `ord` / `axis` combinations delegate to `numpy.linalg.norm`.

```python
n = linalg.norm(A)
```

### `peigen.sparse`

Requires SciPy (`pip install peigen[sparse]`). Sparse matrices are accepted in SciPy format; inputs are converted to CSC internally when needed.

- `spmm(a, b)`
- `spspmm(a, b)`
- `solve(a, b, method="auto", tol=1e-8, maxiter=None, preconditioner="none", ilu_fill_factor=10, ilu_drop_tol=1e-4)`
- `solve_stats(a, b, method="cg", ...)`
- `factorize(a, method="auto")` → `SparseFactorized`
- `to_dense(a)`
- `from_coo(data, row, col, shape)`

#### Sparse @ dense multiply (`spmm`)

Compute `Y = A @ B` where `A` is sparse and `B` is dense. Equivalent to SciPy/NumPy `A @ B` with a sparse left factor.

```python
Y = sparse.spmm(A, B)   # A: (m, n) sparse, B: (n, k) dense → (m, k) dense
```

**Requirements and behavior:**

- `A` must be 2D sparse (any SciPy sparse format; converted to CSC).
- `B` must be a 2D `float64` array with shape `(n, k)` where `n = A.shape[1]`.
- Raises `ValueError` on dimension mismatch.

#### Sparse @ sparse multiply (`spspmm`)

Compute `C = A @ B` with sparse factors. Returns a new SciPy CSC matrix.

```python
C = sparse.spspmm(A, B)   # A: (m, n), B: (n, k) → CSC (m, k)
```

**Requirements and behavior:**

- Both operands must be 2D sparse (converted to CSC).
- Raises `ValueError` if inner dimensions disagree.
- Result is pruned and returned as `scipy.sparse.csc_matrix`.

#### Sparse linear solve (`solve`)

Solve `A x = b` for a square sparse matrix `A`. Supports direct and iterative backends. `b` may be a 1D vector or a 2D array with multiple right-hand sides; a 1D `b` returns a 1D `x`.

```python
# Direct LU (default for method="auto")
x = sparse.solve(A, b, method="lu")
X = sparse.solve(A, B, method="lu")

# Conjugate gradient on SPD systems
x = sparse.solve(A, b, method="cg")
x = sparse.solve(A, b, method="cg", preconditioner="jacobi")
x = sparse.solve(A, b, method="cg", preconditioner="ilu", ilu_fill_factor=40)

# BiCGSTAB for general square systems
x = sparse.solve(A, b, method="bicgstab", tol=1e-8, maxiter=5000)
```

| Parameter | Description |
|-----------|-------------|
| `method="auto"` | Direct solve via Eigen `SparseLU` (same as `method="lu"`). |
| `method="lu"` | Factorize with `SparseLU` and solve each RHS column. |
| `method="cg"` | Conjugate gradient for self-adjoint (symmetric) `A`. |
| `method="bicgstab"` | BiCGSTAB for general square `A`. |
| `tol=1e-8` | Relative residual tolerance for iterative methods (`norm(Ax-b)/norm(b)`). |
| `maxiter=None` | Iteration cap for CG/BiCGSTAB. Default: `2 * n` where `n = A.shape[0]`. |
| `preconditioner="none"` | CG only. Identity preconditioner (unpreconditioned CG). |
| `preconditioner="jacobi"` | CG only. Diagonal (Jacobi) preconditioner. |
| `preconditioner="ilu"` | CG only. Incomplete LU via Eigen `IncompleteLUT`. |
| `ilu_fill_factor=10` | CG + ILU only. Fill-ratio upper bound passed to `IncompleteLUT`. |
| `ilu_drop_tol=1e-4` | CG + ILU only. Drop tolerance for `IncompleteLUT`. |

**Requirements and behavior:**

- `A` must be square sparse; `b` must have length `n` (1D) or shape `(n, k)` (2D).
- `preconditioner` is only valid with `method="cg"`; other methods raise `ValueError` if a nontrivial preconditioner is requested.
- Iterative solvers raise `RuntimeError` if they do not converge within `maxiter` at the requested tolerance.
- LU raises `RuntimeError` if factorization or solve fails.

**When to use which method:**

| Method | Matrix structure | Backend |
|--------|------------------|---------|
| `"lu"` / `"auto"` | General square | Eigen `SparseLU` |
| `"cg"` | Symmetric (SPD in practice) | Eigen `ConjugateGradient` |
| `"bicgstab"` | General square, nonsymmetric | Eigen `BiCGSTAB` |

CG preconditioners apply only to `"cg"`. For difficult SPD problems (e.g. Poisson-like operators), try `"jacobi"` or `"ilu"` with tuned `ilu_fill_factor` / `ilu_drop_tol`.

#### CG diagnostics (`solve_stats`)

Return iteration count and estimated relative error from a CG solve without changing the `solve` API. Useful for checking convergence or comparing preconditioners.

```python
stats = sparse.solve_stats(A, b, method="cg", preconditioner="ilu", ilu_fill_factor=40)
print(stats["iterations"], stats["error"])
```

| Key | Description |
|-----|-------------|
| `"iterations"` | CG iterations performed. |
| `"error"` | Estimated relative residual from Eigen. |

**Requirements and behavior:**

- Only supported for `method="cg"`.
- Requires a single RHS column (1D `b` or `(n, 1)` array).
- Accepts the same `tol`, `maxiter`, `preconditioner`, `ilu_fill_factor`, and `ilu_drop_tol` arguments as `solve`.
- Raises `RuntimeError` if CG does not converge (same as `solve`).

#### Sparse LU factorization (`factorize`)

Precompute a reusable LU factorization for repeated solves with different right-hand sides. Wraps Eigen `SparseLU` (analyze + factorize once, solve many times).

```python
fac = sparse.factorize(A)
x = fac.solve(b)       # 1D or 2D b; returns ndarray (2D if b is 2D)
X = fac.solve(B)
```

| Parameter | Description |
|-----------|-------------|
| `method="auto"` | Only option today; uses Eigen `SparseLU`. |

**Requirements and behavior:**

- `A` must be square sparse.
- `fac.solve(b)` requires `b` with shape `(n,)` or `(n, k)` matching `A.shape[0]`.
- Raises `RuntimeError` if factorization fails; `ValueError` on shape mismatch at solve time.
- The factorized object holds the pattern and numeric factors; reuse it when solving many systems with the same `A`.

A thin wrapper is also available as `peigen.decomp.SparseFactorized(A)`.

### `peigen.decomp`

- `SVD(a)`
- `QR(a)`
- `SparseFactorized(a)`

## Runtime build report

Check which performance backends were compiled into your installed wheel/extension:

```python
import peigen

peigen.show_build_config()
```

Example output:

```text
pEigen build configuration
--------------------------
version:               2.0.0
build_type:            Release
blas_enabled:          True
blas_backend:          accelerate
lapack_enabled:        True
lapack_backend:        accelerate
openmp_enabled:        False
vectorization_enabled: True
eigen_mpl2_only:       True
```

You can also inspect the raw dictionary:

```python
cfg = peigen.build_config()
print(cfg["blas_backend"])
```

From the command line:

```bash
python -c "import peigen; peigen.show_build_config()"
```

## Development

See [DEV.md](DEV.md) for local build, test, benchmark, and release instructions.
