# pEigen

pEigen is a lightweight Python package that exposes Eigen-powered dense and sparse linear algebra routines using a NumPy/SciPy-native API.

## Highlights

- `pybind11` bindings over Eigen kernels
- Dense routines in `peigen.linalg`
- Sparse routines in `peigen.sparse` (optional SciPy dependency)
- Stateful decomposition helpers in `peigen.decomp`
- Sparse Cholesky (`peigen.sparse.cholesky`) with no SuiteSparse/CHOLMOD dependency
- Prebuilt wheels for CPython 3.10–3.15 on Linux (x86_64, aarch64; OpenBLAS) and macOS (arm64, x86_64; Accelerate)

## Installation

The package is published on PyPI as `peigentux` and imported as `peigen`.

Dense-only install:

```bash
pip install peigentux
```

Dense + sparse install:

```bash
pip install "peigentux[sparse]"
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

All routines operate on real `float64` data. Real inputs of other dtypes (ints, `float32`) are converted to `float64`; complex inputs raise `TypeError` rather than silently discarding the imaginary part.

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
- Raises `ValueError` if dimensions mismatch or if `A` is singular / numerically ill-conditioned (LAPACK `info > 0`, or for `method="eigen"` a smallest LU pivot below `n * eps` times the largest, so the check is independent of the matrix's overall scale).
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

Requires SciPy (`pip install "peigentux[sparse]"`). Sparse matrices are accepted in any SciPy format (`*_matrix` or `*_array`); inputs are converted to CSC internally when needed. `int32` or `int64` index arrays and real data of any dtype are accepted (converted to `int32` / `float64` copies when needed). Non-canonical inputs (unsorted or duplicate entries) are canonicalized on a copy; the caller's matrix is never modified. Matrices whose dimensions or nonzero count exceed the 32-bit range raise `ValueError`.

- `spmm(a, b)`
- `spspmm(a, b)`
- `solve(a, b, method="auto", tol=1e-8, maxiter=None, preconditioner="none", ilu_fill_factor=10, ilu_drop_tol=1e-4)`
- `solve_stats(a, b, method="cg", ...)`
- `factorize(a, method="auto")` → `SparseFactorized`
- `cholesky(a, mode="ldlt", ordering="amd", lower=True, shift=0.0)` → `SparseCholesky`
- `to_dense(a)`
- `from_coo(data, row, col, shape)`

#### Sparse @ dense multiply (`spmm`)

Compute `Y = A @ B` where `A` is sparse and `B` is dense. Equivalent to SciPy/NumPy `A @ B` with a sparse left factor.

```python
Y = sparse.spmm(A, B)   # A: (m, n) sparse, B: (n, k) dense → (m, k) dense
```

**Requirements and behavior:**

- `A` must be 2D sparse. CSR and CSC are used as they are; other formats are converted to CSR.
- `B` must be a 2D `float64` array with shape `(n, k)` where `n = A.shape[1]`. Any memory layout works; C-ordered `B` avoids a copy.
- Releases the GIL while multiplying.
- Raises `ValueError` on dimension mismatch.

#### Sparse @ sparse multiply (`spspmm`)

Compute `C = A @ B` with sparse factors. Returns a new SciPy CSC matrix.

```python
C = sparse.spspmm(A, B)   # A: (m, n), B: (n, k) → CSC (m, k)
```

**Requirements and behavior:**

- Both operands must be 2D sparse (converted to CSC).
- Raises `ValueError` if inner dimensions disagree.
- Exact zeros are dropped from the result, which is returned as `scipy.sparse.csc_matrix`.

#### Sparse linear solve (`solve`)

Solve `A x = b` for a square sparse matrix `A`. Supports direct and iterative backends. `b` may be a 1D vector or a 2D array with multiple right-hand sides; a 1D `b` returns a 1D `x`.

```python
# Direct LU (default for method="auto")
x = sparse.solve(A, b, method="lu")
X = sparse.solve(A, B, method="lu")

# Conjugate gradient on SPD systems
x = sparse.solve(A, b, method="cg")
x = sparse.solve(A, b, method="cg", preconditioner="jacobi")
x = sparse.solve(A, b, method="cg", preconditioner="ichol")

# BiCGSTAB for general square systems
x = sparse.solve(A, b, method="bicgstab", tol=1e-8, maxiter=5000)
```

| Parameter | Description |
|-----------|-------------|
| `method="auto"` | Direct solve via Eigen `SparseLU` (same as `method="lu"`). |
| `method="lu"` | Factorize with `SparseLU` and solve each RHS column. |
| `method="cholesky"` | Symmetric `A` only (lower triangle read): one-shot `SimplicialLDLT`, equivalent to `cholesky(A).solve(b)`. Usually much faster than LU for SPD systems. |
| `method="cg"` | Conjugate gradient for self-adjoint (symmetric) `A`. |
| `method="bicgstab"` | BiCGSTAB for general square `A`. |
| `tol=1e-8` | Relative residual tolerance for iterative methods (`norm(Ax-b)/norm(b)`). |
| `maxiter=None` | Iteration cap for CG/BiCGSTAB. Default: `2 * n` where `n = A.shape[0]`. |
| `preconditioner="none"` | CG only. Identity preconditioner (unpreconditioned CG). |
| `preconditioner="jacobi"` | CG only. Diagonal (Jacobi) preconditioner. |
| `preconditioner="ichol"` | CG only. Zero-fill incomplete Cholesky via Eigen `IncompleteCholesky`, in the matrix's own ordering (symmetric, so it suits CG). |
| `preconditioner="ilu"` | CG only. **Deprecated** (`DeprecationWarning`; removal planned for 3.0). Incomplete LU via Eigen `IncompleteLUT`, which is not symmetric, so CG can fail to converge with it. Use `"ichol"`. |
| `ilu_fill_factor=10` | CG + ILU only. Fill-ratio upper bound passed to `IncompleteLUT`. |
| `ilu_drop_tol=1e-4` | CG + ILU only. Drop tolerance for `IncompleteLUT`. |

**Requirements and behavior:**

- `A` must be square sparse; `b` must have length `n` (1D) or shape `(n, k)` (2D).
- `preconditioner` is only valid with `method="cg"`; other methods raise `ValueError` if a nontrivial preconditioner is requested.
- Iterative solvers raise `RuntimeError` if they do not converge within `maxiter` at the requested tolerance.
- Convergence is judged by the true residual. Eigen's solvers stop on a recursively updated residual, which can drift from the true `norm(b - Ax)/norm(b)` (badly so for BiCGSTAB on convection-dominated problems). After each solve pEigen recomputes the true residual; if it is more than 10x above `tol`, the solver restarts from its current iterate (up to 5 times, within the same `maxiter` budget) and raises `RuntimeError` if the residual still misses. A `tol` below what double precision can reach for the problem therefore raises instead of returning a less accurate answer.
- LU raises `RuntimeError` if factorization or solve fails.

**When to use which method:**

| Method | Matrix structure | Backend |
|--------|------------------|---------|
| `"lu"` / `"auto"` | General square | Eigen `SparseLU` |
| `"cholesky"` | Symmetric positive definite | Eigen `SimplicialLDLT` |
| `"cg"` | Symmetric (SPD in practice) | Eigen `ConjugateGradient` |
| `"bicgstab"` | General square, nonsymmetric | Eigen `BiCGSTAB` |

CG preconditioners apply only to `"cg"`. For Poisson-like operators `"ichol"` typically halves the iteration count (on a 100x100 5-point Laplacian: 186 iterations unpreconditioned, 77 with `"ichol"`); `"jacobi"` helps when the diagonal varies a lot. For a repeated or hard SPD solve, a direct `cholesky(A)` is often faster still.

#### CG diagnostics (`solve_stats`)

Return iteration count and residuals from a CG solve without changing the `solve` API. Useful for checking convergence or comparing preconditioners.

```python
stats = sparse.solve_stats(A, b, method="cg", preconditioner="ichol")
print(stats["iterations"], stats["error"], stats["residual"])
```

| Key | Description |
|-----|-------------|
| `"iterations"` | CG iterations performed, including any restart. |
| `"error"` | Eigen's own (recursively updated) relative residual estimate. |
| `"residual"` | True relative residual `norm(b - Ax) / norm(b)` of the returned solution. |

**Requirements and behavior:**

- Only supported for `method="cg"`.
- Requires a single RHS column (1D `b` or `(n, 1)` array).
- Accepts the same `tol`, `maxiter`, `preconditioner`, `ilu_fill_factor`, and `ilu_drop_tol` arguments as `solve`.
- Raises `RuntimeError` if CG does not converge (same as `solve`, including the true-residual check).

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

The returned object is a `peigen.sparse.SparseFactorized` (also exported as `peigen.decomp.SparseFactorized`, so `decomp.SparseFactorized(A)` is equivalent) and exposes `.shape`.

#### Sparse Cholesky (`cholesky`)

Factorize a sparse symmetric matrix once and reuse it. Wraps Eigen's `SimplicialLDLT` / `SimplicialLLT` and (experimental) `MultifrontalLDLT` with a fill-reducing ordering, so it needs no extra native dependency (no SuiteSparse/CHOLMOD) and is MPL2-licensed.

```python
fac = sparse.cholesky(K)                 # P K P^T = L D L^T, AMD ordering
x = fac.solve(b)                         # 1D or 2D b; many RHS are solved in one blocked pass
sign, logdet = fac.slogdet()             # like numpy.linalg.slogdet, without overflow

fac.refactorize(K_new)                   # new values, same sparsity pattern: reuses the symbolic analysis
fac.refactorize(K_new, shift=1e-6)       # factorize K_new + 1e-6 * I

reg = sparse.cholesky(K, shift=0.1)      # factorize K + 0.1 * I (regularization / GP jitter)
up = sparse.cholesky(sp.triu(K), lower=False)  # only one triangle is read

kkt = sparse.cholesky(K_indefinite, mode="multifrontal")  # pivoted LDL^T for indefinite matrices
n_pos, n_neg, n_zero = kkt.inertia      # eigenvalue sign counts, from the pivots of D

L, D, p = fac.L(), fac.D(), fac.P()      # the factors: K[p][:, p] == L @ D @ L.T (SciPy CSC)
```

| Parameter | Description |
|-----------|-------------|
| `mode="ldlt"` | `L D L^T` (no square roots). Default. Also factorizes some symmetric indefinite matrices, but without stability pivoting, so prefer SPD input. |
| `mode="llt"` | Standard Cholesky `L L^T`; requires positive definiteness. |
| `mode="multifrontal"` | *Experimental.* `L D L^T` computed as a tree of dense frontal matrices with level-3 updates (Eigen `MultifrontalLDLT`). Much faster on high-fill problems such as 3D meshes. `D` has 1x1 and 2x2 blocks chosen by threshold pivoting, so it is stable on symmetric **indefinite** matrices (KKT / saddle-point systems). |
| `ordering="amd"` | Approximate minimum degree fill-reducing ordering. Default. |
| `ordering="natural"` | No reordering (useful only for banded matrices or debugging). |
| `lower=True` | Read the lower triangle of `a`. `lower=False` reads the upper triangle. |
| `shift=0.0` | Factorize `a + shift * I`. |

`SparseCholesky` methods and properties: `solve(b)`, `refactorize(a, *, shift=None)` (`None` keeps the current shift), `slogdet()`, `logdet()`, `inertia` (`(positive, negative, zero)` eigenvalue counts, by Sylvester's law of inertia), `L()`, `D()` and `P()` (the factors as SciPy CSC matrices and the permutation as an index array, with `(A + shift*I)[p][:, p] == L @ D @ L.T` as in scikit-sparse; `L` has its diagonal stored, unit except for `llt`, and `D` is the identity for `llt` and has 2x2 blocks for `multifrontal`), `shape`, `mode`, `ordering`, `shift`, and `factor_nnz` (stored nonzeros in `L`, a measure of fill-in; for `multifrontal` it includes explicit zeros of merged fronts).

**Requirements and behavior:**

- `a` must be square. It is not checked for symmetry: only the selected triangle is referenced, as with `linalg.eigh`.
- Raises `numpy.linalg.LinAlgError` if factorization breaks down: `mode="llt"` on a matrix that is not positive definite, or a zero or non-finite pivot.
- For indefinite matrices use `mode="multifrontal"`; `mode="ldlt"` does not pivot and can be inaccurate or fail on them. `multifrontal`'s permutation also depends on the matrix's values (delayed pivots), and it does not scale the matrix, so badly scaled input should be equilibrated first.
- `refactorize` raises `ValueError` if the sparsity pattern (including explicit zeros) differs from the original matrix; create a new factorization instead.
- `logdet()` raises `ValueError` if the determinant is not positive; use `slogdet()` for indefinite input.
- Also exported as `peigen.decomp.SparseCholesky`.

**Performance.** `benchmarks/bench_cholesky.py` compares against SciPy SuperLU (`splu`) and, if scikit-sparse is installed, CHOLMOD. Eigen's factorization is *simplicial* (no dense supernode kernels), which sets its range. Measured on an Apple M1 Pro with the bundled Eigen (master, 2026-11-22):

- **Low-fill problems (2D meshes, graph Laplacians):** factorization is 1.8–6× faster than SciPy's default `splu`, 1.2–2× faster than `splu` configured for symmetric matrices, and 5–35% faster than CHOLMOD's simplicial mode. Solves with many right-hand sides use a blocked kernel and beat every CHOLMOD mode on these problems.
- **High-fill problems (3D meshes, very large 2D grids):** supernodal solvers win. CHOLMOD's default mode was about 4× faster on a 512² grid and 20–45× faster on a 40³ grid (20–30× single-threaded), and symmetric-mode `splu` was up to about 1.7× faster. pEigen is still about 1.7× faster than default `splu` there.
- Single-RHS solves are 1.4–2× slower than CHOLMOD's on low-fill problems, and up to 3× slower on 3D meshes.
- **`mode="multifrontal"` (experimental)** uses dense fronts, so it changes the high-fill picture: it factors 3D meshes 3–9× faster than `ldlt` (40³ grid: 10.9 s → 1.25 s) and is within about 15% of CHOLMOD's supernodal mode on 2D meshes and graphs. On 3D meshes it is still 1.3–3.7× slower than CHOLMOD with the same AMD ordering. Single-RHS solves are on par with CHOLMOD's. Solves with many right-hand sides use a supernodal blocked kernel and were the fastest of all tested solvers on every problem (32 RHS): 1.05–1.5× faster than `ldlt` on low-fill problems, 3–6× faster on 3D meshes, and 1.1–4.4× faster than CHOLMOD's best mode. That kernel's copy of `L` is built on the first many-RHS solve after each factorization, a one-time cost of 0.25–0.65× the factorization (2–9 steady-state solves), and it roughly doubles the factor's memory. `ldlt` stays the default while `multifrontal` is experimental.

### `peigen.decomp`

- `SVD(a, full_matrices=False)` with `.solve(b, rcond=None)`: minimum-norm least squares via the pseudoinverse. Singular values `<= rcond * max(s)` are treated as zero; `rcond` is relative, as in `numpy.linalg.lstsq`, and defaults to `eps * max(M, N)`.
- `QR(a)`
- `SparseFactorized(a)`
- `SparseCholesky(a, mode="ldlt", ...)`

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
eigen_version:         5.0.1-dev+master
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
