#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#if defined(PEIGEN_LAPACK_ENABLED)
#include "lapack_decls.h"
#endif

namespace py = pybind11;

#ifndef PEIGEN_PROJECT_VERSION
#define PEIGEN_PROJECT_VERSION "unknown"
#endif
#ifndef PEIGEN_BUILD_TYPE
#define PEIGEN_BUILD_TYPE "unknown"
#endif
#ifndef PEIGEN_BLAS_BACKEND
#define PEIGEN_BLAS_BACKEND "unknown"
#endif
#ifndef PEIGEN_LAPACK_BACKEND
#define PEIGEN_LAPACK_BACKEND "none"
#endif
#ifndef PEIGEN_OPENMP_ENABLED
#define PEIGEN_OPENMP_ENABLED 0
#endif

using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct SvdFactors {
  ColMatrix u;
  Eigen::VectorXd s;
  ColMatrix vt;
};

static void validate_2d(const py::array &arr, const std::string &name) {
  if (arr.ndim() != 2) {
    throw py::value_error(name + " must be a 2D array");
  }
}

static bool is_c_contiguous(const py::array &arr) {
  return arr.strides(0) == static_cast<ssize_t>(arr.itemsize() * arr.shape(1)) &&
         arr.strides(1) == static_cast<ssize_t>(arr.itemsize());
}

static bool is_f_contiguous(const py::array &arr) {
  return arr.strides(0) == static_cast<ssize_t>(arr.itemsize()) &&
         arr.strides(1) == static_cast<ssize_t>(arr.itemsize() * arr.shape(0));
}

static Eigen::Ref<const RowMatrix> dense_row_ref(const py::array_t<double, py::array::forcecast> &arr,
                                                 const std::string &name,
                                                 std::unique_ptr<RowMatrix> &storage) {
  validate_2d(arr, name);
  const Eigen::Index rows = arr.shape(0);
  const Eigen::Index cols = arr.shape(1);

  if (is_c_contiguous(arr)) {
    const auto *data = static_cast<const double *>(arr.data());
    return Eigen::Map<const RowMatrix>(data, rows, cols);
  }

  storage = std::make_unique<RowMatrix>(rows, cols);
  Eigen::Map<RowMatrix> dst(storage->data(), rows, cols);

  if (is_f_contiguous(arr)) {
    const auto *data = static_cast<const double *>(arr.data());
    const Eigen::Map<const ColMatrix> src(data, rows, cols);
    dst = src;
  } else {
    const auto buf = arr.unchecked<2>();
    for (Eigen::Index i = 0; i < rows; ++i) {
      for (Eigen::Index j = 0; j < cols; ++j) {
        dst(i, j) = buf(i, j);
      }
    }
  }

  return *storage;
}

static ColMatrix dense_col_for_factorization(const py::array_t<double, py::array::forcecast> &arr,
                                             const std::string &name) {
  validate_2d(arr, name);
  const Eigen::Index rows = arr.shape(0);
  const Eigen::Index cols = arr.shape(1);

  if (is_f_contiguous(arr)) {
    const auto *data = static_cast<const double *>(arr.data());
    const Eigen::Map<const ColMatrix> mapped(data, rows, cols);
    return ColMatrix(mapped);
  }

  if (is_c_contiguous(arr)) {
    const auto *data = static_cast<const double *>(arr.data());
    const Eigen::Map<const RowMatrix> mapped(data, rows, cols);
    ColMatrix out(rows, cols);
    out = mapped;
    return out;
  }

  ColMatrix out(rows, cols);
  const auto buf = arr.unchecked<2>();
  for (Eigen::Index i = 0; i < rows; ++i) {
    for (Eigen::Index j = 0; j < cols; ++j) {
      out(i, j) = buf(i, j);
    }
  }
  return out;
}

static py::array_t<double> make_output_array(Eigen::Index rows, Eigen::Index cols) {
  return py::array_t<double>({rows, cols});
}

template <typename Derived>
static py::array_t<double> assign_to_output(const Eigen::DenseBase<Derived> &expr) {
  py::array_t<double> arr = make_output_array(expr.rows(), expr.cols());
  Eigen::Map<RowMatrix> out(arr.mutable_data(), expr.rows(), expr.cols());
  out = expr;
  return arr;
}

static py::array_t<double> vector_to_numpy(const Eigen::VectorXd &v) {
  py::array_t<double> arr(static_cast<py::ssize_t>(v.size()));
  std::memcpy(arr.mutable_data(), v.data(), sizeof(double) * static_cast<size_t>(v.size()));
  return arr;
}

static std::string resolve_svd_method(const std::string &method) {
  if (method == "auto") {
#if defined(PEIGEN_LAPACK_ENABLED)
    return "lapack";
#else
    return "bdcsvd";
#endif
  }
  if (method == "bdcsvd" || method == "lapack") {
#if !defined(PEIGEN_LAPACK_ENABLED)
    if (method == "lapack") {
      throw py::value_error("LAPACK SVD requested but this build was compiled without LAPACK support");
    }
#endif
    return method;
  }
  throw py::value_error("method must be one of: auto, bdcsvd, lapack");
}

static SvdFactors compute_svd_bdcsvd(const ColMatrix &matrix, bool full_matrices) {
  if (full_matrices) {
    Eigen::BDCSVD<ColMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(matrix);
    if (svd.info() != Eigen::Success) {
      throw std::runtime_error("BDCSVD failed");
    }
    SvdFactors out;
    out.u = svd.matrixU();
    out.s = svd.singularValues();
    out.vt = svd.matrixV().adjoint();
    return out;
  }

  Eigen::BDCSVD<ColMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(matrix);
  if (svd.info() != Eigen::Success) {
    throw std::runtime_error("BDCSVD failed");
  }

  SvdFactors out;
  out.u = svd.matrixU();
  out.s = svd.singularValues();
  out.vt = svd.matrixV().adjoint();
  return out;
}

#if defined(PEIGEN_LAPACK_ENABLED)
static SvdFactors compute_svd_lapack(ColMatrix matrix, bool full_matrices) {
  lapack_int m = static_cast<lapack_int>(matrix.rows());
  lapack_int n = static_cast<lapack_int>(matrix.cols());
  const lapack_int k = std::min(m, n);
  char jobz = full_matrices ? 'A' : 'S';

  SvdFactors out;
  out.s = Eigen::VectorXd::Zero(k);

  const lapack_int u_cols = (jobz == 'A') ? m : k;
  const lapack_int vt_rows = (jobz == 'A') ? n : k;
  out.u = ColMatrix::Zero(m, u_cols);
  ColMatrix vt = ColMatrix::Zero(vt_rows, n);

  lapack_int lda = static_cast<lapack_int>(matrix.outerStride());
  lapack_int ldu = static_cast<lapack_int>(out.u.outerStride());
  lapack_int ldvt = static_cast<lapack_int>(vt.outerStride());
  lapack_int lwork = -1;
  double work_query = 0.0;
  lapack_int info = 0;
  std::vector<lapack_int> iwork(static_cast<std::size_t>(8) * static_cast<std::size_t>(k));

  dgesdd_(&jobz, &m, &n, matrix.data(), &lda, out.s.data(), out.u.data(), &ldu, vt.data(), &ldvt,
                   &work_query, &lwork, iwork.data(), &info, 1);
  if (info != 0) {
    throw std::runtime_error("LAPACK dgesdd workspace query failed with info=" + std::to_string(info));
  }

  lwork = static_cast<lapack_int>(work_query);
  std::vector<double> work(static_cast<std::size_t>(lwork));

  dgesdd_(&jobz, &m, &n, matrix.data(), &lda, out.s.data(), out.u.data(), &ldu, vt.data(), &ldvt,
                   work.data(), &lwork, iwork.data(), &info, 1);
  if (info != 0) {
    throw std::runtime_error("LAPACK dgesdd failed with info=" + std::to_string(info));
  }

  out.vt = vt;
  return out;
}
#endif

static SvdFactors compute_svd(const ColMatrix &matrix, bool full_matrices, const std::string &method) {
  const std::string resolved = resolve_svd_method(method);
  if (resolved == "lapack") {
#if defined(PEIGEN_LAPACK_ENABLED)
    return compute_svd_lapack(matrix, full_matrices);
#else
    throw py::value_error("LAPACK SVD requested but LAPACK is unavailable in this build");
#endif
  }
  return compute_svd_bdcsvd(matrix, full_matrices);
}

struct EighFactors {
  Eigen::VectorXd w;
  ColMatrix v;
};

static std::string resolve_eigh_method(const std::string &method) {
  if (method == "auto") {
#if defined(PEIGEN_LAPACK_ENABLED)
    return "lapack";
#else
    return "eigen";
#endif
  }
  if (method == "eigen" || method == "lapack") {
#if !defined(PEIGEN_LAPACK_ENABLED)
    if (method == "lapack") {
      throw py::value_error("LAPACK eigh requested but this build was compiled without LAPACK support");
    }
#endif
    return method;
  }
  throw py::value_error("method must be one of: auto, eigen, lapack");
}

static std::string resolve_lapack_eigen_method(const std::string &method, const char *routine) {
  if (method == "auto") {
#if defined(PEIGEN_LAPACK_ENABLED)
    return "lapack";
#else
    return "eigen";
#endif
  }
  if (method == "eigen" || method == "lapack") {
#if !defined(PEIGEN_LAPACK_ENABLED)
    if (method == "lapack") {
      throw py::value_error(std::string("LAPACK ") + routine +
                            " requested but this build was compiled without LAPACK support");
    }
#endif
    return method;
  }
  throw py::value_error("method must be one of: auto, eigen, lapack");
}

static ColMatrix dense_col_from_row_ref(const Eigen::Ref<const RowMatrix> &rhs) {
  ColMatrix out(rhs.rows(), rhs.cols());
  out = rhs;
  return out;
}

static EighFactors compute_eigh_eigen(const ColMatrix &matrix, bool lower, bool eigenvectors) {
  const unsigned options = eigenvectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly;
  Eigen::SelfAdjointEigenSolver<ColMatrix> solver;
  if (lower) {
    solver.compute(matrix, options);
  } else {
    solver.compute(matrix.selfadjointView<Eigen::Upper>(), options);
  }

  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("SelfAdjointEigenSolver failed");
  }

  EighFactors out;
  out.w = solver.eigenvalues();
  if (eigenvectors) {
    out.v = solver.eigenvectors();
  }
  return out;
}

#if defined(PEIGEN_LAPACK_ENABLED)
static void lapack_dsyev_values(ColMatrix &matrix, bool lower, Eigen::VectorXd &w) {
  lapack_int n = static_cast<lapack_int>(matrix.rows());
  char jobz = 'N';
  char uplo = lower ? 'L' : 'U';
  lapack_int lda = static_cast<lapack_int>(matrix.outerStride());
  lapack_int lwork = -1;
  lapack_int info = 0;
  double work_query = 0.0;

  dsyev_(&jobz, &uplo, &n, matrix.data(), &lda, w.data(), &work_query, &lwork, &info, 1, 1);
  if (info != 0) {
    throw std::runtime_error("LAPACK dsyev workspace query failed with info=" + std::to_string(info));
  }

  lwork = static_cast<lapack_int>(work_query);
  std::vector<double> work(static_cast<std::size_t>(lwork));

  dsyev_(&jobz, &uplo, &n, matrix.data(), &lda, w.data(), work.data(), &lwork, &info, 1, 1);
  if (info != 0) {
    throw std::runtime_error("LAPACK dsyev failed with info=" + std::to_string(info));
  }
}

static bool lapack_dsyevr_values(ColMatrix &matrix, bool lower, Eigen::VectorXd &w) {
  lapack_int n = static_cast<lapack_int>(matrix.rows());
  char jobz = 'N';
  char range = 'A';
  char uplo = lower ? 'L' : 'U';
  lapack_int lda = static_cast<lapack_int>(matrix.outerStride());
  double vl = 0.0;
  double vu = 0.0;
  lapack_int il = 1;
  lapack_int iu = n;
  double abstol = 0.0;
  lapack_int m = 0;
  lapack_int ldz = 1;
  lapack_int lwork = -1;
  lapack_int liwork = -1;
  lapack_int info = 0;
  double work_query = 0.0;
  lapack_int iwork_query = 0;

  dsyevr_(&jobz, &range, &uplo, &n, matrix.data(), &lda, &vl, &vu, &il, &iu, &abstol, &m, w.data(),
                   nullptr, &ldz, nullptr, &work_query, &lwork, &iwork_query, &liwork, &info, 1, 1, 1);
  if (info != 0) {
    return false;
  }

  lwork = static_cast<lapack_int>(work_query);
  liwork = iwork_query;
  std::vector<double> work(static_cast<std::size_t>(lwork));
  std::vector<lapack_int> iwork(static_cast<std::size_t>(liwork));

  dsyevr_(&jobz, &range, &uplo, &n, matrix.data(), &lda, &vl, &vu, &il, &iu, &abstol, &m, w.data(),
                   nullptr, &ldz, nullptr, work.data(), &lwork, iwork.data(), &liwork, &info, 1, 1, 1);
  if (info != 0 || m != n) {
    return false;
  }
  return true;
}

static EighFactors compute_eigh_lapack_vals(ColMatrix matrix, bool lower) {
  if (matrix.rows() != matrix.cols()) {
    throw py::value_error("eigh requires square matrix");
  }

  const lapack_int n = static_cast<lapack_int>(matrix.rows());
  EighFactors out;
  out.w = Eigen::VectorXd::Zero(n);

  if (!lapack_dsyevr_values(matrix, lower, out.w)) {
    lapack_dsyev_values(matrix, lower, out.w);
  }
  return out;
}

static EighFactors compute_eigh_lapack(ColMatrix matrix, bool lower, bool eigenvectors) {
  if (matrix.rows() != matrix.cols()) {
    throw py::value_error("eigh requires square matrix");
  }

  if (!eigenvectors) {
    return compute_eigh_lapack_vals(matrix, lower);
  }

  lapack_int n = static_cast<lapack_int>(matrix.rows());
  char jobz = 'V';
  char uplo = lower ? 'L' : 'U';
  lapack_int lda = static_cast<lapack_int>(matrix.outerStride());
  lapack_int lwork = -1;
  lapack_int liwork = -1;
  double work_query = 0.0;
  lapack_int iwork_query = 0;
  lapack_int info = 0;

  EighFactors out;
  out.w = Eigen::VectorXd::Zero(n);

  dsyevd_(&jobz, &uplo, &n, matrix.data(), &lda, out.w.data(), &work_query, &lwork, &iwork_query,
                   &liwork, &info, 1, 1);
  if (info != 0) {
    throw std::runtime_error("LAPACK dsyevd workspace query failed with info=" + std::to_string(info));
  }

  lwork = static_cast<lapack_int>(work_query);
  liwork = iwork_query;
  std::vector<double> work(static_cast<std::size_t>(lwork));
  std::vector<lapack_int> iwork(static_cast<std::size_t>(liwork));

  dsyevd_(&jobz, &uplo, &n, matrix.data(), &lda, out.w.data(), work.data(), &lwork, iwork.data(), &liwork,
                   &info, 1, 1);
  if (info != 0) {
    throw std::runtime_error("LAPACK dsyevd failed with info=" + std::to_string(info));
  }

  out.v = std::move(matrix);
  return out;
}
#endif

static EighFactors compute_eigh(const ColMatrix &matrix, bool lower, bool eigenvectors,
                                const std::string &method) {
  const std::string resolved = resolve_eigh_method(method);
  if (resolved == "lapack") {
#if defined(PEIGEN_LAPACK_ENABLED)
    return compute_eigh_lapack(matrix, lower, eigenvectors);
#else
    throw py::value_error("LAPACK eigh requested but LAPACK is unavailable in this build");
#endif
  }
  return compute_eigh_eigen(matrix, lower, eigenvectors);
}

static py::tuple svd_to_python(const SvdFactors &factors) {
  return py::make_tuple(assign_to_output(factors.u), vector_to_numpy(factors.s), assign_to_output(factors.vt));
}

// Zero-copy (when possible) view of a SciPy sparse matrix as an Eigen CSC map.
//
// The Eigen::Map points into the three arrays below, so the view must own them: when SciPy stores
// indices as int64 or data as anything other than float64, the forcecast produces fresh
// converted arrays whose only reference is held here.
struct SparseCscView {
  py::object owner;
  py::array_t<int, py::array::c_style | py::array::forcecast> indptr;
  py::array_t<int, py::array::c_style | py::array::forcecast> indices;
  py::array_t<double, py::array::c_style | py::array::forcecast> data;
  Eigen::Map<const Sparse> mat;

  SparseCscView(py::object owner_, py::array_t<int, py::array::c_style | py::array::forcecast> indptr_,
                py::array_t<int, py::array::c_style | py::array::forcecast> indices_,
                py::array_t<double, py::array::c_style | py::array::forcecast> data_, int rows, int cols,
                int nnz)
      : owner(std::move(owner_)),
        indptr(std::move(indptr_)),
        indices(std::move(indices_)),
        data(std::move(data_)),
        mat(rows, cols, nnz, indptr.data(), indices.data(), data.data()) {}

  SparseCscView(const SparseCscView &) = delete;
  SparseCscView &operator=(const SparseCscView &) = delete;
};

static int checked_int32(py::handle value, const char *what) {
  const long long v = value.cast<long long>();
  if (v < 0 || v > static_cast<long long>(std::numeric_limits<int>::max())) {
    throw py::value_error(std::string("sparse matrix ") + what +
                          " exceeds the 32-bit index range supported by pEigen");
  }
  return static_cast<int>(v);
}

static std::unique_ptr<SparseCscView> map_sparse_csc(py::object matrix_obj) {
  py::module_ scipy_sparse = py::module_::import("scipy.sparse");

  py::object csc_obj;
  if (!scipy_sparse.attr("issparse")(matrix_obj).cast<bool>()) {
    csc_obj = scipy_sparse.attr("csc_matrix")(matrix_obj);
  } else if (matrix_obj.attr("format").cast<std::string>() == "csc") {
    csc_obj = matrix_obj;
  } else {
    csc_obj = matrix_obj.attr("tocsc")();
  }

  const py::object dtype = csc_obj.attr("dtype");
  if (dtype.attr("kind").cast<std::string>() == "c") {
    throw py::type_error("complex sparse matrices are not supported; pEigen operates on real float64 data");
  }

  // Eigen requires sorted, duplicate-free inner indices. Canonicalize a copy so the caller's
  // matrix is never mutated.
  if (!csc_obj.attr("has_canonical_format").cast<bool>()) {
    csc_obj = csc_obj.attr("copy")();
    csc_obj.attr("sum_duplicates")();
  }

  const auto shape = csc_obj.attr("shape").cast<py::tuple>();
  const int rows = checked_int32(shape[0], "row count");
  const int cols = checked_int32(shape[1], "column count");
  const int nnz = checked_int32(csc_obj.attr("nnz"), "nonzero count");

  // Index values are bounded by the shape and nnz checked above, so narrowing to int32 is lossless.
  using IndexArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
  using DataArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
  auto indptr = IndexArray::ensure(csc_obj.attr("indptr"));
  auto indices = IndexArray::ensure(csc_obj.attr("indices"));
  auto data = DataArray::ensure(csc_obj.attr("data"));
  if (!indptr || !indices || !data) {
    throw py::type_error("sparse matrix index/data arrays could not be converted to int32/float64");
  }

  return std::make_unique<SparseCscView>(std::move(csc_obj), std::move(indptr), std::move(indices),
                                         std::move(data), rows, cols, nnz);
}

static py::object to_scipy_csc(const Sparse &mat) {
  py::module_ sparse_mod = py::module_::import("scipy.sparse");

  Sparse cpy = mat;
  cpy.makeCompressed();

  py::array_t<double> data(cpy.nonZeros());
  py::array_t<int> indices(cpy.nonZeros());
  py::array_t<int> indptr(cpy.outerSize() + 1);

  std::memcpy(data.mutable_data(), cpy.valuePtr(), sizeof(double) * cpy.nonZeros());
  std::memcpy(indices.mutable_data(), cpy.innerIndexPtr(), sizeof(int) * cpy.nonZeros());
  std::memcpy(indptr.mutable_data(), cpy.outerIndexPtr(), sizeof(int) * (cpy.outerSize() + 1));

  py::tuple shape = py::make_tuple(cpy.rows(), cpy.cols());
  py::tuple args = py::make_tuple(py::make_tuple(data, indices, indptr), shape);
  return sparse_mod.attr("csc_matrix")(*args);
}

static py::array_t<double> core_matmul(const py::array_t<double, py::array::forcecast> &a,
                                       const py::array_t<double, py::array::forcecast> &b) {
  std::unique_ptr<RowMatrix> owned_a;
  std::unique_ptr<RowMatrix> owned_b;
  const Eigen::Ref<const RowMatrix> lhs = dense_row_ref(a, "a", owned_a);
  const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);

  if (lhs.cols() != rhs.rows()) {
    throw py::value_error("matmul dimension mismatch");
  }

  py::array_t<double> out_arr = make_output_array(lhs.rows(), rhs.cols());
  Eigen::Map<RowMatrix> out(out_arr.mutable_data(), lhs.rows(), rhs.cols());
  out.noalias() = lhs * rhs;
  return out_arr;
}

#if defined(PEIGEN_LAPACK_ENABLED)
static py::array_t<double> solve_lapack(ColMatrix lhs, ColMatrix rhs) {
  lapack_int n = static_cast<lapack_int>(lhs.rows());
  lapack_int nrhs = static_cast<lapack_int>(rhs.cols());
  lapack_int lda = static_cast<lapack_int>(lhs.outerStride());
  lapack_int ldb = static_cast<lapack_int>(rhs.outerStride());
  lapack_int info = 0;
  std::vector<lapack_int> ipiv(static_cast<std::size_t>(n));

  dgesv_(&n, &nrhs, lhs.data(), &lda, ipiv.data(), rhs.data(), &ldb, &info);
  if (info != 0) {
    throw py::value_error("matrix is singular or ill-conditioned");
  }
  return assign_to_output(rhs);
}
#endif

static double core_norm(const py::array_t<double, py::array::forcecast> &a) {
  validate_2d(a, "a");
  const Eigen::Index rows = a.shape(0);
  const Eigen::Index cols = a.shape(1);
  const Eigen::Index size = rows * cols;

  if (is_c_contiguous(a) || is_f_contiguous(a)) {
    const auto *data = static_cast<const double *>(a.data());
    return Eigen::Map<const Vector>(data, size).norm();
  }

  std::unique_ptr<RowMatrix> owned_a;
  const Eigen::Ref<const RowMatrix> m = dense_row_ref(a, "a", owned_a);
  return m.norm();
}

static py::array_t<double> core_solve(const py::array_t<double, py::array::forcecast> &a,
                                      const py::array_t<double, py::array::forcecast> &b,
                                      const std::string &method) {
  const ColMatrix lhs = dense_col_for_factorization(a, "a");
  std::unique_ptr<RowMatrix> owned_b;
  const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);

  if (lhs.rows() != lhs.cols()) {
    throw py::value_error("a must be square");
  }
  if (lhs.rows() != rhs.rows()) {
    throw py::value_error("a and b shape mismatch");
  }

  const std::string resolved = resolve_lapack_eigen_method(method, "solve");
  if (resolved == "lapack") {
#if defined(PEIGEN_LAPACK_ENABLED)
    return solve_lapack(lhs, dense_col_from_row_ref(rhs));
#else
    throw py::value_error("LAPACK solve requested but LAPACK is unavailable in this build");
#endif
  }

  Eigen::PartialPivLU<ColMatrix> lu(lhs);
  if (lhs.rows() > 0) {
    // Scale-aware singularity test: compare the smallest pivot to the largest rather than to a
    // fixed constant, so well-conditioned matrices with tiny entries (e.g. 1e-16 * I) are accepted.
    // The negated comparison also rejects NaN pivots.
    const Vector pivots = lu.matrixLU().diagonal().cwiseAbs();
    const double max_pivot = pivots.maxCoeff();
    const double tol = static_cast<double>(lhs.rows()) * std::numeric_limits<double>::epsilon() * max_pivot;
    if (!(max_pivot > 0.0) || !(pivots.minCoeff() > tol) || !std::isfinite(max_pivot)) {
      throw py::value_error("matrix is singular or ill-conditioned");
    }
  }

  ColMatrix rhs_col = dense_col_from_row_ref(rhs);
  return assign_to_output(lu.solve(rhs_col));
}

static py::tuple core_qr(const py::array_t<double, py::array::forcecast> &a,
                         const std::string &mode) {
  const ColMatrix m = dense_col_for_factorization(a, "a");
  Eigen::HouseholderQR<ColMatrix> qr(m);

  const int rows = static_cast<int>(m.rows());
  const int cols = static_cast<int>(m.cols());
  const int k = std::min(rows, cols);
  const ColMatrix r_full = qr.matrixQR().template triangularView<Eigen::Upper>();

  if (mode == "reduced") {
    const ColMatrix q_thin = qr.householderQ() * ColMatrix::Identity(rows, k);
    return py::make_tuple(assign_to_output(q_thin), assign_to_output(r_full.topRows(k)));
  }
  if (mode == "complete") {
    const ColMatrix q_full = qr.householderQ() * ColMatrix::Identity(rows, rows);
    return py::make_tuple(assign_to_output(q_full), assign_to_output(r_full));
  }

  throw py::value_error("mode must be 'reduced' or 'complete'");
}

static py::tuple core_svd(const py::array_t<double, py::array::forcecast> &a,
                          bool full_matrices,
                          const std::string &method) {
  const ColMatrix matrix = dense_col_for_factorization(a, "a");
  return svd_to_python(compute_svd(matrix, full_matrices, method));
}

static double core_svd_compute(const py::array_t<double, py::array::forcecast> &a,
                               bool full_matrices,
                               const std::string &method) {
  const ColMatrix matrix = dense_col_for_factorization(a, "a");
  const SvdFactors factors = compute_svd(matrix, full_matrices, method);
  return factors.s.sum();
}

static py::object core_eigh(const py::array_t<double, py::array::forcecast> &a, bool lower, bool eigenvectors,
                            const std::string &method) {
  const ColMatrix m = dense_col_for_factorization(a, "a");
  if (m.rows() != m.cols()) {
    throw py::value_error("eigh requires square matrix");
  }

  const EighFactors factors = compute_eigh(m, lower, eigenvectors, method);
  if (eigenvectors) {
    return py::make_tuple(vector_to_numpy(factors.w), assign_to_output(factors.v));
  }
  return vector_to_numpy(factors.w);
}

static double core_eigh_compute(const py::array_t<double, py::array::forcecast> &a, bool lower,
                                const std::string &method) {
  const ColMatrix m = dense_col_for_factorization(a, "a");
  if (m.rows() != m.cols()) {
    throw py::value_error("eigh requires square matrix");
  }

  const EighFactors factors = compute_eigh(m, lower, false, method);
  return factors.w.sum();
}

class SparseFactorized {
 public:
  explicit SparseFactorized(const Sparse &a) : lu_(std::make_unique<Eigen::SparseLU<Sparse>>()) {
    lu_->analyzePattern(a);
    lu_->factorize(a);
    if (lu_->info() != Eigen::Success) {
      throw std::runtime_error("sparse factorization failed");
    }
  }

  py::array_t<double> solve(const py::array_t<double, py::array::forcecast> &b) const {
    std::unique_ptr<RowMatrix> owned_b;
    const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);
    if (rhs.rows() != lu_->rows()) {
      throw py::value_error("factorized matrix and rhs shape mismatch");
    }

    py::array_t<double> out_arr = make_output_array(rhs.rows(), rhs.cols());
    Eigen::Map<RowMatrix> out(out_arr.mutable_data(), rhs.rows(), rhs.cols());
    for (int col = 0; col < rhs.cols(); ++col) {
      out.col(col) = lu_->solve(rhs.col(col));
      if (lu_->info() != Eigen::Success) {
        throw std::runtime_error("SparseLU solve failed");
      }
    }
    return out_arr;
  }

  Eigen::Index rows() const { return lu_->rows(); }
  Eigen::Index cols() const { return lu_->cols(); }

 private:
  std::unique_ptr<Eigen::SparseLU<Sparse>> lu_;
};

// ---------------------------------------------------------------------------------------------
// Sparse Cholesky (SimplicialLDLT / SimplicialLLT)
// ---------------------------------------------------------------------------------------------

[[noreturn]] static void raise_linalg_error(const std::string &msg) {
  const py::object exc = py::module_::import("numpy.linalg").attr("LinAlgError");
  PyErr_SetString(exc.ptr(), msg.c_str());
  throw py::error_already_set();
}

// Type-erased interface over the (simplicial LDLT | simplicial LLT | multifrontal LDLT) x
// (AMD | natural) Eigen solver instantiations.
class CholeskyBackend {
 public:
  virtual ~CholeskyBackend() = default;
  virtual void analyze(const Sparse &a) = 0;
  virtual void factorize(const Sparse &a, double shift) = 0;
  virtual Eigen::ComputationInfo info() const = 0;
  // Solve A x = b for all columns of b at once; writes x (row-major, same shape as b) into out.
  virtual void solve_into(const Eigen::Ref<const RowMatrix> &b, double *out) const = 0;
  virtual std::pair<double, double> slogdet() const = 0;
  virtual Eigen::Index factor_nnz() const = 0;
  // (positive, negative, zero) pivot counts: the inertia of the factorized matrix by Sylvester's law.
  virtual std::array<Eigen::Index, 3> inertia() const = 0;
};

template <typename Solver, bool IsLDLT>
class CholeskyBackendImpl final : public CholeskyBackend {
 public:
  void analyze(const Sparse &a) override { solver_.analyzePattern(a); }

  void factorize(const Sparse &a, double shift) override {
    solver_.setShift(shift);
    solver_.factorize(a);
  }

  Eigen::ComputationInfo info() const override { return solver_.info(); }

  // Blocked multi-RHS solve. Eigen's built-in solve handles one column at a time, re-reading the
  // whole factor per RHS. Keeping the RHS block row-major lets each nonzero L(i, j) update a
  // contiguous row of k values, so L is streamed once regardless of k.
  //
  // Factor layout (Eigen SimplicialCholesky): L is column-major; for LLT the diagonal is the first
  // entry of each column, for LDLT L is unit-lower (diagonal not stored) and D is kept separately.
  void solve_into(const Eigen::Ref<const RowMatrix> &b, double *out) const override {
    const auto &L = solver_.matrixL().nestedExpression();
    const Eigen::Index n = L.rows();
    const Eigen::Index k = b.cols();
    const auto &perm = solver_.permutationP();
    const bool permuted = perm.size() > 0;

    // x = P b   (Eigen convention: (P b)[perm[i]] = b[i])
    std::vector<double> x(static_cast<std::size_t>(n * k));
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::Index dst = permuted ? perm.indices()[i] : i;
      std::copy_n(&b(i, 0), k, &x[static_cast<std::size_t>(dst * k)]);
    }

    // Raw CSC access (columns may be uncompressed: use innerNonZeroPtr when present).
    const int *Lp = L.outerIndexPtr();
    const int *Lnz = L.innerNonZeroPtr();
    const int *Li = L.innerIndexPtr();
    const double *Lx = L.valuePtr();
    auto col_end = [&](Eigen::Index j) -> int { return Lnz ? Lp[j] + Lnz[j] : Lp[j + 1]; };

    // Reciprocal diagonal: D^{-1} for LDLT, diag(L)^{-1} for LLT (computed off the critical path).
    Vector inv_diag(n);
    if constexpr (IsLDLT) {
      inv_diag = solver_.vectorD().cwiseInverse();
    } else {
      for (Eigen::Index j = 0; j < n; ++j) inv_diag[j] = 1.0 / Lx[Lp[j]];
    }
    constexpr int skip_diag = IsLDLT ? 0 : 1;  // LLT stores the diagonal first in each column
    double *X = x.data();

    if (k == 1) {
      // Forward: L y = x
      for (Eigen::Index j = 0; j < n; ++j) {
        if constexpr (!IsLDLT) X[j] *= inv_diag[j];
        const double xj = X[j];
        if (xj != 0.0) {
          for (int p = Lp[j] + skip_diag, e = col_end(j); p < e; ++p) X[Li[p]] -= Lx[p] * xj;
        }
      }
      if constexpr (IsLDLT) {
        for (Eigen::Index j = 0; j < n; ++j) X[j] *= inv_diag[j];
      }
      // Backward: L^T z = y
      for (Eigen::Index j = n - 1; j >= 0; --j) {
        double acc = X[j];
        for (int p = Lp[j] + skip_diag, e = col_end(j); p < e; ++p) acc -= Lx[p] * X[Li[p]];
        if constexpr (!IsLDLT) acc *= inv_diag[j];
        X[j] = acc;
      }
    } else {
      // Forward: L y = x (column-oriented: scatter column j into later rows).
      for (Eigen::Index j = 0; j < n; ++j) {
        double *xj = X + j * k;
        if constexpr (!IsLDLT) {
          const double inv = inv_diag[j];
          for (Eigen::Index c = 0; c < k; ++c) xj[c] *= inv;
        }
        for (int p = Lp[j] + skip_diag, e = col_end(j); p < e; ++p) {
          const double v = Lx[p];
          double *xi = X + static_cast<Eigen::Index>(Li[p]) * k;
          for (Eigen::Index c = 0; c < k; ++c) xi[c] -= v * xj[c];
        }
      }
      if constexpr (IsLDLT) {
        for (Eigen::Index j = 0; j < n; ++j) {
          const double inv = inv_diag[j];
          double *xj = X + j * k;
          for (Eigen::Index c = 0; c < k; ++c) xj[c] *= inv;
        }
      }
      // Backward: L^T z = y (row j of L^T is column j of L: gather from later rows).
      for (Eigen::Index j = n - 1; j >= 0; --j) {
        double *xj = X + j * k;
        for (int p = Lp[j] + skip_diag, e = col_end(j); p < e; ++p) {
          const double v = Lx[p];
          const double *xi = X + static_cast<Eigen::Index>(Li[p]) * k;
          for (Eigen::Index c = 0; c < k; ++c) xj[c] -= v * xi[c];
        }
        if constexpr (!IsLDLT) {
          const double inv = inv_diag[j];
          for (Eigen::Index c = 0; c < k; ++c) xj[c] *= inv;
        }
      }
    }

    // out = P^{-1} z   (out[i] = z[perm[i]])
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::Index src = permuted ? perm.indices()[i] : i;
      std::copy_n(X + src * k, k, out + i * k);
    }
  }

  // (sign, log|det|) of the factorized matrix, as numpy.linalg.slogdet. The fill-reducing
  // permutation is symmetric (P A P^T), so it does not change the determinant.
  std::pair<double, double> slogdet() const override {
    if constexpr (IsLDLT) {
      const Vector d = solver_.vectorD();
      double sign = 1.0;
      double logabs = 0.0;
      for (Eigen::Index i = 0; i < d.size(); ++i) {
        if (d[i] == 0.0) {
          return {0.0, -std::numeric_limits<double>::infinity()};
        }
        if (d[i] < 0.0) {
          sign = -sign;
        }
        logabs += std::log(std::abs(d[i]));
      }
      return {sign, logabs};
    } else {
      const Vector diag = solver_.matrixL().nestedExpression().diagonal();
      return {1.0, 2.0 * diag.array().log().sum()};
    }
  }

  Eigen::Index factor_nnz() const override { return solver_.matrixL().nestedExpression().nonZeros(); }

  std::array<Eigen::Index, 3> inertia() const override {
    const Eigen::Index n = solver_.rows();
    if constexpr (IsLDLT) {
      const Vector d = solver_.vectorD();
      const Eigen::Index pos = (d.array() > 0.0).count();
      const Eigen::Index neg = (d.array() < 0.0).count();
      return {pos, neg, n - pos - neg};
    } else {
      return {n, 0, 0};  // LLT only succeeds on positive definite input
    }
  }

 private:
  Solver solver_;
};

// The factor of a multifrontal LDLT in supernodal form, for blocked many-RHS solves.
//
// Eigen's MultifrontalLDLT::solve applies large fronts as dense products but solves small fronts
// (most of them, on 2D meshes and graphs) one right-hand side at a time over a column-major X, so
// it re-reads L once per column. Here X is row-major, so every entry L(i, j) updates a contiguous
// row of k values, and large supernodes still go through dense TRSM / GEMM.
//
// The fronts themselves are private to Eigen, so the supernodes are recovered from matrixL(): it
// keeps each front's explicit zeros, so a front's pivot columns form a dense trapezoid, and
// consecutive columns whose row patterns nest (column j+1 = column j without row j) are merged.
// Each supernode is packed as a dense column-major (order x width) block whose first `width` rows
// are its own pivots, which are consecutive in pivot order.
class SupernodalLDLT {
 public:
  template <typename Solver>
  explicit SupernodalLDLT(const Solver &solver) : n_(solver.rows()) {
    const auto &perm = solver.permutationP().indices();
    perm_.assign(perm.data(), perm.data() + n_);
    build_blocks(solver.matrixL());
    build_dinv(solver.vectorD(), solver.subDiagonal());
  }

  // b, out: row-major n x k.
  void solve_into(const Eigen::Ref<const RowMatrix> &b, double *out) const {
    const Eigen::Index k = b.cols();
    // X = P b: (P b)[perm[i]] = b[i]
    RowMatrix X(n_, k);
    for (Eigen::Index i = 0; i < n_; ++i) X.row(perm_[i]) = b.row(i);
    RowMatrix scratch(max_below_, k);
    double *x = X.data();

    // L Y = X
    for (const Super &s : supers_) {
      const double *lv = values_.data() + s.values;
      const int *rows = rows_.data() + s.rows;
      const Eigen::Index m = s.order, c = s.width, below = m - c;
      if (dense(s)) {
        Eigen::Map<const ColMatrix> blk(lv, m, c);
        Eigen::Map<RowMatrix> top(x + s.first * k, c, k);
        blk.topRows(c).triangularView<Eigen::UnitLower>().solveInPlace(top);
        auto t = scratch.topRows(below);
        t.noalias() = blk.bottomRows(below) * top;
        for (Eigen::Index i = 0; i < below; ++i) X.row(rows[c + i]) -= t.row(i);
      } else {
        for (Eigen::Index j = 0; j < c; ++j) {
          const double *xj = x + (s.first + j) * k;
          const double *col = lv + j * m;
          for (Eigen::Index i = j + 1; i < m; ++i) {
            const double v = col[i];
            double *xi = x + static_cast<Eigen::Index>(rows[i]) * k;
            for (Eigen::Index q = 0; q < k; ++q) xi[q] -= v * xj[q];
          }
        }
      }
    }

    // D Z = Y, with 1x1 and 2x2 blocks.
    for (Eigen::Index p = 0; p < n_; ++p) {
      double *xp = x + p * k;
      if (two_by_two_[p]) {
        double *xq = xp + k;
        const double a = dinv_[p], off = dinv_off_[p], d = dinv_[p + 1];
        for (Eigen::Index q = 0; q < k; ++q) {
          const double y1 = xp[q], y2 = xq[q];
          xp[q] = a * y1 + off * y2;
          xq[q] = off * y1 + d * y2;
        }
        ++p;
      } else {
        const double a = dinv_[p];
        for (Eigen::Index q = 0; q < k; ++q) xp[q] *= a;
      }
    }

    // L^T W = Z
    for (auto it = supers_.rbegin(); it != supers_.rend(); ++it) {
      const Super &s = *it;
      const double *lv = values_.data() + s.values;
      const int *rows = rows_.data() + s.rows;
      const Eigen::Index m = s.order, c = s.width, below = m - c;
      if (dense(s)) {
        Eigen::Map<const ColMatrix> blk(lv, m, c);
        Eigen::Map<RowMatrix> top(x + s.first * k, c, k);
        auto g = scratch.topRows(below);
        for (Eigen::Index i = 0; i < below; ++i) g.row(i) = X.row(rows[c + i]);
        top.noalias() -= blk.bottomRows(below).transpose() * g;
        blk.topRows(c).transpose().triangularView<Eigen::UnitUpper>().solveInPlace(top);
      } else {
        for (Eigen::Index j = c - 1; j >= 0; --j) {
          double *xj = x + (s.first + j) * k;
          const double *col = lv + j * m;
          for (Eigen::Index i = j + 1; i < m; ++i) {
            const double v = col[i];
            const double *xi = x + static_cast<Eigen::Index>(rows[i]) * k;
            for (Eigen::Index q = 0; q < k; ++q) xj[q] -= v * xi[q];
          }
        }
      }
    }

    // out = P^{-1} W: out[i] = W[perm[i]]
    for (Eigen::Index i = 0; i < n_; ++i) {
      std::copy_n(x + static_cast<Eigen::Index>(perm_[i]) * k, k, out + i * k);
    }
  }

 private:
  struct Super {
    Eigen::Index first, width, order;
    std::size_t rows, values;
  };

  // Below this block size the dense kernels' call overhead outweighs their speed.
  static constexpr Eigen::Index kDenseMinSize = 1024;
  static bool dense(const Super &s) { return s.width > 1 && s.order * s.width >= kDenseMinSize; }

  void build_blocks(const Sparse &L) {
    const int *lp = L.outerIndexPtr();
    const int *li = L.innerIndexPtr();
    const double *lx = L.valuePtr();
    // L comes from setFromTriplets: compressed, sorted rows, unit diagonal first in each column.
    Eigen::Index j = 0;
    while (j < n_) {
      Eigen::Index width = 1;
      while (j + width < n_) {
        const Eigen::Index prev = j + width - 1, next = j + width;
        const int prev_len = lp[prev + 1] - lp[prev];
        const int next_len = lp[next + 1] - lp[next];
        if (next_len != prev_len - 1 || prev_len < 2 || li[lp[prev] + 1] != next ||
            !std::equal(li + lp[prev] + 1, li + lp[prev + 1], li + lp[next])) {
          break;
        }
        ++width;
      }
      const Eigen::Index order = lp[j + 1] - lp[j];
      Super s{j, width, order, rows_.size(), values_.size()};
      rows_.insert(rows_.end(), li + lp[j], li + lp[j + 1]);
      values_.resize(values_.size() + static_cast<std::size_t>(order * width), 0.0);
      double *blk = values_.data() + s.values;
      for (Eigen::Index c = 0; c < width; ++c) {
        // Column j + c holds rows (j + c, ...) = the supernode's rows from index c on.
        std::copy(lx + lp[j + c], lx + lp[j + c + 1], blk + c * order + c);
      }
      max_below_ = std::max(max_below_, order - width);
      supers_.push_back(s);
      j += width;
    }
  }

  void build_dinv(const Vector &d, const Vector &sub) {
    dinv_.assign(static_cast<std::size_t>(n_), 0.0);
    dinv_off_.assign(static_cast<std::size_t>(n_), 0.0);
    two_by_two_.assign(static_cast<std::size_t>(n_), 0);
    for (Eigen::Index p = 0; p < n_; ++p) {
      if (p + 1 < n_ && sub[p] != 0.0) {
        // Inverse of [[d1, b], [b, d2]], scaled so that the determinant doesn't overflow.
        const double d1 = d[p], d2 = d[p + 1], b = sub[p];
        const double scale = std::max({std::abs(d1), std::abs(d2), std::abs(b)});
        const double a = d1 / scale, c = d2 / scale, bs = b / scale;
        const double det = a * c - bs * bs;
        dinv_[p] = (c / det) / scale;
        dinv_[p + 1] = (a / det) / scale;
        dinv_off_[p] = (-bs / det) / scale;
        two_by_two_[p] = 1;
        ++p;
      } else {
        dinv_[p] = d[p] != 0.0 ? 1.0 / d[p] : 0.0;  // zero pivot: minimum-norm solution, as Eigen
      }
    }
  }

  Eigen::Index n_;
  Eigen::Index max_below_ = 0;
  std::vector<int> perm_;
  std::vector<Super> supers_;
  std::vector<int> rows_;
  std::vector<double> values_;
  std::vector<double> dinv_, dinv_off_;
  std::vector<char> two_by_two_;
};

// Eigen MultifrontalLDLT: dense frontal matrices with level-3 updates and Bunch-Kaufman-style
// 1x1 / 2x2 pivoting, so it also handles symmetric indefinite matrices stably.
//
// It has no setShift(), so the shift is applied to a copy: A + shift * I with an explicit
// diagonal. The identity contributes structural diagonal entries even when shift == 0, so the
// analyzed pattern is the same for every shift and refactorize() can change it freely.
template <typename Ordering>
class MultifrontalBackend final : public CholeskyBackend {
 public:
  using Solver = Eigen::MultifrontalLDLT<Sparse, Eigen::Lower, Ordering>;

  void analyze(const Sparse &a) override {
    identity_.resize(a.rows(), a.cols());
    identity_.setIdentity();
    solver_.analyzePattern(shifted(a, 0.0));
  }

  void factorize(const Sparse &a, double shift) override {
    std::lock_guard<std::mutex> lock(blocked_mutex_);
    blocked_.reset();
    solver_.factorize(shifted(a, shift));
  }

  Eigen::ComputationInfo info() const override { return solver_.info(); }

  // One right-hand side: Eigen's solve (vector kernels on large fronts, no extra memory).
  // Several: the supernodal blocked solve, built from the factor on first use and then kept
  // until the next factorize(). Building it costs about as much as one many-RHS solve and holds
  // a second copy of L.
  void solve_into(const Eigen::Ref<const RowMatrix> &b, double *out) const override {
    if (b.cols() == 1) {
      const Vector x = solver_.solve(Vector(b.col(0)));
      std::copy_n(x.data(), x.size(), out);
      return;
    }
    std::shared_ptr<const SupernodalLDLT> blocked;
    {
      std::lock_guard<std::mutex> lock(blocked_mutex_);
      if (!blocked_) blocked_ = std::make_shared<const SupernodalLDLT>(solver_);
      blocked = blocked_;
    }
    blocked->solve_into(b, out);
  }

  std::pair<double, double> slogdet() const override {
    const double sign = solver_.signDeterminant();
    if (sign == 0.0) return {0.0, -std::numeric_limits<double>::infinity()};
    return {sign, solver_.logAbsDeterminant()};
  }

  Eigen::Index factor_nnz() const override { return solver_.nonZeros(); }

  std::array<Eigen::Index, 3> inertia() const override {
    return {solver_.numPositive(), solver_.numNegative(), solver_.numZero()};
  }

 private:
  Sparse shifted(const Sparse &a, double shift) const {
    Sparse out = a + shift * identity_;
    out.makeCompressed();
    return out;
  }

  Solver solver_;
  Sparse identity_;
  mutable std::mutex blocked_mutex_;
  mutable std::shared_ptr<const SupernodalLDLT> blocked_;
};

static std::unique_ptr<CholeskyBackend> make_cholesky_backend(const std::string &mode, const std::string &ordering) {
  using Amd = Eigen::AMDOrdering<int>;
  using Natural = Eigen::NaturalOrdering<int>;
  if (ordering != "amd" && ordering != "natural") {
    throw py::value_error("ordering must be one of: amd, natural");
  }
  const bool amd = ordering == "amd";
  if (mode == "ldlt") {
    if (amd) return std::make_unique<CholeskyBackendImpl<Eigen::SimplicialLDLT<Sparse, Eigen::Lower, Amd>, true>>();
    return std::make_unique<CholeskyBackendImpl<Eigen::SimplicialLDLT<Sparse, Eigen::Lower, Natural>, true>>();
  }
  if (mode == "llt") {
    if (amd) return std::make_unique<CholeskyBackendImpl<Eigen::SimplicialLLT<Sparse, Eigen::Lower, Amd>, false>>();
    return std::make_unique<CholeskyBackendImpl<Eigen::SimplicialLLT<Sparse, Eigen::Lower, Natural>, false>>();
  }
  if (mode == "multifrontal") {
    if (amd) return std::make_unique<MultifrontalBackend<Amd>>();
    return std::make_unique<MultifrontalBackend<Natural>>();
  }
  throw py::value_error("mode must be one of: ldlt, llt, multifrontal");
}

// Reusable sparse Cholesky factorization of a symmetric matrix (lower triangle referenced).
//
// The symbolic analysis (ordering + elimination tree) is computed once; refactorize() reruns only
// the numeric phase for a matrix with an identical sparsity pattern. Eigen does not verify the
// pattern itself and a mismatch would write past the analyzed storage, so the pattern is stored
// and checked here.
class SparseCholesky {
 public:
  SparseCholesky(py::object a, const std::string &mode, const std::string &ordering, double shift)
      : backend_(make_cholesky_backend(mode, ordering)), mode_(mode), ordering_(ordering) {
    const auto view = map_sparse_csc(std::move(a));
    if (view->mat.rows() != view->mat.cols()) {
      throw py::value_error("cholesky requires a square sparse matrix");
    }
    n_ = view->mat.rows();
    outer_.assign(view->indptr.data(), view->indptr.data() + view->indptr.size());
    inner_.assign(view->indices.data(), view->indices.data() + view->mat.nonZeros());

    const Sparse mat = view->mat;
    backend_->analyze(mat);
    factorize_checked(mat, shift);
  }

  void refactorize(py::object a, std::optional<double> shift) {
    const auto view = map_sparse_csc(std::move(a));
    const auto &m = view->mat;
    const bool same_pattern =
        m.rows() == n_ && m.cols() == n_ && static_cast<std::size_t>(m.nonZeros()) == inner_.size() &&
        std::equal(outer_.begin(), outer_.end(), view->indptr.data()) &&
        std::equal(inner_.begin(), inner_.end(), view->indices.data());
    if (!same_pattern) {
      throw py::value_error(
          "refactorize requires the same sparsity pattern as the original matrix; "
          "create a new factorization with cholesky() instead");
    }
    const Sparse mat = m;
    factorize_checked(mat, shift.value_or(shift_));
  }

  py::array_t<double> solve(const py::array_t<double, py::array::forcecast> &b) const {
    std::unique_ptr<RowMatrix> owned_b;
    const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);
    if (rhs.rows() != n_) {
      throw py::value_error("factorized matrix and rhs shape mismatch");
    }
    py::array_t<double> out = make_output_array(rhs.rows(), rhs.cols());
    double *out_ptr = out.mutable_data();
    {
      py::gil_scoped_release release;
      backend_->solve_into(rhs, out_ptr);
    }
    return out;
  }

  py::tuple slogdet() const {
    const auto res = backend_->slogdet();
    return py::make_tuple(res.first, res.second);
  }

  Eigen::Index n() const { return n_; }
  Eigen::Index factor_nnz() const { return backend_->factor_nnz(); }
  py::tuple inertia() const {
    const auto in = backend_->inertia();
    return py::make_tuple(in[0], in[1], in[2]);
  }
  const std::string &mode() const { return mode_; }
  const std::string &ordering() const { return ordering_; }
  double shift() const { return shift_; }

 private:
  void factorize_checked(const Sparse &mat, double shift) {
    {
      py::gil_scoped_release release;
      backend_->factorize(mat, shift);
    }
    shift_ = shift;
    if (backend_->info() != Eigen::Success) {
      raise_linalg_error(mode_ == "llt"            ? "matrix is not positive definite (Cholesky LLT failed)"
                         : mode_ == "multifrontal" ? "matrix is singular or not finite (multifrontal LDLT failed)"
                                                   : "matrix is singular (Cholesky LDLT hit a zero pivot)");
    }
  }

  std::unique_ptr<CholeskyBackend> backend_;
  std::vector<int> outer_;
  std::vector<int> inner_;
  Eigen::Index n_ = 0;
  std::string mode_;
  std::string ordering_;
  double shift_ = 0.0;
};

static py::array_t<double> core_spmm(py::object a, const py::array_t<double, py::array::forcecast> &b) {
  const auto sparse = map_sparse_csc(a);
  std::unique_ptr<RowMatrix> owned_b;
  const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);

  if (sparse->mat.cols() != rhs.rows()) {
    throw py::value_error("spmm dimension mismatch");
  }

  return assign_to_output(sparse->mat * rhs);
}

static py::object core_spspmm(py::object a, py::object b) {
  const auto sparse_a = map_sparse_csc(a);
  const auto sparse_b = map_sparse_csc(b);

  if (sparse_a->mat.cols() != sparse_b->mat.rows()) {
    throw py::value_error("spspmm dimension mismatch");
  }

  Sparse out = (sparse_a->mat * sparse_b->mat).pruned();
  out.makeCompressed();
  return to_scipy_csc(out);
}

// ---------------------------------------------------------------------------------------------
// Iterative solvers (CG, BiCGSTAB)
//
// Eigen's iterative solvers stop when a recursively updated residual drops below the tolerance.
// That recursion can drift away from the true residual b - A x, badly so for BiCGSTAB on
// nonnormal (e.g. convection-dominated) problems, so "converged" can mean an answer that misses
// `tol` by orders of magnitude. After every solve we therefore recompute the true relative
// residual. If it misses the tolerance by more than a small margin, we restart from the current
// iterate (which resets the recursion) while iteration budget remains, and raise if it still
// misses.

struct IterativeStats {
  int iterations = 0;     // total over the initial solve and any restarts
  double error = 0.0;     // the solver's own (recursive) relative residual estimate
  double residual = 0.0;  // true relative residual ||b - A x|| / ||b||
};

static std::string format_sci(double value) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(2) << value;
  return os.str();
}

static double true_relative_residual(const Sparse &mat, const Eigen::VectorXd &b,
                                     const Eigen::VectorXd &x) {
  const double bnorm = b.norm();
  const double rnorm = (b - mat * x).norm();
  return bnorm > 0.0 ? rnorm / bnorm : rnorm;
}

// The true residual may exceed `tol` by this factor before we treat it as a failure; the floor
// keeps rounding error from tripping the check when `tol` is near machine precision.
static double residual_limit(double tol) {
  return std::max(10.0 * tol, 100.0 * std::numeric_limits<double>::epsilon());
}

constexpr int kMaxResidualRestarts = 5;

template <typename Solver>
static IterativeStats iterative_solve_checked(Solver &solver,
                                              const Sparse &mat,
                                              const Eigen::VectorXd &b,
                                              Eigen::VectorXd &x,
                                              double tol,
                                              int maxiter,
                                              const std::string &name,
                                              const std::string &hint) {
  IterativeStats st;
  solver.setMaxIterations(maxiter);
  x = solver.solve(b);
  st.iterations = static_cast<int>(solver.iterations());
  st.error = solver.error();

  const auto throw_not_converged = [&]() {
    throw std::runtime_error(name + " did not converge (iters=" + std::to_string(st.iterations) +
                             ", error=" + format_sci(st.error) + ")" + hint);
  };
  if (solver.info() != Eigen::Success) {
    throw_not_converged();
  }

  st.residual = true_relative_residual(mat, b, x);
  const double limit = residual_limit(tol);
  int restarts = 0;
  while (st.residual > limit && restarts < kMaxResidualRestarts && st.iterations < maxiter) {
    ++restarts;
    solver.setMaxIterations(maxiter - st.iterations);
    const Eigen::VectorXd guess = x;
    x = solver.solveWithGuess(b, guess);
    st.iterations += static_cast<int>(solver.iterations());
    st.error = solver.error();
    if (solver.info() != Eigen::Success) {
      throw_not_converged();
    }
    st.residual = true_relative_residual(mat, b, x);
  }

  if (st.residual > limit) {
    throw std::runtime_error(
        name + " did not converge: its residual estimate reached tol, but the true relative "
               "residual ||b - Ax||/||b|| = " +
        format_sci(st.residual) + " exceeds tol=" + format_sci(tol) +
        " (iters=" + std::to_string(st.iterations) + ", restarts=" + std::to_string(restarts) +
        ")" + hint);
  }
  return st;
}

// Solves every column of `rhs` (writing into `out` when given) and returns the last column's stats.
template <typename Solver>
static IterativeStats run_iterative(Solver &solver,
                                    const Sparse &mat,
                                    const Eigen::Ref<const RowMatrix> &rhs,
                                    Eigen::Map<RowMatrix> *out,
                                    double tol,
                                    int maxiter,
                                    const std::string &name,
                                    const std::string &hint = "") {
  solver.setTolerance(tol);
  solver.setMaxIterations(maxiter);
  solver.compute(mat);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(name + " setup failed" + hint);
  }
  IterativeStats st;
  Eigen::VectorXd b(mat.rows());
  Eigen::VectorXd x(mat.rows());
  for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
    b = rhs.col(col);
    st = iterative_solve_checked(solver, mat, b, x, tol, maxiter, name, hint);
    if (out != nullptr) {
      out->col(col) = x;
    }
  }
  return st;
}

static IterativeStats run_conjugate_gradient(const Sparse &mat,
                                             const Eigen::Ref<const RowMatrix> &rhs,
                                             Eigen::Map<RowMatrix> *out,
                                             const std::string &preconditioner,
                                             double tol,
                                             int maxiter,
                                             int ilu_fill_factor,
                                             double ilu_drop_tol) {
  constexpr int UpLo = Eigen::Lower | Eigen::Upper;
  const std::string name = "ConjugateGradient";
  if (preconditioner == "none") {
    Eigen::ConjugateGradient<Sparse, UpLo, Eigen::IdentityPreconditioner> cg;
    return run_iterative(cg, mat, rhs, out, tol, maxiter, name);
  }
  if (preconditioner == "jacobi") {
    Eigen::ConjugateGradient<Sparse, UpLo, Eigen::DiagonalPreconditioner<double> > cg;
    return run_iterative(cg, mat, rhs, out, tol, maxiter, name);
  }
  if (preconditioner == "ichol") {
    // Symmetric preconditioner, as CG requires. Eigen's IncompleteCholesky (zero fill beyond the
    // pattern of A) reads the lower triangle, applies diagonal scaling, and retries with a growing
    // diagonal shift if the factorization breaks down. We keep the matrix's own ordering: a
    // fill-reducing ordering such as AMD weakens zero-fill IC badly (on a 100x100 5-point
    // Laplacian, 154 CG iterations with AMD vs 77 in natural order, against 186 unpreconditioned).
    Eigen::ConjugateGradient<
        Sparse, UpLo, Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int> > >
        cg;
    return run_iterative(cg, mat, rhs, out, tol, maxiter, name);
  }
  if (preconditioner == "ilu") {
    if (ilu_fill_factor <= 0) {
      throw py::value_error("ilu_fill_factor must be positive when preconditioner='ilu'");
    }
    if (ilu_drop_tol < 0.0) {
      throw py::value_error("ilu_drop_tol must be non-negative when preconditioner='ilu'");
    }
    // Deprecated for CG: IncompleteLUT is not symmetric, which breaks CG's assumptions.
    Eigen::ConjugateGradient<Sparse, UpLo, Eigen::IncompleteLUT<double> > cg;
    cg.preconditioner().setFillfactor(ilu_fill_factor);
    cg.preconditioner().setDroptol(ilu_drop_tol);
    return run_iterative(cg, mat, rhs, out, tol, maxiter, name,
                         "; preconditioner='ilu' (IncompleteLUT) is not symmetric, use "
                         "preconditioner='ichol' with method='cg'");
  }
  throw py::value_error("preconditioner must be one of: none, jacobi, ichol, ilu");
}

static py::array_t<double> core_sparse_solve(py::object a,
                                              const py::array_t<double, py::array::forcecast> &b,
                                              const std::string &method,
                                              double tol,
                                              int maxiter,
                                              const std::string &preconditioner,
                                              int ilu_fill_factor,
                                              double ilu_drop_tol) {
  const auto sparse = map_sparse_csc(a);
  std::unique_ptr<RowMatrix> owned_b;
  const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);

  if (sparse->mat.rows() != sparse->mat.cols()) {
    throw py::value_error("sparse solve requires square matrix");
  }
  if (sparse->mat.rows() != rhs.rows()) {
    throw py::value_error("sparse solve shape mismatch");
  }

  py::array_t<double> out_arr = make_output_array(rhs.rows(), rhs.cols());
  Eigen::Map<RowMatrix> out(out_arr.mutable_data(), rhs.rows(), rhs.cols());

  const int effective_maxiter = maxiter > 0 ? maxiter : static_cast<int>(sparse->mat.rows() * 2);
  const double effective_tol = tol > 0.0 ? tol : 1e-8;

  if (method != "cg" && preconditioner != "none") {
    throw py::value_error("preconditioner is only supported for method='cg'");
  }

  if (method == "auto" || method == "lu") {
    Eigen::SparseLU<Sparse> lu;
    lu.analyzePattern(sparse->mat);
    lu.factorize(sparse->mat);
    if (lu.info() != Eigen::Success) {
      throw std::runtime_error("SparseLU factorization failed");
    }
    for (int col = 0; col < rhs.cols(); ++col) {
      out.col(col) = lu.solve(rhs.col(col));
      if (lu.info() != Eigen::Success) {
        throw std::runtime_error("SparseLU solve failed");
      }
    }
    return out_arr;
  }

  if (method == "cg") {
    run_conjugate_gradient(sparse->mat, rhs, &out, preconditioner, effective_tol, effective_maxiter,
                           ilu_fill_factor, ilu_drop_tol);
    return out_arr;
  }

  if (method == "bicgstab") {
    Eigen::BiCGSTAB<Sparse> solver;
    run_iterative(solver, sparse->mat, rhs, &out, effective_tol, effective_maxiter, "BiCGSTAB");
    return out_arr;
  }

  throw py::value_error("method must be one of: auto, lu, cg, bicgstab");
}

static py::dict core_sparse_solve_stats(py::object a,
                                        const py::array_t<double, py::array::forcecast> &b,
                                        const std::string &method,
                                        double tol,
                                        int maxiter,
                                        const std::string &preconditioner,
                                        int ilu_fill_factor,
                                        double ilu_drop_tol) {
  if (method != "cg") {
    throw py::value_error("solve_stats is only supported for method='cg'");
  }

  const auto sparse = map_sparse_csc(a);
  std::unique_ptr<RowMatrix> owned_b;
  const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);

  if (sparse->mat.rows() != sparse->mat.cols()) {
    throw py::value_error("sparse solve requires square matrix");
  }
  if (rhs.rows() != sparse->mat.rows()) {
    throw py::value_error("sparse solve shape mismatch");
  }
  if (rhs.cols() != 1) {
    throw py::value_error("solve_stats requires a single RHS column");
  }

  const int effective_maxiter = maxiter > 0 ? maxiter : static_cast<int>(sparse->mat.rows() * 2);
  const double effective_tol = tol > 0.0 ? tol : 1e-8;

  const IterativeStats stats = run_conjugate_gradient(sparse->mat, rhs, nullptr, preconditioner,
                                                      effective_tol, effective_maxiter,
                                                      ilu_fill_factor, ilu_drop_tol);

  py::dict out;
  out["iterations"] = stats.iterations;
  out["error"] = stats.error;
  out["residual"] = stats.residual;
  return out;
}

static std::shared_ptr<SparseFactorized> core_sparse_factorize(py::object a) {
  const auto sparse = map_sparse_csc(a);
  if (sparse->mat.rows() != sparse->mat.cols()) {
    throw py::value_error("factorize requires square sparse matrix");
  }
  return std::make_shared<SparseFactorized>(sparse->mat);
}

static py::dict core_build_config() {
  py::dict cfg;
  cfg["version"] = PEIGEN_PROJECT_VERSION;
  cfg["build_type"] = PEIGEN_BUILD_TYPE;
#ifdef EIGEN_USE_BLAS
  cfg["blas_enabled"] = true;
#else
  cfg["blas_enabled"] = false;
#endif
  cfg["blas_backend"] = PEIGEN_BLAS_BACKEND;
#if defined(PEIGEN_LAPACK_ENABLED)
  cfg["lapack_enabled"] = true;
#else
  cfg["lapack_enabled"] = false;
#endif
  cfg["lapack_backend"] = PEIGEN_LAPACK_BACKEND;
#if PEIGEN_OPENMP_ENABLED
  cfg["openmp_enabled"] = true;
#else
  cfg["openmp_enabled"] = false;
#endif
#ifdef EIGEN_VECTORIZE
  cfg["vectorization_enabled"] = true;
#else
  cfg["vectorization_enabled"] = false;
#endif
  cfg["eigen_mpl2_only"] = true;
#ifdef EIGEN_VERSION_STRING
  cfg["eigen_version"] = EIGEN_VERSION_STRING;
#else
  cfg["eigen_version"] = std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
                         std::to_string(EIGEN_MINOR_VERSION);
#endif
  return cfg;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "pEigen core extension";

  py::class_<SparseFactorized, std::shared_ptr<SparseFactorized>>(m, "SparseFactorized")
      .def("solve", &SparseFactorized::solve, py::arg("b"))
      .def_property_readonly("shape", [](const SparseFactorized &self) {
        return py::make_tuple(self.rows(), self.cols());
      });

  m.def("matmul", &core_matmul, py::arg("a"), py::arg("b"));
  m.def("solve", &core_solve, py::arg("a"), py::arg("b"), py::arg("method") = "auto");
  m.def("qr", &core_qr, py::arg("a"), py::arg("mode") = "reduced");
  m.def("svd", &core_svd, py::arg("a"), py::arg("full_matrices") = false, py::arg("method") = "auto");
  m.def("svd_compute", &core_svd_compute, py::arg("a"), py::arg("full_matrices") = false,
        py::arg("method") = "auto");
  m.def("eigh", &core_eigh, py::arg("a"), py::arg("lower") = true, py::arg("eigenvectors") = true,
        py::arg("method") = "auto");
  m.def("eigh_compute", &core_eigh_compute, py::arg("a"), py::arg("lower") = true, py::arg("method") = "auto");
  m.def("norm", &core_norm, py::arg("a"));

  m.def("spmm", &core_spmm, py::arg("a"), py::arg("b"));
  m.def("spspmm", &core_spspmm, py::arg("a"), py::arg("b"));
  m.def("sparse_solve", &core_sparse_solve,
        py::arg("a"),
        py::arg("b"),
        py::arg("method") = "auto",
        py::arg("tol") = 1e-8,
        py::arg("maxiter") = 0,
        py::arg("preconditioner") = "none",
        py::arg("ilu_fill_factor") = 10,
        py::arg("ilu_drop_tol") = 1e-4);
  m.def("sparse_solve_stats", &core_sparse_solve_stats,
        py::arg("a"),
        py::arg("b"),
        py::arg("method") = "cg",
        py::arg("tol") = 1e-8,
        py::arg("maxiter") = 0,
        py::arg("preconditioner") = "none",
        py::arg("ilu_fill_factor") = 10,
        py::arg("ilu_drop_tol") = 1e-4);
  m.def("sparse_factorize", &core_sparse_factorize, py::arg("a"));

  py::class_<SparseCholesky>(m, "SparseCholesky")
      .def(py::init<py::object, const std::string &, const std::string &, double>(), py::arg("a"),
           py::arg("mode") = "ldlt", py::arg("ordering") = "amd", py::arg("shift") = 0.0)
      .def("refactorize", &SparseCholesky::refactorize, py::arg("a"), py::arg("shift") = py::none())
      .def("solve", &SparseCholesky::solve, py::arg("b"))
      .def("slogdet", &SparseCholesky::slogdet)
      .def_property_readonly("shape", [](const SparseCholesky &self) { return py::make_tuple(self.n(), self.n()); })
      .def_property_readonly("factor_nnz", &SparseCholesky::factor_nnz)
      .def_property_readonly("inertia", &SparseCholesky::inertia)
      .def_property_readonly("mode", &SparseCholesky::mode)
      .def_property_readonly("ordering", &SparseCholesky::ordering)
      .def_property_readonly("shift", &SparseCholesky::shift);
  m.def("build_config", &core_build_config);
}
