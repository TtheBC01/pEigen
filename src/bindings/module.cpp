#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
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
#include <lapack/lapack.h>

using lapack_int = int;

// Not declared in Eigen's bundled lapack.h (used for divide-and-conquer / MRRR paths).
extern "C" {
EIGEN_LAPACK_API void BLASFUNC(dsyevd)(const char *, const char *, int *, double *, int *, double *, double *, int *,
                                       int *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dsyevr)(const char *, const char *, const char *, int *, double *, int *,
                                       double *, double *, int *, int *, double *, int *, double *, double *, int *,
                                       int *, double *, int *, int *, int *, int *);
}
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

  BLASFUNC(dgesdd)(&jobz, &m, &n, matrix.data(), &lda, out.s.data(), out.u.data(), &ldu, vt.data(), &ldvt,
                   &work_query, &lwork, iwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("LAPACK dgesdd workspace query failed with info=" + std::to_string(info));
  }

  lwork = static_cast<lapack_int>(work_query);
  std::vector<double> work(static_cast<std::size_t>(lwork));

  BLASFUNC(dgesdd)(&jobz, &m, &n, matrix.data(), &lda, out.s.data(), out.u.data(), &ldu, vt.data(), &ldvt,
                   work.data(), &lwork, iwork.data(), &info);
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

  BLASFUNC(dsyev)(&jobz, &uplo, &n, matrix.data(), &lda, w.data(), &work_query, &lwork, &info);
  if (info != 0) {
    throw std::runtime_error("LAPACK dsyev workspace query failed with info=" + std::to_string(info));
  }

  lwork = static_cast<lapack_int>(work_query);
  std::vector<double> work(static_cast<std::size_t>(lwork));

  BLASFUNC(dsyev)(&jobz, &uplo, &n, matrix.data(), &lda, w.data(), work.data(), &lwork, &info);
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

  BLASFUNC(dsyevr)(&jobz, &range, &uplo, &n, matrix.data(), &lda, &vl, &vu, &il, &iu, &abstol, &m, w.data(),
                   nullptr, &ldz, nullptr, &work_query, &lwork, &iwork_query, &liwork, &info);
  if (info != 0) {
    return false;
  }

  lwork = static_cast<lapack_int>(work_query);
  liwork = iwork_query;
  std::vector<double> work(static_cast<std::size_t>(lwork));
  std::vector<lapack_int> iwork(static_cast<std::size_t>(liwork));

  BLASFUNC(dsyevr)(&jobz, &range, &uplo, &n, matrix.data(), &lda, &vl, &vu, &il, &iu, &abstol, &m, w.data(),
                   nullptr, &ldz, nullptr, work.data(), &lwork, iwork.data(), &liwork, &info);
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

  BLASFUNC(dsyevd)(&jobz, &uplo, &n, matrix.data(), &lda, out.w.data(), &work_query, &lwork, &iwork_query,
                   &liwork, &info);
  if (info != 0) {
    throw std::runtime_error("LAPACK dsyevd workspace query failed with info=" + std::to_string(info));
  }

  lwork = static_cast<lapack_int>(work_query);
  liwork = iwork_query;
  std::vector<double> work(static_cast<std::size_t>(lwork));
  std::vector<lapack_int> iwork(static_cast<std::size_t>(liwork));

  BLASFUNC(dsyevd)(&jobz, &uplo, &n, matrix.data(), &lda, out.w.data(), work.data(), &lwork, iwork.data(), &liwork,
                   &info);
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

struct SparseCscView {
  py::object owner;
  Eigen::Map<const Sparse> mat;
};

static SparseCscView map_sparse_csc(py::object matrix_obj, bool allow_csr = false) {
  py::module_ scipy_sparse = py::module_::import("scipy.sparse");
  py::object csc_type = scipy_sparse.attr("csc_matrix");
  py::object csr_type = scipy_sparse.attr("csr_matrix");

  py::object csc_obj;
  if (py::isinstance(matrix_obj, csc_type)) {
    csc_obj = matrix_obj;
  } else if (allow_csr && py::isinstance(matrix_obj, csr_type)) {
    csc_obj = matrix_obj.attr("tocsc")();
  } else {
    csc_obj = csc_type(matrix_obj);
  }

  const auto shape = csc_obj.attr("shape").cast<py::tuple>();
  const int rows = shape[0].cast<int>();
  const int cols = shape[1].cast<int>();

  auto indptr = csc_obj.attr("indptr").cast<py::array_t<int, py::array::forcecast>>();
  auto indices = csc_obj.attr("indices").cast<py::array_t<int, py::array::forcecast>>();
  auto data = csc_obj.attr("data").cast<py::array_t<double, py::array::forcecast>>();

  return SparseCscView{
      std::move(csc_obj),
      Eigen::Map<const Sparse>(rows, cols, static_cast<int>(data.size()), indptr.data(),
                               indices.data(), data.data()),
  };
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

  BLASFUNC(dgesv)(&n, &nrhs, lhs.data(), &lda, ipiv.data(), rhs.data(), &ldb, &info);
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
  if (lu.matrixLU().diagonal().cwiseAbs().minCoeff() < 1e-15) {
    throw py::value_error("matrix is singular or ill-conditioned");
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
    }
    return out_arr;
  }

 private:
  std::unique_ptr<Eigen::SparseLU<Sparse>> lu_;
};

static py::array_t<double> core_spmm(py::object a, const py::array_t<double, py::array::forcecast> &b) {
  SparseCscView sparse = map_sparse_csc(a, true);
  std::unique_ptr<RowMatrix> owned_b;
  const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);

  if (sparse.mat.cols() != rhs.rows()) {
    throw py::value_error("spmm dimension mismatch");
  }

  return assign_to_output(sparse.mat * rhs);
}

static py::object core_spspmm(py::object a, py::object b) {
  SparseCscView sparse_a = map_sparse_csc(a, true);
  SparseCscView sparse_b = map_sparse_csc(b, true);

  if (sparse_a.mat.cols() != sparse_b.mat.rows()) {
    throw py::value_error("spspmm dimension mismatch");
  }

  Sparse out = (sparse_a.mat * sparse_b.mat).pruned();
  out.makeCompressed();
  return to_scipy_csc(out);
}

template <typename Preconditioner>
static void run_conjugate_gradient(const Sparse &mat,
                                   const Eigen::Ref<const RowMatrix> &rhs,
                                   Eigen::Map<RowMatrix> &out,
                                   double effective_tol,
                                   int effective_maxiter) {
  Eigen::ConjugateGradient<Sparse, Eigen::Lower | Eigen::Upper, Preconditioner> cg;
  cg.setTolerance(effective_tol);
  cg.setMaxIterations(effective_maxiter);
  cg.compute(mat);
  if (cg.info() != Eigen::Success) {
    throw std::runtime_error("ConjugateGradient setup failed");
  }
  for (int col = 0; col < rhs.cols(); ++col) {
    out.col(col) = cg.solve(rhs.col(col));
    if (cg.info() != Eigen::Success) {
      throw std::runtime_error(
          "ConjugateGradient did not converge (iters=" + std::to_string(cg.iterations()) +
          ", error=" + std::to_string(cg.error()) + ")");
    }
  }
}

template <typename Preconditioner>
static std::pair<int, double> conjugate_gradient_stats(const Sparse &mat,
                                                       const Eigen::VectorXd &rhs,
                                                       double effective_tol,
                                                       int effective_maxiter) {
  Eigen::ConjugateGradient<Sparse, Eigen::Lower | Eigen::Upper, Preconditioner> cg;
  cg.setTolerance(effective_tol);
  cg.setMaxIterations(effective_maxiter);
  cg.compute(mat);
  if (cg.info() != Eigen::Success) {
    throw std::runtime_error("ConjugateGradient setup failed");
  }
  Eigen::VectorXd x = cg.solve(rhs);
  if (cg.info() != Eigen::Success) {
    throw std::runtime_error(
        "ConjugateGradient did not converge (iters=" + std::to_string(cg.iterations()) +
        ", error=" + std::to_string(cg.error()) + ")");
  }
  (void)x;
  return {static_cast<int>(cg.iterations()), cg.error()};
}

static void run_conjugate_gradient_ilu(const Sparse &mat,
                                       const Eigen::Ref<const RowMatrix> &rhs,
                                       Eigen::Map<RowMatrix> &out,
                                       double effective_tol,
                                       int effective_maxiter,
                                       int ilu_fill_factor,
                                       double ilu_drop_tol) {
  Eigen::ConjugateGradient<Sparse, Eigen::Lower | Eigen::Upper, Eigen::IncompleteLUT<double> > cg;
  cg.preconditioner().setFillfactor(ilu_fill_factor);
  cg.preconditioner().setDroptol(ilu_drop_tol);
  cg.setTolerance(effective_tol);
  cg.setMaxIterations(effective_maxiter);
  cg.compute(mat);
  if (cg.info() != Eigen::Success) {
    throw std::runtime_error("ConjugateGradient setup failed");
  }
  for (int col = 0; col < rhs.cols(); ++col) {
    out.col(col) = cg.solve(rhs.col(col));
    if (cg.info() != Eigen::Success) {
      throw std::runtime_error(
          "ConjugateGradient did not converge (iters=" + std::to_string(cg.iterations()) +
          ", error=" + std::to_string(cg.error()) + ")");
    }
  }
}

static std::pair<int, double> conjugate_gradient_ilu_stats(const Sparse &mat,
                                                           const Eigen::VectorXd &rhs,
                                                           double effective_tol,
                                                           int effective_maxiter,
                                                           int ilu_fill_factor,
                                                           double ilu_drop_tol) {
  Eigen::ConjugateGradient<Sparse, Eigen::Lower | Eigen::Upper, Eigen::IncompleteLUT<double> > cg;
  cg.preconditioner().setFillfactor(ilu_fill_factor);
  cg.preconditioner().setDroptol(ilu_drop_tol);
  cg.setTolerance(effective_tol);
  cg.setMaxIterations(effective_maxiter);
  cg.compute(mat);
  if (cg.info() != Eigen::Success) {
    throw std::runtime_error("ConjugateGradient setup failed");
  }
  Eigen::VectorXd x = cg.solve(rhs);
  if (cg.info() != Eigen::Success) {
    throw std::runtime_error(
        "ConjugateGradient did not converge (iters=" + std::to_string(cg.iterations()) +
        ", error=" + std::to_string(cg.error()) + ")");
  }
  (void)x;
  return {static_cast<int>(cg.iterations()), cg.error()};
}

static void dispatch_conjugate_gradient(const Sparse &mat,
                                        const Eigen::Ref<const RowMatrix> &rhs,
                                        Eigen::Map<RowMatrix> &out,
                                        const std::string &preconditioner,
                                        double effective_tol,
                                        int effective_maxiter,
                                        int ilu_fill_factor,
                                        double ilu_drop_tol) {
  if (preconditioner == "none") {
    run_conjugate_gradient<Eigen::IdentityPreconditioner>(mat, rhs, out, effective_tol,
                                                          effective_maxiter);
  } else if (preconditioner == "jacobi") {
    run_conjugate_gradient<Eigen::DiagonalPreconditioner<double> >(mat, rhs, out, effective_tol,
                                                                   effective_maxiter);
  } else if (preconditioner == "ilu") {
    if (ilu_fill_factor <= 0) {
      throw py::value_error("ilu_fill_factor must be positive when preconditioner='ilu'");
    }
    if (ilu_drop_tol < 0.0) {
      throw py::value_error("ilu_drop_tol must be non-negative when preconditioner='ilu'");
    }
    run_conjugate_gradient_ilu(mat, rhs, out, effective_tol, effective_maxiter, ilu_fill_factor,
                               ilu_drop_tol);
  } else {
    throw py::value_error("preconditioner must be one of: none, jacobi, ilu");
  }
}

static std::pair<int, double> dispatch_conjugate_gradient_stats(const Sparse &mat,
                                                                const Eigen::VectorXd &rhs,
                                                                const std::string &preconditioner,
                                                                double effective_tol,
                                                                int effective_maxiter,
                                                                int ilu_fill_factor,
                                                                double ilu_drop_tol) {
  if (preconditioner == "none") {
    return conjugate_gradient_stats<Eigen::IdentityPreconditioner>(mat, rhs, effective_tol,
                                                                 effective_maxiter);
  }
  if (preconditioner == "jacobi") {
    return conjugate_gradient_stats<Eigen::DiagonalPreconditioner<double> >(mat, rhs, effective_tol,
                                                                           effective_maxiter);
  }
  if (preconditioner == "ilu") {
    if (ilu_fill_factor <= 0) {
      throw py::value_error("ilu_fill_factor must be positive when preconditioner='ilu'");
    }
    if (ilu_drop_tol < 0.0) {
      throw py::value_error("ilu_drop_tol must be non-negative when preconditioner='ilu'");
    }
    return conjugate_gradient_ilu_stats(mat, rhs, effective_tol, effective_maxiter, ilu_fill_factor,
                                        ilu_drop_tol);
  }
  throw py::value_error("preconditioner must be one of: none, jacobi, ilu");
}

static py::array_t<double> core_sparse_solve(py::object a,
                                              const py::array_t<double, py::array::forcecast> &b,
                                              const std::string &method,
                                              double tol,
                                              int maxiter,
                                              const std::string &preconditioner,
                                              int ilu_fill_factor,
                                              double ilu_drop_tol) {
  SparseCscView sparse = map_sparse_csc(a, true);
  std::unique_ptr<RowMatrix> owned_b;
  const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);

  if (sparse.mat.rows() != sparse.mat.cols()) {
    throw py::value_error("sparse solve requires square matrix");
  }
  if (sparse.mat.rows() != rhs.rows()) {
    throw py::value_error("sparse solve shape mismatch");
  }

  py::array_t<double> out_arr = make_output_array(rhs.rows(), rhs.cols());
  Eigen::Map<RowMatrix> out(out_arr.mutable_data(), rhs.rows(), rhs.cols());

  const int effective_maxiter = maxiter > 0 ? maxiter : static_cast<int>(sparse.mat.rows() * 2);
  const double effective_tol = tol > 0.0 ? tol : 1e-8;

  if (method != "cg" && preconditioner != "none") {
    throw py::value_error("preconditioner is only supported for method='cg'");
  }

  if (method == "auto" || method == "lu") {
    Eigen::SparseLU<Sparse> lu;
    lu.analyzePattern(sparse.mat);
    lu.factorize(sparse.mat);
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
    dispatch_conjugate_gradient(sparse.mat, rhs, out, preconditioner, effective_tol, effective_maxiter,
                                ilu_fill_factor, ilu_drop_tol);
    return out_arr;
  }

  if (method == "bicgstab") {
    Eigen::BiCGSTAB<Sparse> solver;
    solver.setTolerance(effective_tol);
    solver.setMaxIterations(effective_maxiter);
    solver.compute(sparse.mat);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("BiCGSTAB setup failed");
    }
    for (int col = 0; col < rhs.cols(); ++col) {
      out.col(col) = solver.solve(rhs.col(col));
      if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "BiCGSTAB did not converge (iters=" + std::to_string(solver.iterations()) +
            ", error=" + std::to_string(solver.error()) + ")");
      }
    }
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

  SparseCscView sparse = map_sparse_csc(a, true);
  std::unique_ptr<RowMatrix> owned_b;
  const Eigen::Ref<const RowMatrix> rhs = dense_row_ref(b, "b", owned_b);

  if (sparse.mat.rows() != sparse.mat.cols()) {
    throw py::value_error("sparse solve requires square matrix");
  }
  if (rhs.rows() != sparse.mat.rows()) {
    throw py::value_error("sparse solve shape mismatch");
  }
  if (rhs.cols() != 1) {
    throw py::value_error("solve_stats requires a single RHS column");
  }

  const int effective_maxiter = maxiter > 0 ? maxiter : static_cast<int>(sparse.mat.rows() * 2);
  const double effective_tol = tol > 0.0 ? tol : 1e-8;

  const auto stats = dispatch_conjugate_gradient_stats(
      sparse.mat, rhs.col(0), preconditioner, effective_tol, effective_maxiter, ilu_fill_factor,
      ilu_drop_tol);

  py::dict out;
  out["iterations"] = stats.first;
  out["error"] = stats.second;
  return out;
}

static std::shared_ptr<SparseFactorized> core_sparse_factorize(py::object a) {
  SparseCscView sparse = map_sparse_csc(a, true);
  if (sparse.mat.rows() != sparse.mat.cols()) {
    throw py::value_error("factorize requires square sparse matrix");
  }
  return std::make_shared<SparseFactorized>(sparse.mat);
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
  return cfg;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "pEigen core extension";

  py::class_<SparseFactorized, std::shared_ptr<SparseFactorized>>(m, "SparseFactorized")
      .def("solve", &SparseFactorized::solve, py::arg("b"));

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
  m.def("build_config", &core_build_config);
}
