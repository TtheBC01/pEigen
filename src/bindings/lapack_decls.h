// Fortran LAPACK prototypes for the routines pEigen calls directly.
//
// Declared here instead of using Eigen's bundled lapack/lapack.h, which is an Eigen-internal
// header: it changes between Eigen versions (master dropped dgesv and added an ILP64 mode whose
// 64-bit integers would silently mismatch the LP64 Accelerate/OpenBLAS libraries we link).
//
// Conventions: LP64 (32-bit int), trailing-underscore symbol names, and the hidden
// character-length arguments that gfortran-compiled LAPACK (e.g. OpenBLAS's reference LAPACK)
// expects after the explicit arguments. Libraries that do not use them (Accelerate) ignore the
// extra trailing arguments, which is ABI-safe on all supported platforms.
#pragma once

#include <cstddef>

using lapack_int = int;
using fortran_strlen = std::size_t;

extern "C" {

void dgesv_(const lapack_int *n, const lapack_int *nrhs, double *a, const lapack_int *lda, lapack_int *ipiv,
            double *b, const lapack_int *ldb, lapack_int *info);

void dgesdd_(const char *jobz, const lapack_int *m, const lapack_int *n, double *a, const lapack_int *lda,
             double *s, double *u, const lapack_int *ldu, double *vt, const lapack_int *ldvt, double *work,
             const lapack_int *lwork, lapack_int *iwork, lapack_int *info, fortran_strlen jobz_len);

void dsyev_(const char *jobz, const char *uplo, const lapack_int *n, double *a, const lapack_int *lda, double *w,
            double *work, const lapack_int *lwork, lapack_int *info, fortran_strlen jobz_len,
            fortran_strlen uplo_len);

void dsyevd_(const char *jobz, const char *uplo, const lapack_int *n, double *a, const lapack_int *lda, double *w,
             double *work, const lapack_int *lwork, lapack_int *iwork, const lapack_int *liwork, lapack_int *info,
             fortran_strlen jobz_len, fortran_strlen uplo_len);

void dsyevr_(const char *jobz, const char *range, const char *uplo, const lapack_int *n, double *a,
             const lapack_int *lda, const double *vl, const double *vu, const lapack_int *il, const lapack_int *iu,
             const double *abstol, lapack_int *m, double *w, double *z, const lapack_int *ldz, lapack_int *isuppz,
             double *work, const lapack_int *lwork, lapack_int *iwork, const lapack_int *liwork, lapack_int *info,
             fortran_strlen jobz_len, fortran_strlen range_len, fortran_strlen uplo_len);

}  // extern "C"
