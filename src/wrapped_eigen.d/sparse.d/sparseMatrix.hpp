#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <fstream>
#include <string>

#include <boost/python.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/complex.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

// Forward declaration for sparse/dense operations
template <class scalar>
class denseMatrix;

template <class scalar>
class sparseMatrix
{
public:
  sparseMatrix();

  sparseMatrix(int rows, int cols);

  sparseMatrix(std::vector<scalar> &data, std::vector<int> &outer_, std::vector<int> &inner_, int rows, int cols);

  sparseMatrix(boost::python::list &dataList, boost::python::list &outerList, boost::python::list &innerList, int rows, int cols);

  void resize(int rows, int cols);

  void clear();

  void save(std::string fname);
  void load(std::string fname);

  boost::python::list dataToList();
  boost::python::list outerToList();
  boost::python::list innerToList();

  std::string repr() {return "Sparse Matrix: " + std::to_string(rows()) + " rows, " + std::to_string(cols()) + " cols, " + std::to_string(nnz()) + " nnz";}

  int nnz() { return nnz_; }
  int nnz() const { return nnz_; }

  int rows() { return rows_; }
  int rows() const { return rows_; }
  int getRows() { return transpose_mat ? cols_ : rows_; }
  int getRows() const { return transpose_mat ? cols_ : rows_; }

  int cols() { return cols_; }
  int cols() const { return cols_; }
  int getCols() { return transpose_mat ? rows_ : cols_; }
  int getCols() const { return transpose_mat ? rows_ : cols_; }

  sparseMatrix getCol(int col);

  bool isTranspose() { return transpose_mat; }
  bool isTranspose() const { return transpose_mat; }

  std::vector<scalar> &dataContainer() { return data_; }
  std::vector<scalar> dataContainer() const { return data_; }

  std::vector<int> &innerContainer() { return inner_; }
  std::vector<int> innerContainer() const { return inner_; }

  std::vector<int> &outerContainer() { return outer_; }
  std::vector<int> outerContainer() const { return outer_; }

  Eigen::Map<Eigen::SparseMatrix<scalar>> getEigenMap();
  Eigen::Map<const Eigen::SparseMatrix<scalar>> getEigenMap() const;

  scalar *data() { return data_.data(); }
  const scalar *data() const { return data_.data(); }

  int *innerPtr() { return inner_.data(); }
  const int *innerPtr() const { return inner_.data(); }

  int *outerPtr() { return outer_.data(); }
  const int *outerPtr() const { return outer_.data(); }

  scalar &operator[](int i) { return data_[i]; }
  scalar &operator[](int i) const { return data_[i]; }
  // scalar &operator()(int row, int col);

  sparseMatrix &operator=(const sparseMatrix &other);
  sparseMatrix &operator+=(const sparseMatrix &other);
  sparseMatrix &operator-=(const sparseMatrix &other);
  sparseMatrix &operator*=(const double a);
  sparseMatrix operator*(const double a);
  sparseMatrix operator+(const sparseMatrix &other);
  sparseMatrix operator-(const sparseMatrix &other);
  sparseMatrix operator*(const sparseMatrix &other);
  denseMatrix<scalar> operator*(const denseMatrix<scalar> &other);
  denseMatrix<scalar> operator+(const denseMatrix<scalar> &other);
  denseMatrix<scalar> operator-(const denseMatrix<scalar> &other);

  sparseMatrix transpose();

  /// extra member functions to facilitate python wrapping
  void assign(const sparseMatrix &other);
  void setElem(scalar elem, int row, int col);
  scalar getElem(int row, int col);

  size_t size();

  void print();
  void printBlock(int startRow, int startCol, int rows, int cols);

  scalar norm();

private:
  friend class boost::serialization::access;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & nnz_;
    ar & cols_;
    ar & rows_;
    ar & data_;
    ar & inner_;
    ar & outer_;
    ar & transpose_mat;
  }

  /// size of matrix
  int cols_;
  int rows_;
  int nnz_;

  // track if matrix is transposed
  bool transpose_mat;
  void setTranspose() { transpose_mat = true; }

  /// container class to hold data for eigen map
  std::vector<scalar> data_;

  /// container classes to hold indices
  std::vector<int> inner_;
  std::vector<int> outer_;
};

#include "sparseMatrix_impl.hpp"

#endif /*SPARSEMATRIX_HPP*/
