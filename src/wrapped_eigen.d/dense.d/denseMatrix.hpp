#ifndef DENSEMATRIX_HPP
#define DENSEMATRIX_HPP

#include <vector>
#include <fstream>
#include <exception>
#include <string>
#include <sstream>

#include <boost/python.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/vector.hpp>

#include <Eigen/Core>

template<class scalar> class sparseMatrix;

class dimensionMismatch : public std::exception
{
  virtual const char *what() const throw()
  {
    return "Dimension Mismatch. Operation not possible.";
  }
} dimensionMismatch;

class invalidRange : public std::exception
{
  virtual const char *what() const throw()
  {
    return "ERROR: Requested element is out of range.";
  }
} invalidRange;

template <class scalar>
class denseMatrix
{
public:
  denseMatrix();

  denseMatrix(int rows, int cols);

  denseMatrix(std::vector<scalar> &data, int rows, int cols);

  denseMatrix(boost::python::list &dataList, int rows, int cols);

  void resize(int rows, int cols);

  void setRandom(int seed);

  void save(std::string fname);
  void load(std::string fname);

  std::string repr() {return "Dense Matrix: " + std::to_string(rows()) + " rows, " + std::to_string(cols()) + " cols";}

  boost::python::list toList();

  int rows() { return rows_; };
  int rows() const { return rows_; };
  int getRows() { return transpose_mat ? cols_ : rows_; }
  int getRows() const { return transpose_mat ? cols_ : rows_; }

  int cols() { return cols_; };
  int cols() const { return cols_; }
  int getCols() { return transpose_mat ? rows_ : cols_; }
  int getCols() const { return transpose_mat ? rows_ : cols_; }

  denseMatrix getRow(int i);
  denseMatrix getCol(int i);
  denseMatrix getRowOrCol(int i);
  denseMatrix getDiagonal(int i);
  denseMatrix getBlock(int i, int j, int k, int l);

  void setRow(int i, const denseMatrix &r);
  void setCol(int i, const denseMatrix &c);
  void setDiagonal(int i, const denseMatrix &d);
  void setBlock(int i, int j, int k, int l, const denseMatrix &b);

  bool isTranspose() { return transpose_mat; }
  bool isTranspose() const { return transpose_mat; }

  std::vector<scalar> &container() { return data_; }
  std::vector<scalar> container() const { return data_; }

  Eigen::Map<Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>> getEigenMap();
  Eigen::Map<const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>> getEigenMap() const;

  scalar *data() { return data_.data(); }
  const scalar *data() const { return data_.data(); }

  scalar &operator[](int i) { return data_[i]; }
  scalar &operator[](int i) const { return data_[i]; }
  scalar &operator()(int row, int col) { return data_[row + col * rows_]; }

  denseMatrix &operator=(const denseMatrix &other);
  denseMatrix &operator+=(const denseMatrix &other);
  denseMatrix &operator-=(const denseMatrix &other);
  denseMatrix &operator*=(const double a);
  denseMatrix operator*(const double a);
  template<scalar> friend denseMatrix operator*(double a, denseMatrix<scalar>& other);
  denseMatrix operator+(const denseMatrix &other);
  denseMatrix operator-(const denseMatrix &other);
  denseMatrix operator-();
  denseMatrix operator*(const denseMatrix &other);
  denseMatrix operator*(const sparseMatrix<scalar> &other);
  denseMatrix operator+(const sparseMatrix<scalar> &other);
  denseMatrix operator-(const sparseMatrix<scalar> &other);

  denseMatrix transpose();

  /// extra member functions to fascilitation python wrapping
  void assign(const denseMatrix &other);
  void setElem(scalar elem, int row, int col);
  scalar getElem(int row, int col);

  scalar norm();

  scalar trace();

  size_t size();

  void print();
  std::string str();

private:
  friend class boost::serialization::access;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & cols_;
    ar & rows_;
    ar & data_;
    ar & transpose_mat;
  }

  /// size of matrix
  int cols_;
  int rows_;

  // track if matrix is transposed
  bool transpose_mat;
  void setTranspose() { transpose_mat = true; }

  /// container class to hold data for eigen map
  std::vector<scalar> data_;
};

#include "denseMatrix_impl.hpp"

#endif /*DENSEMATRIX_HPP*/
