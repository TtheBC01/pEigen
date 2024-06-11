#ifndef DENSEMATRIX_HPP
#define DENSEMATRIX_HPP

#include <vector>
#include <fstream>
#include <exception>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/vector.hpp>

#include <Eigen/Core>

class dimensionMismatch: public std::exception 
{
  virtual const char* what() const throw()
  {
    return "Dimension Mismatch. Operation not possible.";
  }
} dimensionMismatch;

template<class scalar>
class denseMatrix
{
public:

  denseMatrix();

  denseMatrix(int rows, int cols);

  denseMatrix(std::vector<scalar> &data, int rows, int cols);

  denseMatrix(std::vector<scalar> &data, int rows, int cols, bool trnsps);
 
  void resize(int rows, int cols);

  void setRandom(int seed);

  void save(std::string fname);
  void load(std::string fname);

  int rows() { return rows_; };
  int rows() const { return rows_; };
  int get_rows();

  int cols() { return cols_; };
  int cols() const { return cols_; }
  int get_cols();

  denseMatrix get_row(int i);
  denseMatrix get_col(int i);
  denseMatrix get_diagonal(int i);
  denseMatrix get_block(int i, int j, int k, int l);

  void set_row(int i, const denseMatrix& r);
  void set_col(int i, const denseMatrix& c);
  void set_diagonal(int i, const denseMatrix& d);
  void set_block(int i, int j, int k, int l, const denseMatrix& b);

  bool is_transpose() { return transpose_mat; }
  bool is_transpose() const { return transpose_mat; } 

  std::vector<scalar>& container() { return data_; }
  std::vector<scalar>  container() const { return data_; }

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > getEigenMap(); 
  Eigen::Map< const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > getEigenMap() const; 

  scalar* data() { return data_.data(); }
  const scalar* data() const { return data_.data(); }
 
  scalar &operator[](int i) { return data_[i]; }
  scalar &operator[](int i) const { return data_[i]; }
  scalar &operator()(int row, int col) { return data_[row + col*rows_]; }

  denseMatrix& operator=(const  denseMatrix& other);
  denseMatrix& operator+=(const denseMatrix& other);
  denseMatrix& operator-=(const denseMatrix& other);
  denseMatrix& operator*=(const double a);
  denseMatrix  operator*(const double a);
  denseMatrix  operator+(const denseMatrix& other);
  denseMatrix  operator-(const denseMatrix& other);
  denseMatrix  operator*(const denseMatrix& other);

  denseMatrix transpose();

  /// extra member functions to fascilitation python wrapping
  void assign(const denseMatrix& other);
  void setElem(scalar elem, int row, int col);
  scalar getElem(int row, int col) { return data_[row + col*rows_]; }

  scalar norm();

  scalar trace();

  size_t size();

  void print();

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
  void set_transpose() { transpose_mat = true; }

  /// container class to hold data for eigen map
  std::vector<scalar> data_; 

};

#include "denseMatrix_impl.hpp"

#endif /*DENSEMATRIX_HPP*/
