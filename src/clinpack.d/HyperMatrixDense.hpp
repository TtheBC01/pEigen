#ifndef HYPERMATRIXDENSE_HPP
#define HYPERMATRIXDENSE_HPP

#include <vector>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/complex.hpp>

#include <Eigen/Core>

template<class scalar>
class HyperMatrixDense
{
public:

  HyperMatrixDense();

  HyperMatrixDense(int rows, int cols);

  HyperMatrixDense(std::vector<scalar> &data, int rows, int cols);

  HyperMatrixDense(std::vector<scalar> &data, int rows, int cols, bool trnsps);
 
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

  HyperMatrixDense get_row(int i);
  HyperMatrixDense get_col(int i);
  HyperMatrixDense get_diagonal(int i);
  HyperMatrixDense get_block(int i, int j, int k, int l);

  void set_row(int i, const HyperMatrixDense& r);
  void set_col(int i, const HyperMatrixDense& c);
  void set_diagonal(int i, const HyperMatrixDense& d);
  void set_block(int i, int j, int k, int l, const HyperMatrixDense& b);

  bool is_transpose() { return transpose_mat; }
  bool is_transpose() const { return transpose_mat; } 

  std::vector<scalar>& container() { return data_; }
  std::vector<scalar>  container() const { return data_; }

  scalar* data() { return data_.data(); }
  scalar* data() const { return data_.data(); }
 
  scalar &operator[](int i) { return data_[i]; }
  scalar &operator[](int i) const { return data_[i]; }
  scalar &operator()(int row, int col) { return data_[row + col*rows_]; }

  HyperMatrixDense& operator=(const  HyperMatrixDense& other);
  HyperMatrixDense& operator+=(const HyperMatrixDense& other);
  HyperMatrixDense& operator-=(const HyperMatrixDense& other);
  HyperMatrixDense& operator*=(const double a);
  HyperMatrixDense  operator*(const double a);
  HyperMatrixDense  operator+(const HyperMatrixDense& other);
  HyperMatrixDense  operator-(const HyperMatrixDense& other);
  HyperMatrixDense  operator*(const HyperMatrixDense& other);

  HyperMatrixDense transpose();

  /// extra member functions to fascilitation python wrapping
  void assign(const HyperMatrixDense& other);
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
  
  bool transpose_mat;
  void set_transpose() { transpose_mat = true; }

  /// container class to hold data for eigen map
  std::vector<scalar> data_; 

};

#include "HyperMatrixDense_impl.hpp"

#endif /*HYPERMATRIXDENSE_HPP*/
