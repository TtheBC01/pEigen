#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/complex.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

template<class scalar>
class sparseMatrix
{
public:

  sparseMatrix();

  sparseMatrix(int rows, int cols);
 
  sparseMatrix(int rows, int cols, int nnz);

  sparseMatrix(std::vector<scalar> &data, std::vector<int> &outer_, std::vector<int> &inner_, int rows, int cols);
 
  void resize(int rows, int cols);

  void reserve(int rows, int cols, int nnz);

  void clear();

  void save(std::string fname);
  void load(std::string fname);

  int nnz() { return nnz_; }
  int nnz() const { return nnz_; }

  int rows() { return rows_; }
  int rows() const { return rows_; }

  int cols() { return cols_; }
  int cols() const { return cols_; }

  std::vector<scalar>& data_container() { return data_; }
  std::vector<scalar>  data_container() const { return data_; }

  std::vector<int>& inner_container() { return inner_; }
  std::vector<int>  inner_container() const { return inner_; }

  std::vector<int>& outer_container() { return outer_; }
  std::vector<int>  outer_container() const { return outer_; }

  scalar* data() { return data_.data(); }
  scalar* data() const { return data_.data(); }
 
  int* innerPtr() { return inner_.data(); }
  int* innerPtr() const { return inner_.data(); }

  int* outerPtr() { return outer_.data(); }
  int* outerPtr() const { return outer_.data(); }

  scalar &operator[](int i) { return data_[i]; }
  scalar &operator[](int i) const { return data_[i]; }
  //scalar &operator()(int row, int col);

  sparseMatrix& operator=(const  sparseMatrix& other);
  sparseMatrix& operator+=(const sparseMatrix& other);
  sparseMatrix& operator*=(const double a);
  sparseMatrix  operator*(const double a);
  sparseMatrix  operator+(const sparseMatrix& other);
  sparseMatrix  operator*(const sparseMatrix& other);

  /// extra member functions to fascilitation python wrapping
  void assign(const sparseMatrix& other);
  void setElem(scalar elem, int row, int col);

  size_t size();

  void print();

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
  }

  /// size of matrix
  int cols_;
  int rows_;
  int nnz_;
  
  /// container class to hold data for eigen map
  std::vector<scalar> data_; 

  /// container classes to hold indices
  std::vector<int> inner_;
  std::vector<int> outer_;

};

#include "sparseMatrix_impl.hpp"

#endif /*SPARSEMATRIX_HPP*/
