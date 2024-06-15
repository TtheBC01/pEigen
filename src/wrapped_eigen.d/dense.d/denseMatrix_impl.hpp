#include <wrapped_eigen.d/sparse.d/sparseMatrix.hpp>

template <class scalar>
denseMatrix<scalar>::denseMatrix()
    : rows_(0),
      cols_(0),
      transpose_mat(false)
{
}

template <class scalar>
denseMatrix<scalar>::denseMatrix(int rows, int cols)
    : rows_(rows),
      cols_(cols),
      transpose_mat(false)
{
  resize(rows_, cols_);
}

template <class scalar>
denseMatrix<scalar>::denseMatrix(std::vector<scalar> &data, int rows, int cols)
    : data_(data),
      rows_(rows),
      cols_(cols),
      transpose_mat(false)
{
}

template <class scalar>
void denseMatrix<scalar>::resize(int rows, int cols)
{
  rows_ = rows;
  cols_ = cols;
  data_.resize(rows_ * cols_, 0);
}

template <class scalar>
void denseMatrix<scalar>::setRandom(int seed)
{
  std::srand(seed);
  this->getEigenMap() = Eigen::MatrixXd::Random(rows(), cols());
}

template <class scalar>
void denseMatrix<scalar>::save(std::string fname)
{
  std::ofstream ofs(fname.c_str());
  boost::archive::text_oarchive oa(ofs);
  oa << *this;
  ofs.close();
}

template <class scalar>
void denseMatrix<scalar>::load(std::string fname)
{
  std::ifstream ifs(fname.c_str());
  boost::archive::text_iarchive ia(ifs);
  ia >> *this;
  ifs.close();
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::get_row(int row)
{
  denseMatrix newrow(1, cols_);
  newrow.getEigenMap() = this->getEigenMap().row(row);
  return newrow;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::get_col(int col)
{
  denseMatrix column(rows_, 1);
  column.getEigenMap() = this->getEigenMap().col(col);
  return column;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::get_diagonal(int n)
{
  denseMatrix diag(rows_, 1);
  diag.getEigenMap() = this->getEigenMap().diagonal(n);
  return diag;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::get_block(int row, int col, int nrows, int ncols)
{
  if ((row > rows()) || (col > cols()) || (row < 0) || (col < 0))
    throw invalidRange;

  denseMatrix block(nrows, ncols);
  block.getEigenMap() = this->getEigenMap().block(row, col, nrows, ncols);
  return block;
}

template <class scalar>
void denseMatrix<scalar>::set_row(int row, const denseMatrix &rmat)
{
  if ((rmat.rows() != 1) && (rmat.cols() != cols_))
    throw dimensionMismatch;

  this->getEigenMap().row(row) = rmat.getEigenMap();
}

template <class scalar>
void denseMatrix<scalar>::set_col(int col, const denseMatrix &cmat)
{
  if ((cmat.cols() != 1) && (cmat.rows() != rows_))
    throw dimensionMismatch;

  this->getEigenMap().col(col) = cmat.getEigenMap();
}

template <class scalar>
void denseMatrix<scalar>::set_diagonal(int d, const denseMatrix &dmat)
{
  // first, dmat must be a vector 
  if ((dmat.cols() != 1) && (dmat.rows() != 1))
    throw dimensionMismatch;
  
  // second, the length of the diagonal must equal the length of dmat
  if ((dmat.cols() != this->getEigenMap().diagonal(d).rows()) && (dmat.rows() != this->getEigenMap().diagonal(d).rows()))
    throw dimensionMismatch;

  this->getEigenMap().diagonal(d) = dmat.getEigenMap();
}

template <class scalar>
void denseMatrix<scalar>::set_block(int i, int j, int k, int l, const denseMatrix &bmat)
{
  if ( (i > rows()) || (j > cols()) || (i < 0) || (j < 0))
    throw invalidRange;

  if ((k != bmat.rows()) && (l != bmat.cols()))
    throw dimensionMismatch;

  this->getEigenMap().block(i, j, k, l) = bmat.getEigenMap();
}

template <class scalar>
Eigen::Map<Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>> denseMatrix<scalar>::getEigenMap()
{
  Eigen::Map<Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>> x(this->data(),
                                                                      rows_,
                                                                      cols_);

  return x;
}

template <class scalar>
Eigen::Map<const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>> denseMatrix<scalar>::getEigenMap() const
{
  Eigen::Map<const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>> x(this->data(),
                                                                            rows_,
                                                                            cols_);

  return x;
}

template <class scalar>
denseMatrix<scalar> &denseMatrix<scalar>::operator=(const denseMatrix &other)
{
  rows_ = other.rows();
  cols_ = other.cols();
  data_ = other.container();
  transpose_mat = other.is_transpose();
  return *this;
}

template <class scalar>
denseMatrix<scalar> &denseMatrix<scalar>::operator+=(const denseMatrix &other)
{
  if ((get_rows() != other.get_rows()) && (get_cols() != other.get_cols()))
    throw dimensionMismatch;

  if (!is_transpose() && !other.is_transpose()) // this + other
    this->getEigenMap() += other.getEigenMap();
  else if (is_transpose() && !other.is_transpose()) // this^T + other
    this->getEigenMap().transpose() += other.getEigenMap();
  else if (is_transpose() && other.is_transpose()) // this^T + other^T
    this->getEigenMap().transpose() += other.getEigenMap().transpose();
  else if (!is_transpose() && other.is_transpose()) // this + other^T
    this->getEigenMap() += other.getEigenMap().transpose();

  return *this;
}

template <class scalar>
denseMatrix<scalar> &denseMatrix<scalar>::operator-=(const denseMatrix &other)
{
  if ((get_rows() != other.get_rows()) && (get_cols() != other.get_cols()))
    throw dimensionMismatch;

  if (!is_transpose() && !other.is_transpose()) // this - other
    this->getEigenMap() -= other.getEigenMap();
  else if (is_transpose() && !other.is_transpose()) // this^T - other
    this->getEigenMap().transpose() -= other.getEigenMap();
  else if (is_transpose() && other.is_transpose()) // this^T - other^T
    this->getEigenMap().transpose() -= other.getEigenMap().transpose();
  else if (!is_transpose() && other.is_transpose()) // this - other^T
    this->getEigenMap() -= other.getEigenMap().transpose();

  return *this;
}

template <class scalar>
denseMatrix<scalar> &denseMatrix<scalar>::operator*=(const double a)
{
  this->getEigenMap() *= a;

  return *this;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator+(const denseMatrix &other)
{
  denseMatrix<scalar> result = other;
  result += *this;

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator-(const denseMatrix &other)
{
  denseMatrix<scalar> result = *this;
  result -= other;

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator*(const denseMatrix &other)
{
  if (get_cols() != other.get_rows())
    throw dimensionMismatch;

  // size the result properly
  denseMatrix<scalar> result(get_rows(), other.get_cols());

  if (!is_transpose() && !other.is_transpose())
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap();
  else if (is_transpose() && !other.is_transpose())
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap();
  else if (is_transpose() && other.is_transpose())
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap().transpose();
  else if (!is_transpose() && other.is_transpose())
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap().transpose();

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator*(const sparseMatrix<scalar> &other)
{
  if (get_cols() != other.get_rows())
  {
    throw dimensionMismatch;
  }

  denseMatrix<scalar> result(get_rows(), other.get_cols());

  if (!is_transpose() && !other.is_transpose()) // this * this
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap();
  if (!is_transpose() && other.is_transpose()) // this * this^T
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap().transpose();
  if (is_transpose() && other.is_transpose()) // this^T * this^T
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap().transpose();
  if (is_transpose() && !other.is_transpose()) // this^T * this
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap();

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator+(const sparseMatrix<scalar> &other)
{
  if ((get_cols() != other.get_cols()) && (get_rows() != other.get_rows()))
  {
    throw dimensionMismatch;
  }

  denseMatrix<scalar> result(get_rows(), get_cols());

  if (!is_transpose() && !other.is_transpose()) // this + this
    result.getEigenMap() = this->getEigenMap() + other.getEigenMap();
  if (!is_transpose() && other.is_transpose()) // this + this^T
    result.getEigenMap() = this->getEigenMap() + other.getEigenMap().transpose();
  if (is_transpose() && other.is_transpose()) // this^T + this^T
    result.getEigenMap() = this->getEigenMap().transpose() + other.getEigenMap().transpose();
  if (is_transpose() && !other.is_transpose()) // this^T + this
    result.getEigenMap() = this->getEigenMap().transpose() + other.getEigenMap();

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator-(const sparseMatrix<scalar> &other)
{
  if ((get_cols() != other.get_cols()) && (get_rows() != other.get_rows()))
  {
    throw dimensionMismatch;
  }

  denseMatrix<scalar> result(get_rows(), get_cols());

  if (!is_transpose() && !other.is_transpose()) // this - this
    result.getEigenMap() = this->getEigenMap() - other.getEigenMap();
  if (!is_transpose() && other.is_transpose()) // this - this^T
    result.getEigenMap() = this->getEigenMap() - other.getEigenMap().transpose();
  if (is_transpose() && other.is_transpose()) // this^T - this^T
    result.getEigenMap() = this->getEigenMap().transpose() - other.getEigenMap().transpose();
  if (is_transpose() && !other.is_transpose()) // this^T - this
    result.getEigenMap() = this->getEigenMap().transpose() - other.getEigenMap();

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator*(const double a)
{
  denseMatrix<scalar> result;
  result = *this;
  result *= a;
  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::transpose()
{
  denseMatrix<scalar> tmat(data_, rows_, cols_);
  tmat.set_transpose();
  return tmat;
}

template <class scalar>
void denseMatrix<scalar>::assign(const denseMatrix &other)
{
  *this = other;
}

template <class scalar>
scalar denseMatrix<scalar>::getElem(int row, int col)
{
  if ((row >= rows()) || (col >= cols()) || (row < 0) || (col < 0))
    throw invalidRange;
  else
    return data_[row + col * rows_];
}

template <class scalar>
void denseMatrix<scalar>::setElem(scalar elem, int row, int col)
{
  if ((row >= rows()) || (col >= cols()) || (row < 0) || (col < 0))
    throw invalidRange;
  else
    (*this)(row, col) = elem;
}

template <class scalar>
scalar denseMatrix<scalar>::norm()
{
  return this->getEigenMap().norm();
}

template <class scalar>
scalar denseMatrix<scalar>::trace()
{
  return this->getEigenMap().trace();
}

template <class scalar>
size_t denseMatrix<scalar>::size()
{
  return data_.size();
}

template <class scalar>
void denseMatrix<scalar>::print()
{
  if (!is_transpose())
    std::cout << "data =\n"
              << this->getEigenMap() << std::endl;
  else
    std::cout << "data = \n"
              << this->getEigenMap().transpose() << std::endl;
}
