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
denseMatrix<scalar>::denseMatrix(boost::python::list &dataList, int rows, int cols)
    : rows_(rows),
      cols_(cols),
      transpose_mat(false)
{
  if (boost::python::len(dataList) != rows * cols)
    throw dimensionMismatch;

  for (int i = 0; i < boost::python::len(dataList); ++i)
    data_.push_back(boost::python::extract<scalar>(dataList[i]));
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
boost::python::list denseMatrix<scalar>::toList()
{
  boost::python::list list;
  const std::vector<scalar> &vec = this->container();
  for (typename std::vector<scalar>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter)
  {
    list.append(*iter);
  }

  return list;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::getRow(int row)
{
  if ((row >= rows()) || (row < 0))
    throw invalidRange;

  denseMatrix newrow(1, cols_);
  newrow.getEigenMap() = this->getEigenMap().row(row);
  return newrow;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::getCol(int col)
{
  if ((col >= cols()) || (col < 0))
    throw invalidRange;

  denseMatrix column(rows_, 1);
  column.getEigenMap() = this->getEigenMap().col(col);
  return column;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::getRowOrCol(int selector)
{
  // this is here for implementing the __getitem__ dunder method
  // Example:
  // dm[0] will return the first row
  // dm[0][0] will return the first element of the first row
  if (rows() > 1)
  { // return a row
    if ((selector + 1) > rows())
      throw invalidRange;

    denseMatrix selectedRow(1, cols());
    selectedRow.getEigenMap() = this->getEigenMap().row(selector);
    return selectedRow;
  }
  else
  {
    if ((selector + 1) > cols())
      throw invalidRange;

    denseMatrix selectedColumn(rows(), 1);
    selectedColumn.getEigenMap() = this->getEigenMap().col(selector);
    return selectedColumn;
  }
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::getDiagonal(int n)
{
  denseMatrix diag(rows_, 1);
  diag.getEigenMap() = this->getEigenMap().diagonal(n);
  return diag;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::getBlock(int row, int col, int nrows, int ncols)
{
  if ((row >= rows()) || (col >= cols()) || (row < 0) || (col < 0))
    throw invalidRange;

  denseMatrix block(nrows, ncols);
  block.getEigenMap() = this->getEigenMap().block(row, col, nrows, ncols);
  return block;
}

template <class scalar>
void denseMatrix<scalar>::setRow(int row, const denseMatrix &rmat)
{
  if ((rmat.rows() != 1) && (rmat.cols() != cols()))
    throw dimensionMismatch;

  this->getEigenMap().row(row) = rmat.getEigenMap();
}

template <class scalar>
void denseMatrix<scalar>::setCol(int col, const denseMatrix &cmat)
{
  if ((cmat.cols() != 1) && (cmat.rows() != rows()))
    throw dimensionMismatch;

  this->getEigenMap().col(col) = cmat.getEigenMap();
}

template <class scalar>
void denseMatrix<scalar>::setDiagonal(int d, const denseMatrix &dmat)
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
void denseMatrix<scalar>::setBlock(int i, int j, int k, int l, const denseMatrix &bmat)
{
  if (((i + 1) > rows()) || ((j + 1) > cols()) || (i < 0) || (j < 0))
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
  transpose_mat = other.isTranspose();
  return *this;
}

template <class scalar>
denseMatrix<scalar> &denseMatrix<scalar>::operator+=(const denseMatrix &other)
{
  if ((getRows() != other.getRows()) && (getCols() != other.getCols()))
    throw dimensionMismatch;

  if (!isTranspose() && !other.isTranspose()) // this + other
    this->getEigenMap() += other.getEigenMap();
  else if (isTranspose() && !other.isTranspose()) // this^T + other
    this->getEigenMap().transpose() += other.getEigenMap();
  else if (isTranspose() && other.isTranspose()) // this^T + other^T
    this->getEigenMap().transpose() += other.getEigenMap().transpose();
  else if (!isTranspose() && other.isTranspose()) // this + other^T
    this->getEigenMap() += other.getEigenMap().transpose();

  return *this;
}

template <class scalar>
denseMatrix<scalar> &denseMatrix<scalar>::operator-=(const denseMatrix &other)
{
  if ((getRows() != other.getRows()) && (getCols() != other.getCols()))
    throw dimensionMismatch;

  if (!isTranspose() && !other.isTranspose()) // this - other
    this->getEigenMap() -= other.getEigenMap();
  else if (isTranspose() && !other.isTranspose()) // this^T - other
    this->getEigenMap().transpose() -= other.getEigenMap();
  else if (isTranspose() && other.isTranspose()) // this^T - other^T
    this->getEigenMap().transpose() -= other.getEigenMap().transpose();
  else if (!isTranspose() && other.isTranspose()) // this - other^T
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
denseMatrix<scalar> denseMatrix<scalar>::operator-()
{
  return (*this)*(-1);
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator*(const denseMatrix &other)
{
  if (getCols() != other.getRows())
    throw dimensionMismatch;

  // size the result properly
  denseMatrix<scalar> result(getRows(), other.getCols());

  if (!isTranspose() && !other.isTranspose())
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap();
  else if (isTranspose() && !other.isTranspose())
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap();
  else if (isTranspose() && other.isTranspose())
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap().transpose();
  else if (!isTranspose() && other.isTranspose())
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap().transpose();

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator*(const sparseMatrix<scalar> &other)
{
  if (getCols() != other.getRows())
    throw dimensionMismatch;

  denseMatrix<scalar> result(getRows(), other.getCols());

  if (!isTranspose() && !other.isTranspose()) // this * this
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap();
  if (!isTranspose() && other.isTranspose()) // this * this^T
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap().transpose();
  if (isTranspose() && other.isTranspose()) // this^T * this^T
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap().transpose();
  if (isTranspose() && !other.isTranspose()) // this^T * this
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap();

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator+(const sparseMatrix<scalar> &other)
{
  if ((getCols() != other.getCols()) && (getRows() != other.getRows()))
    throw dimensionMismatch;

  denseMatrix<scalar> result(getRows(), getCols());

  if (!isTranspose() && !other.isTranspose()) // this + this
    result.getEigenMap() = this->getEigenMap() + other.getEigenMap();
  if (!isTranspose() && other.isTranspose()) // this + this^T
    result.getEigenMap() = this->getEigenMap() + other.getEigenMap().transpose();
  if (isTranspose() && other.isTranspose()) // this^T + this^T
    result.getEigenMap() = this->getEigenMap().transpose() + other.getEigenMap().transpose();
  if (isTranspose() && !other.isTranspose()) // this^T + this
    result.getEigenMap() = this->getEigenMap().transpose() + other.getEigenMap();

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator-(const sparseMatrix<scalar> &other)
{
  if ((getCols() != other.getCols()) && (getRows() != other.getRows()))
    throw dimensionMismatch;

  denseMatrix<scalar> result(getRows(), getCols());

  if (!isTranspose() && !other.isTranspose()) // this - this
    result.getEigenMap() = this->getEigenMap() - other.getEigenMap();
  if (!isTranspose() && other.isTranspose()) // this - this^T
    result.getEigenMap() = this->getEigenMap() - other.getEigenMap().transpose();
  if (isTranspose() && other.isTranspose()) // this^T - this^T
    result.getEigenMap() = this->getEigenMap().transpose() - other.getEigenMap().transpose();
  if (isTranspose() && !other.isTranspose()) // this^T - this
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
denseMatrix<scalar> operator*(double a, denseMatrix<scalar>& other)
{
  denseMatrix<scalar> result;
  result = other;
  result *= a;
  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::transpose()
{
  denseMatrix<scalar> tmat(data_, rows_, cols_);
  tmat.setTranspose();
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
  if (!isTranspose())
    std::cout << "data =\n"
              << this->getEigenMap() << std::endl;
  else
    std::cout << "data = \n"
              << this->getEigenMap().transpose() << std::endl;
}
