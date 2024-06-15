#include <wrapped_eigen.d/dense.d/denseMatrix.hpp>

template <class scalar>
sparseMatrix<scalar>::sparseMatrix()
    : rows_(0),
      cols_(0),
      outer_(1, 0),
      nnz_(0),
      transpose_mat(false)
{
  inner_.push_back(0);
}

template <class scalar>
sparseMatrix<scalar>::sparseMatrix(int rows, int cols)
    : rows_(rows),
      cols_(cols),
      outer_(cols + 1, 0),
      nnz_(0),
      transpose_mat(false)
{
}

template <class scalar>
sparseMatrix<scalar>::sparseMatrix(int rows, int cols, int nnz)
    : rows_(rows),
      cols_(cols),
      data_(nnz),
      outer_(cols + 1, 0),
      inner_(nnz),
      nnz_(nnz),
      transpose_mat(false)
{
}

template <class scalar>
sparseMatrix<scalar>::sparseMatrix(std::vector<scalar> &data, std::vector<int> &outer, std::vector<int> &inner, int rows, int cols)
    : data_(data),
      outer_(outer),
      inner_(inner),
      rows_(rows),
      cols_(cols),
      nnz_(data.size()),
      transpose_mat(false)
{
}

template <class scalar>
void sparseMatrix<scalar>::resize(int rows, int cols)
{
  rows_ = rows;
  cols_ = cols;
  outer_.assign(cols + 1, 0);
}

template <class scalar>
void sparseMatrix<scalar>::reserve(int rows, int cols, int nnz)
{
  rows_ = rows;
  cols_ = cols;
  nnz_ = nnz;
  data_.assign(nnz_, 0);
  outer_.assign(cols + 1, 0);
  inner_.assign(nnz_, 0);
}

template <class scalar>
void sparseMatrix<scalar>::clear()
{
  nnz_ = 0;
  data_.clear();
  inner_.clear();
  outer_.clear();
  outer_.assign(cols_ + 1, 0);
}

template <class scalar>
void sparseMatrix<scalar>::save(std::string fname)
{
  std::ofstream ofs(fname.c_str());
  boost::archive::text_oarchive oa(ofs);
  oa << *this;
  ofs.close();
}

template <class scalar>
void sparseMatrix<scalar>::load(std::string fname)
{
  std::ifstream ifs(fname.c_str());
  boost::archive::text_iarchive ia(ifs);
  ia >> *this;
  ifs.close();
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::get_col(int col)
{

  int start = outer_[col];
  int end = outer_[col + 1];
  int nnz = end - start;

  std::vector<int> newInner(nnz);
  std::vector<scalar> newData(nnz);
  for (int i = 0; i < nnz; i++)
  {
    newInner[i] = inner_[i + start];
    newData[i] = data_[i + start];
  }

  std::vector<int> newOuter(2);
  newOuter[0] = 0;
  newOuter[1] = nnz;

  sparseMatrix newCol(newData, newOuter, newInner, rows_, 1);

  return newCol;
}

template <class scalar>
Eigen::Map<Eigen::SparseMatrix<scalar>> sparseMatrix<scalar>::getEigenMap()
{
  Eigen::Map<Eigen::SparseMatrix<scalar>> sm(rows(), cols(), nnz(), outer_.data(), inner_.data(), data_.data());

  return sm;
}

template <class scalar>
Eigen::Map<const Eigen::SparseMatrix<scalar>> sparseMatrix<scalar>::getEigenMap() const
{
  Eigen::Map<const Eigen::SparseMatrix<scalar>> sm(rows_, cols_, nnz_, outer_.data(), inner_.data(), data_.data());

  return sm;
}

template <class scalar>
sparseMatrix<scalar> &sparseMatrix<scalar>::operator=(const sparseMatrix &other)
{
  nnz_ = other.nnz();
  rows_ = other.rows();
  cols_ = other.cols();
  data_ = other.data_container();
  inner_ = other.inner_container();
  outer_ = other.outer_container();
  return *this;
}

template <class scalar>
sparseMatrix<scalar> &sparseMatrix<scalar>::operator+=(const sparseMatrix &other)
{
  if (get_rows() != other.get_rows() || get_cols() != other.get_cols())
    throw dimensionMismatch;

  Eigen::SparseMatrix<scalar> result(get_rows(), get_rows());

  if (is_transpose())
    result = Eigen::SparseMatrix<scalar>(this->getEigenMap().transpose());
  else
    result = this->getEigenMap();

  if (other.is_transpose())
    result += Eigen::SparseMatrix<scalar>(other.getEigenMap().transpose());
  else
    result += other.getEigenMap();

  clear();

  for (int k = 0; k < result.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<scalar>::InnerIterator it(result, k); it; ++it)
      setElem(it.value(), (is_transpose() ? it.col() : it.row()), (is_transpose() ? it.row() : it.col()));

  return *this;
}

template <class scalar>
sparseMatrix<scalar> &sparseMatrix<scalar>::operator-=(const sparseMatrix &other)
{
  if (get_rows() != other.get_rows() || get_cols() != other.get_cols())
    throw dimensionMismatch;

  Eigen::SparseMatrix<scalar> result(get_rows(), get_rows());

  if (is_transpose())
    result = Eigen::SparseMatrix<scalar>(this->getEigenMap().transpose());
  else
    result = this->getEigenMap();

  if (other.is_transpose())
    result -= Eigen::SparseMatrix<scalar>(other.getEigenMap().transpose());
  else
    result -= other.getEigenMap();

  clear();

  for (int k = 0; k < result.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<scalar>::InnerIterator it(result, k); it; ++it)
      setElem(it.value(), (is_transpose() ? it.col() : it.row()), (is_transpose() ? it.row() : it.col()));

  return *this;
}

template <class scalar>
sparseMatrix<scalar> &sparseMatrix<scalar>::operator*=(const double a)
{
  this->getEigenMap() *= a;
  return *this;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::operator+(const sparseMatrix &other)
{
  if ((get_cols() != other.get_cols()) && (get_rows() != other.get_rows()))
  {
    throw dimensionMismatch;
  }

  sparseMatrix<scalar> result = other;
  result += *this;

  return result;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::operator-(const sparseMatrix &other)
{
  if ((get_cols() != other.get_cols()) && (get_rows() != other.get_rows()))
  {
    throw dimensionMismatch;
  }

  sparseMatrix<scalar> result = *this;
  result -= other;

  return result;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::operator*(const sparseMatrix &other)
{
  if (get_cols() != other.get_rows())
  {
    throw dimensionMismatch;
  }

  Eigen::SparseMatrix<scalar> result(get_rows(), other.get_cols());
  sparseMatrix<scalar> container(get_rows(), other.get_cols());

  if (!is_transpose() && !other.is_transpose())                    // this * other
    result = (this->getEigenMap() * other.getEigenMap()).pruned(); // prune explicit zeros from the result
  else if (!is_transpose() && other.is_transpose())                // this * other^T
    result = (this->getEigenMap() * other.getEigenMap().transpose()).pruned();
  else if (is_transpose() && other.is_transpose()) // this^T * other^T
    result = (this->getEigenMap().transpose() * other.getEigenMap().transpose()).pruned();
  else if (is_transpose() && !other.is_transpose()) // this^T * other
    result = (this->getEigenMap().transpose() * other.getEigenMap()).pruned();

  for (int k = 0; k < result.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<scalar>::InnerIterator it(result, k); it; ++it)
      container.setElem(it.value(), it.row(), it.col());

  return container;
}

template <class scalar>
denseMatrix<scalar> sparseMatrix<scalar>::operator*(const denseMatrix<scalar> &other)
{
  if (get_cols() != other.get_rows())
  {
    throw dimensionMismatch;
  }
  denseMatrix<scalar> result(get_rows(), other.get_cols());

  if (!is_transpose() && !other.is_transpose()) // this * other
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap();
  else if (!is_transpose() && other.is_transpose()) // this * other^T
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap().transpose();
  else if (is_transpose() && other.is_transpose()) // this^T * other^T
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap().transpose();
  else if (is_transpose() && !other.is_transpose()) // this^T * other
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap();

  return result;
}

template <class scalar>
denseMatrix<scalar> sparseMatrix<scalar>::operator+(const denseMatrix<scalar> &other)
{
  if ((get_cols() != other.get_cols()) && (get_rows() != other.get_rows()))
  {
    throw dimensionMismatch;
  }

  denseMatrix<scalar> result(get_rows(), get_cols());

  if (other.is_transpose())
    result.getEigenMap() = other.getEigenMap().transpose();
  else
    result.getEigenMap() = other.getEigenMap();

  if (is_transpose())
    result.getEigenMap() += this->getEigenMap().transpose();
  else
    result.getEigenMap() += this->getEigenMap();

  return result;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::operator*(const double a)
{
  sparseMatrix<scalar> result;
  result = *this;
  result *= a;
  return result;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::transpose()
{
  sparseMatrix<scalar> tmat(data_, outer_, inner_, rows_, cols_);
  tmat.set_transpose();
  return tmat;
}

template <class scalar>
void sparseMatrix<scalar>::assign(const sparseMatrix &other)
{
  *this = other;
}

template <class scalar>
void sparseMatrix<scalar>::setElem(scalar elem, int row, int col)
{
  if ((row >= rows()) || (row < 0) || (col >= cols()) || (col < 0))
    throw invalidRange;

  // get nnz up to row before insertion row
  if (nnz() > 0)
  {
    int nnz1 = outer_[col];     // number of slots to skip
    int nnz2 = outer_[col + 1]; // number fo nnzs in col being changed
    std::vector<int>::iterator itInner = inner_.begin() + nnz1;
    std::vector<int>::iterator itInnerStop = inner_.begin() + nnz2;
    typename std::vector<scalar>::iterator itData = data_.begin() + nnz1;
    // find where to insert in the row indices
    for (; itInner != itInnerStop; itInner++, itData++)
      if (*itInner == row)
      { // if the element is set already, we're going to overwrite it
        *itData = elem;
        return;
      }
      else if (*itInner > row)
      { // if it hasn't been set already, we insert it
        break;
      }

    inner_.insert(itInner, row);
    data_.insert(itData, elem);
  }
  else
  {
    data_.push_back(elem);
    inner_.push_back(row);
  }
  // update nnz counter
  for (std::vector<int>::iterator it = outer_.begin() + col + 1; it != outer_.end(); it++)
    (*it)++;

  nnz_++;
}

template <class scalar>
scalar sparseMatrix<scalar>::getElem(int row, int col)
{
  if ((row >= rows()) || (row < 0) || (col >= cols()) || (col < 0))
    throw invalidRange;

  // get nnz up to row before insertion row
  if (nnz() > 0)
  {
    int nnz1 = outer_[col];     // number of slots to skip
    int nnz2 = outer_[col + 1]; // number fo nnzs in col being changed
    std::vector<int>::iterator itInner = inner_.begin() + nnz1;
    std::vector<int>::iterator itInnerStop = inner_.begin() + nnz2;
    typename std::vector<scalar>::iterator itData = data_.begin() + nnz1;
    // find where to insert in the row indices
    for (; itInner != itInnerStop; itInner++, itData++)
      if (*itInner == row) // if the row is in inner_ return it
        return *itData;
      else if (*itInner > row) // if we've already passed the row, we can return 0
        break;
  }

  return scalar(0);
}

template <class scalar>
size_t sparseMatrix<scalar>::size()
{
  return data_.size();
}

template <class scalar>
void sparseMatrix<scalar>::print()
{
  if (!is_transpose())
    std::cout << "data =\n"
              << this->getEigenMap() << std::endl;
  else
    std::cout << "data = \n"
              << this->getEigenMap().transpose() << std::endl;
}

template <class scalar>
void sparseMatrix<scalar>::print_block(int startRow, int startCol, int rows, int cols)
{
  if (!is_transpose())
    std::cout << "data =\n"
              << this->getEigenMap().block(startRow, startCol, rows, cols) << std::endl;
  else
    std::cout << "data = \n"
              << this->getEigenMap().transpose().block(startRow, startCol, rows, cols) << std::endl;
}

template <class scalar>
scalar sparseMatrix<scalar>::norm()
{
  return this->getEigenMap().norm();
}