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
sparseMatrix<scalar>::sparseMatrix(boost::python::list &dataList, boost::python::list &outerList, boost::python::list &innerList, int rows, int cols)
    : rows_(rows),
      cols_(cols),
      nnz_(boost::python::len(dataList)),
      transpose_mat(false)
{
  if ((boost::python::len(dataList) > rows * cols) || (boost::python::len(outerList) != cols + 1) || (boost::python::len(dataList) != boost::python::len(innerList)))
    throw dimensionMismatch;
    
  for (int i = 0; i < boost::python::len(dataList); ++i)
    data_.push_back(boost::python::extract<scalar>(dataList[i]));

  for (int i = 0; i < boost::python::len(outerList); ++i)
    outer_.push_back(boost::python::extract<int>(outerList[i]));

  for (int i = 0; i < boost::python::len(innerList); ++i)
  {
    int newRow = boost::python::extract<int>(innerList[i]);
    if (newRow >= rows)
      throw invalidRange;
      
    inner_.push_back(newRow);
  }
}

template <class scalar>
void sparseMatrix<scalar>::resize(int rows, int cols)
{
  rows_ = rows;
  cols_ = cols;
  outer_.assign(cols + 1, 0);
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
boost::python::list sparseMatrix<scalar>::dataToList()
{
  boost::python::list list;
  const std::vector<scalar> &vec = this->dataContainer();
  for (typename std::vector<scalar>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter)
  {
    list.append(*iter);
  }

  return list;
}

template <class scalar>
boost::python::list sparseMatrix<scalar>::outerToList()
{
  boost::python::list list;
  const std::vector<int> &vec = this->outerContainer();
  for (std::vector<int>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter)
  {
    list.append(*iter);
  }

  return list;
}

template <class scalar>
boost::python::list sparseMatrix<scalar>::innerToList()
{
  boost::python::list list;
  const std::vector<int> &vec = this->innerContainer();
  for (std::vector<int>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter)
  {
    list.append(*iter);
  }

  return list;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::getCol(int col)
{
  if (col >= cols() || (col < 0))
    throw invalidRange;

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
  data_ = other.dataContainer();
  inner_ = other.innerContainer();
  outer_ = other.outerContainer();
  transpose_mat = other.isTranspose();
  return *this;
}

template <class scalar>
sparseMatrix<scalar> &sparseMatrix<scalar>::operator+=(const sparseMatrix &other)
{
  if (getRows() != other.getRows() || getCols() != other.getCols())
    throw dimensionMismatch;

  Eigen::SparseMatrix<scalar> result(getRows(), getRows());

  if (isTranspose())
    result = Eigen::SparseMatrix<scalar>(this->getEigenMap().transpose());
  else
    result = this->getEigenMap();

  if (other.isTranspose())
    result += Eigen::SparseMatrix<scalar>(other.getEigenMap().transpose());
  else
    result += other.getEigenMap();

  clear();

  for (int k = 0; k < result.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<scalar>::InnerIterator it(result, k); it; ++it)
      setElem(it.value(), (isTranspose() ? it.col() : it.row()), (isTranspose() ? it.row() : it.col()));

  return *this;
}

template <class scalar>
sparseMatrix<scalar> &sparseMatrix<scalar>::operator-=(const sparseMatrix &other)
{
  if (getRows() != other.getRows() || getCols() != other.getCols())
    throw dimensionMismatch;

  Eigen::SparseMatrix<scalar> result(getRows(), getRows());

  if (isTranspose())
    result = Eigen::SparseMatrix<scalar>(this->getEigenMap().transpose());
  else
    result = this->getEigenMap();

  if (other.isTranspose())
    result -= Eigen::SparseMatrix<scalar>(other.getEigenMap().transpose());
  else
    result -= other.getEigenMap();

  clear();

  for (int k = 0; k < result.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<scalar>::InnerIterator it(result, k); it; ++it)
      setElem(it.value(), (isTranspose() ? it.col() : it.row()), (isTranspose() ? it.row() : it.col()));

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
  if ((getCols() != other.getCols()) && (getRows() != other.getRows()))
    throw dimensionMismatch;

  sparseMatrix<scalar> result = other;
  result += *this;

  return result;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::operator-(const sparseMatrix &other)
{
  if ((getCols() != other.getCols()) && (getRows() != other.getRows()))
    throw dimensionMismatch;

  sparseMatrix<scalar> result = *this;
  result -= other;

  return result;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::operator*(const sparseMatrix &other)
{
  if (getCols() != other.getRows())
    throw dimensionMismatch;

  Eigen::SparseMatrix<scalar> result(getRows(), other.getCols());
  sparseMatrix<scalar> container(getRows(), other.getCols());

  if (!isTranspose() && !other.isTranspose())                    // this * other
    result = (this->getEigenMap() * other.getEigenMap()).pruned(); // prune explicit zeros from the result
  else if (!isTranspose() && other.isTranspose())                // this * other^T
    result = (this->getEigenMap() * other.getEigenMap().transpose()).pruned();
  else if (isTranspose() && other.isTranspose()) // this^T * other^T
    result = (this->getEigenMap().transpose() * other.getEigenMap().transpose()).pruned();
  else if (isTranspose() && !other.isTranspose()) // this^T * other
    result = (this->getEigenMap().transpose() * other.getEigenMap()).pruned();

  for (int k = 0; k < result.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<scalar>::InnerIterator it(result, k); it; ++it)
      container.setElem(it.value(), it.row(), it.col());

  return container;
}

template <class scalar>
denseMatrix<scalar> sparseMatrix<scalar>::operator*(const denseMatrix<scalar> &other)
{
  if (getCols() != other.getRows())
    throw dimensionMismatch;

  denseMatrix<scalar> result(getRows(), other.getCols());

  if (!isTranspose() && !other.isTranspose()) // this * other
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap();
  else if (!isTranspose() && other.isTranspose()) // this * other^T
    result.getEigenMap() = this->getEigenMap() * other.getEigenMap().transpose();
  else if (isTranspose() && other.isTranspose()) // this^T * other^T
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap().transpose();
  else if (isTranspose() && !other.isTranspose()) // this^T * other
    result.getEigenMap() = this->getEigenMap().transpose() * other.getEigenMap();

  return result;
}

template <class scalar>
denseMatrix<scalar> sparseMatrix<scalar>::operator+(const denseMatrix<scalar> &other)
{
  if ((getCols() != other.getCols()) && (getRows() != other.getRows()))
    throw dimensionMismatch;

  denseMatrix<scalar> result(getRows(), getCols());

  if (other.isTranspose())
    result.getEigenMap() = other.getEigenMap().transpose();
  else
    result.getEigenMap() = other.getEigenMap();

  if (isTranspose())
    result.getEigenMap() += this->getEigenMap().transpose();
  else
    result.getEigenMap() += this->getEigenMap();

  return result;
}

template <class scalar>
denseMatrix<scalar> sparseMatrix<scalar>::operator-(const denseMatrix<scalar> &other)
{
  if ((getCols() != other.getCols()) && (getRows() != other.getRows()))
    throw dimensionMismatch;

  denseMatrix<scalar> result(getRows(), getCols());

  if (other.isTranspose())
    result.getEigenMap() = -1 * other.getEigenMap().transpose();
  else
    result.getEigenMap() = -1 * other.getEigenMap();

  if (isTranspose())
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
  tmat.setTranspose();
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
  std::cout << str(); 
}

template <class scalar>
std::string sparseMatrix<scalar>::str()
{
  std::stringstream ss;
  if (!isTranspose())
    ss << this->getEigenMap() << std::endl;
  else
    ss << this->getEigenMap().transpose() << std::endl;

  return ss.str();
}

template <class scalar>
void sparseMatrix<scalar>::printBlock(int startRow, int startCol, int rows, int cols)
{
  if (!isTranspose())
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