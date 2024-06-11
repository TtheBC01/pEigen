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
int sparseMatrix<scalar>::get_rows()
{
  int tmp = rows_;
  if (is_transpose())
    tmp = cols_;

  return tmp;
}

template <class scalar>
int sparseMatrix<scalar>::get_cols()
{
  int tmp = cols_;
  if (is_transpose())
    tmp = rows_;

  return tmp;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::get_col(int col)
{

  int start = outer_[col];
  int end = outer_[col+1];
  int nnz = end - start;

  std::vector<int> newInner(nnz);
  std::vector<scalar> newData(nnz);
  for(int i = 0; i < nnz; i++) {
    newInner[i] = inner_[i+start];
    newData[i] = data_[i+start];
  }

  std::vector<int> newOuter(2);
  newOuter[0] = 0;
  newOuter[1] = nnz;

  sparseMatrix newCol(newData, newOuter, newInner, rows_, 1);

  return newCol;
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
  if (rows_ != other.rows() || cols_ != other.cols())
  {
    std::cerr << "Error: Invalid dimensions in matrix addition: lhs.rows != rhs.rows || lhs.cols_ != rhs.cols_" << std::endl;
    abort();
  }

  Eigen::SparseMatrix<scalar> result(rows_, cols_);

  {
    sparseMatrix<scalar> buffer;
    buffer = other;
    Eigen::Map<Eigen::SparseMatrix<scalar>> x(rows_, cols_, nnz_, outer_.data(), inner_.data(), data_.data());
    Eigen::Map<Eigen::SparseMatrix<scalar>> y(buffer.rows(), buffer.cols(), buffer.nnz(), buffer.outerPtr(), buffer.innerPtr(), buffer.data());

    result = x + y;
  }

  clear();

  for (int k = 0; k < result.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<scalar>::InnerIterator it(result, k); it; ++it)
      setElem(it.value(), it.row(), it.col());

  return *this;
}

template <class scalar>
sparseMatrix<scalar> &sparseMatrix<scalar>::operator*=(const double a)
{

  Eigen::Map<Eigen::SparseMatrix<scalar>> x(rows_, cols_, nnz_, outer_.data(), inner_.data(), data_.data());

  x *= a;

  return *this;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::operator+(const sparseMatrix &other)
{
  sparseMatrix<scalar> result = other;
  result += *this;

  return result;
}

template <class scalar>
sparseMatrix<scalar> sparseMatrix<scalar>::operator*(const sparseMatrix &other)
{
  if (rows_ != other.cols())
  {
    std::cerr << "Error: Invalid dimensions in matrix multiply: lhs.rows != rhs.cols" << std::endl;
    abort();
  }

  Eigen::SparseMatrix<scalar> result(rows_, other.cols());
  sparseMatrix<scalar> container(rows_, other.cols());

  {
    sparseMatrix<scalar> buffer;
    buffer = other;
    Eigen::Map<Eigen::SparseMatrix<scalar>> x(rows_, cols_, nnz_, outer_.data(), inner_.data(), data_.data());
    Eigen::Map<Eigen::SparseMatrix<scalar>> y(buffer.rows(), buffer.cols(), buffer.nnz(), buffer.outerPtr(), buffer.innerPtr(), buffer.data());

    result = x * y;
  }

  for (int k = 0; k < result.outerSize(); ++k)
    for (typename Eigen::SparseMatrix<scalar>::InnerIterator it(result, k); it; ++it)
      container.setElem(it.value(), it.row(), it.col());

  return container;
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
  if ((row >= rows_) || (col >= cols_))
  {
    std::cerr << "WARNING: requested element is outside of range" << std::endl;
  }
  else
  {

    // get nnz up to row before insertion row
    if (nnz_ > 0)
    {
      int nnz1 = outer_[col];     // number of slots to skip
      int nnz2 = outer_[col + 1]; // number fo nnzs in col being changed
      std::vector<int>::iterator itInner = inner_.begin() + nnz1;
      std::vector<int>::iterator itInnerStop = inner_.begin() + nnz2;
      typename std::vector<scalar>::iterator itData = data_.begin() + nnz1;
      if (nnz1 != nnz2) // then theres nothing in the column yet
      {
        // find where to insert in the row indices
        for (; itInner != itInnerStop; itInner++, itData++)
          if (*itInner > row)
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
}

template <class scalar>
size_t sparseMatrix<scalar>::size()
{
  return data_.size();
}

template <class scalar>
void sparseMatrix<scalar>::print()
{
  Eigen::Map<Eigen::SparseMatrix<scalar>> x(rows_, cols_, nnz_, outer_.data(), inner_.data(), data_.data());

  if (!is_transpose())
    std::cout << "data =\n"
              << x << std::endl;
  else
    std::cout << "data = \n"
              << x.transpose() << std::endl;
}

template <class scalar>
void sparseMatrix<scalar>::print_block(int startRow, int startCol, int rows, int cols)
{
  Eigen::Map<Eigen::SparseMatrix<scalar>> x(rows_, cols_, nnz_, outer_.data(), inner_.data(), data_.data());

  if (!is_transpose())
    std::cout << "data =\n"
              << x.block(startRow, startCol, rows, cols) << std::endl;
  else
    std::cout << "data = \n"
              << x.transpose().block(startRow, startCol, rows, cols) << std::endl;
}

template <class scalar>
scalar sparseMatrix<scalar>::norm()
{
  Eigen::Map<Eigen::SparseMatrix<scalar>> x(rows_, cols_, nnz_, outer_.data(), inner_.data(), data_.data());

  return x.norm();
}