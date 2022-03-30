template <class scalar>
HyperMatrixDense<scalar>::HyperMatrixDense()
: rows_(0),
  cols_(0),
  transpose_mat(false)
{}

template <class scalar>
HyperMatrixDense<scalar>::HyperMatrixDense(int rows, int cols)
: rows_(rows),
  cols_(cols),
  transpose_mat(false)
{
  resize(rows_,cols_);
}

template <class scalar>
HyperMatrixDense<scalar>::HyperMatrixDense(std::vector<scalar> &data, int rows, int cols)
: data_(data),
  rows_(rows),
  cols_(cols),
  transpose_mat(false)
{}

template <class scalar>
void HyperMatrixDense<scalar>::resize(int rows, int cols)
{
  rows_ = rows;
  cols_ = cols;
  data_.resize(rows_*cols_, 0);
}

template <class scalar>
void HyperMatrixDense<scalar>::setRandom(int seed)
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  std::srand(seed);
  x = Eigen::MatrixXd::Random(x.rows(),x.cols());
}

template <class scalar>
void HyperMatrixDense<scalar>::save(std::string fname)
{
  std::ofstream ofs(fname.c_str());
  boost::archive::text_oarchive oa(ofs);
  oa << *this;
  ofs.close();
}

template <class scalar>
void HyperMatrixDense<scalar>::load(std::string fname)
{ 
  std::ifstream ifs(fname.c_str());
  boost::archive::text_iarchive ia(ifs);
  ia >> *this;
  ifs.close();
}

template <class scalar>
int HyperMatrixDense<scalar>::get_rows()
{
  int tmp = rows_; 
  if(is_transpose())
    tmp = cols_;

  return tmp;
}

template <class scalar>
int HyperMatrixDense<scalar>::get_cols()
{
  int tmp = cols_;
  if(is_transpose())
    tmp = rows_;

  return tmp;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::get_row(int row)
{

  HyperMatrixDense newrow(1,cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(newrow.data(),
                                                                        newrow.rows(),
                                                                        newrow.cols());

  y = x.row(row);

  return newrow;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::get_col(int col)
{
 
  HyperMatrixDense column(rows_,1);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(column.data(),
                                                                        column.rows(),
                                                                        column.cols());

  y = x.col(col);

  return column;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::get_diagonal(int n)
{

  HyperMatrixDense diag(rows_,1);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(diag.data(),
                                                                        diag.rows(),
                                                                        diag.cols());

  y = x.diagonal(n);

  return diag;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::get_block(int row, int col, int nrows, int ncols)
{

  HyperMatrixDense block(nrows,ncols);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(block.data(),
                                                                        block.rows(),
                                                                        block.cols());

  y = x.block(row,col,nrows,ncols);

  return block;
}

template <class scalar>
void HyperMatrixDense<scalar>::set_row(int row, const HyperMatrixDense& rmat)
{

  HyperMatrixDense<scalar> buffer = rmat; 
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());
  
  x.row(row) = y;
}

template <class scalar>
void HyperMatrixDense<scalar>::set_col(int col, const HyperMatrixDense& cmat)
{

  HyperMatrixDense<scalar> buffer = cmat; 
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());

  x.col(col) = y;
}
    
template <class scalar>
void HyperMatrixDense<scalar>::set_diagonal(int d, const HyperMatrixDense& dmat)
{ 

  HyperMatrixDense<scalar> buffer = dmat; 
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());

  x.diagonal(d) = y;
}

template <class scalar>
void HyperMatrixDense<scalar>::set_block(int i, int j, int k, int l, const HyperMatrixDense& bmat)
{

  HyperMatrixDense<scalar> buffer = bmat; 
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());

  x.block(i,j,k,l) = y;
}


template <class scalar>
HyperMatrixDense<scalar>& HyperMatrixDense<scalar>::operator=(const HyperMatrixDense& other)
{
  rows_ = other.rows();
  cols_ = other.cols();
  data_ = other.container();
  transpose_mat = other.is_transpose();
}

template <class scalar>
HyperMatrixDense<scalar>& HyperMatrixDense<scalar>::operator+=(const HyperMatrixDense& other)
{
  HyperMatrixDense<scalar> buffer = other;
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());

  if(!is_transpose() && !other.is_transpose())
    x += y;
  else if(is_transpose() && !other.is_transpose())
    x.transpose() += y;
  else if (is_transpose() && other.is_transpose())
    x.transpose() += y.transpose();
  else if (!is_transpose() && other.is_transpose())
    x += y.transpose();


  return *this;
}

template <class scalar>
HyperMatrixDense<scalar>& HyperMatrixDense<scalar>::operator-=(const HyperMatrixDense& other)
{
  HyperMatrixDense<scalar> buffer;
  buffer = other;
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());

  if(!is_transpose() && !other.is_transpose())
    x -= y;
  else if(is_transpose() && !other.is_transpose())
    x.transpose() -= y;
  else if (is_transpose() && other.is_transpose())
    x.transpose() -= y.transpose();
  else if (!is_transpose() && other.is_transpose())
    x -= y.transpose();

  return *this;
}

template <class scalar>
HyperMatrixDense<scalar>& HyperMatrixDense<scalar>::operator*=(const double a)
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);
  x *= a;

  return *this;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::operator+(const HyperMatrixDense& other)
{
  HyperMatrixDense<scalar> result = other;
  result += *this;

  return result;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::operator-(const HyperMatrixDense& other)
{
  HyperMatrixDense<scalar> result = *this;
  result -= other;

  return result;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::operator*(const HyperMatrixDense& other)
{

  // size the result properly
  HyperMatrixDense<scalar> result;
  if(!is_transpose() && !other.is_transpose())
    result.resize(rows_,other.cols());
  else if(is_transpose() && !other.is_transpose())
    result.resize(cols_,other.cols());
  else if (is_transpose() && other.is_transpose())
    result.resize(cols_,other.rows());
  else if (!is_transpose() && other.is_transpose())
    result.resize(rows_,other.rows());

  HyperMatrixDense<scalar> buffer;
  buffer = other;

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(data_.data(),
                                                                        rows_,
                                                                        cols_);
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > z(result.data(),
                                                                        result.rows(),
                                                                        result.cols());

  if(!is_transpose() && !other.is_transpose())
    z = x*y;
  else if(is_transpose() && !other.is_transpose())
    z = x.transpose()*y;
  else if (is_transpose() && other.is_transpose())
    z = x.transpose()*y.transpose();
  else if (!is_transpose() && other.is_transpose())
    z = x*y.transpose();

  
  return result;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::operator*(const double a)
{
  HyperMatrixDense<scalar> result;
  result = *this;
  result *= a;
  return result;
}

template <class scalar>
HyperMatrixDense<scalar> HyperMatrixDense<scalar>::transpose()
{
  HyperMatrixDense<scalar> tmat(data_,rows_,cols_);
  tmat.set_transpose();
  return tmat;
}

template <class scalar>
void HyperMatrixDense<scalar>::assign(const HyperMatrixDense& other)
{
  *this = other;
}

template <class scalar>
void HyperMatrixDense<scalar>::setElem(scalar elem, int row, int col)
{
  if((row >= rows_) || (col >= cols_))
    std::cerr << "WARNING: requested element is outside of range" << std::endl;
  else
    (*this)(row,col) = elem; 
}

template <class scalar>
scalar HyperMatrixDense<scalar>::norm()
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(data_.data(),
                                                                        rows_,
                                                                        cols_);

  return x.norm();
}

template <class scalar>
scalar HyperMatrixDense<scalar>::trace()
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(data_.data(),
                                                                        rows_,
                                                                        cols_);

  return x.trace();
}

template <class scalar>
size_t HyperMatrixDense<scalar>::size()
{
  return data_.size();
}

template <class scalar>
void HyperMatrixDense<scalar>::print()
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(data_.data(), 
                                                                        rows_, 
                                                                        cols_);
  if(!is_transpose())
    std::cout << "data =\n" << x << std::endl;
  else
    std::cout << "data = \n" << x.transpose() << std::endl;
}
