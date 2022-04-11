template <class scalar>
denseMatrix<scalar>::denseMatrix()
: rows_(0),
  cols_(0),
  transpose_mat(false)
{}

template <class scalar>
denseMatrix<scalar>::denseMatrix(int rows, int cols)
: rows_(rows),
  cols_(cols),
  transpose_mat(false)
{
  resize(rows_,cols_);
}

template <class scalar>
denseMatrix<scalar>::denseMatrix(std::vector<scalar> &data, int rows, int cols)
: data_(data),
  rows_(rows),
  cols_(cols),
  transpose_mat(false)
{}

template <class scalar>
void denseMatrix<scalar>::resize(int rows, int cols)
{
  rows_ = rows;
  cols_ = cols;
  data_.resize(rows_*cols_, 0);
}

template <class scalar>
void denseMatrix<scalar>::setRandom(int seed)
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  std::srand(seed);
  x = Eigen::MatrixXd::Random(x.rows(),x.cols());
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
int denseMatrix<scalar>::get_rows()
{
  int tmp = rows_; 
  if(is_transpose())
    tmp = cols_;

  return tmp;
}

template <class scalar>
int denseMatrix<scalar>::get_cols()
{
  int tmp = cols_;
  if(is_transpose())
    tmp = rows_;

  return tmp;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::get_row(int row)
{

  denseMatrix newrow(1,cols_);

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
denseMatrix<scalar> denseMatrix<scalar>::get_col(int col)
{
 
  denseMatrix column(rows_,1);

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
denseMatrix<scalar> denseMatrix<scalar>::get_diagonal(int n)
{

  denseMatrix diag(rows_,1);

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
denseMatrix<scalar> denseMatrix<scalar>::get_block(int row, int col, int nrows, int ncols)
{

  denseMatrix block(nrows,ncols);

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
void denseMatrix<scalar>::set_row(int row, const denseMatrix& rmat)
{

  denseMatrix<scalar> buffer = rmat; 
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());
  
  x.row(row) = y;
}

template <class scalar>
void denseMatrix<scalar>::set_col(int col, const denseMatrix& cmat)
{

  denseMatrix<scalar> buffer = cmat; 
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());

  x.col(col) = y;
}
    
template <class scalar>
void denseMatrix<scalar>::set_diagonal(int d, const denseMatrix& dmat)
{ 

  denseMatrix<scalar> buffer = dmat; 
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());

  x.diagonal(d) = y;
}

template <class scalar>
void denseMatrix<scalar>::set_block(int i, int j, int k, int l, const denseMatrix& bmat)
{

  denseMatrix<scalar> buffer = bmat; 
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);

  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > y(buffer.data(),
                                                                        buffer.rows(),
                                                                        buffer.cols());

  x.block(i,j,k,l) = y;
}


template <class scalar>
denseMatrix<scalar>& denseMatrix<scalar>::operator=(const denseMatrix& other)
{
  rows_ = other.rows();
  cols_ = other.cols();
  data_ = other.container();
  transpose_mat = other.is_transpose();
  return *this;
}

template <class scalar>
denseMatrix<scalar>& denseMatrix<scalar>::operator+=(const denseMatrix& other)
{
  denseMatrix<scalar> buffer = other;
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
denseMatrix<scalar>& denseMatrix<scalar>::operator-=(const denseMatrix& other)
{
  denseMatrix<scalar> buffer;
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
denseMatrix<scalar>& denseMatrix<scalar>::operator*=(const double a)
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(this->data(),
                                                                        rows_,
                                                                        cols_);
  x *= a;

  return *this;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator+(const denseMatrix& other)
{
  denseMatrix<scalar> result = other;
  result += *this;

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator-(const denseMatrix& other)
{
  denseMatrix<scalar> result = *this;
  result -= other;

  return result;
}

template <class scalar>
denseMatrix<scalar> denseMatrix<scalar>::operator*(const denseMatrix& other)
{

  // size the result properly
  denseMatrix<scalar> result;
  if(!is_transpose() && !other.is_transpose()) {
	  if (cols_ != other.rows()) {
		  std::cerr << "ERROR: inner dimension mismatch:" << cols_ << " != " << other.rows() << std::endl;
	      return result;
	  } else {
          result.resize(rows_,other.cols());
	  }
  } else if(is_transpose() && !other.is_transpose()) {
	  if (rows_ != other.rows()) {
		  std::cerr << "ERROR: inner dimension mismatch:" << rows_ << " != " << other.rows() << std::endl;
	      return result;
	  } else {
          result.resize(cols_,other.cols());
	  }
  } else if (is_transpose() && other.is_transpose()) {
	  if (rows_ != other.cols()) {
		  std::cerr << "ERROR: inner dimension mismatch:" << rows_ << " != " << other.cols() << std::endl;
	      return result;
	  } else {
          result.resize(cols_,other.rows());
	  }
  } else if (!is_transpose() && other.is_transpose()) {
	  if (cols_ != other.cols()) {
		  std::cerr << "ERROR: inner dimension mismatch:" << cols_ << " != " << other.cols() << std::endl;
	      return result;
	  } else {
          result.resize(rows_,other.rows());
	  }
  }

  denseMatrix<scalar> buffer;
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
    z.noalias() = x*y;
  else if(is_transpose() && !other.is_transpose())
    z.noalias() = x.transpose()*y;
  else if (is_transpose() && other.is_transpose())
    z.noalias() = x.transpose()*y.transpose();
  else if (!is_transpose() && other.is_transpose())
    z.noalias() = x*y.transpose();

  
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
  denseMatrix<scalar> tmat(data_,rows_,cols_);
  tmat.set_transpose();
  return tmat;
}

template <class scalar>
void denseMatrix<scalar>::assign(const denseMatrix& other)
{
  *this = other;
}

template <class scalar>
void denseMatrix<scalar>::setElem(scalar elem, int row, int col)
{
  if((row >= rows_) || (col >= cols_))
    std::cerr << "WARNING: requested element is outside of range" << std::endl;
  else
    (*this)(row,col) = elem; 
}

template <class scalar>
scalar denseMatrix<scalar>::norm()
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(data_.data(),
                                                                        rows_,
                                                                        cols_);

  return x.norm();
}

template <class scalar>
scalar denseMatrix<scalar>::trace()
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(data_.data(),
                                                                        rows_,
                                                                        cols_);

  return x.trace();
}

template <class scalar>
size_t denseMatrix<scalar>::size()
{
  return data_.size();
}

template <class scalar>
void denseMatrix<scalar>::print()
{
  Eigen::Map< Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> > x(data_.data(), 
                                                                        rows_, 
                                                                        cols_);
  if(!is_transpose())
    std::cout << "data =\n" << x << std::endl;
  else
    std::cout << "data = \n" << x.transpose() << std::endl;
}
