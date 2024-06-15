template <class serialType>
denseFactorizationFactory<serialType>::denseFactorizationFactory()
{}

template <class serialType>
denseFactorizationFactory<serialType>::denseFactorizationFactory(serialType &mat)
: dense_matrix_(&mat)
{}

template<class serialType>
void denseFactorizationFactory<serialType>::reset(serialType &mat)
{
  dense_matrix_ = &mat;
}

template<class serialType>
void denseFactorizationFactory<serialType>::BDCSVD()
{

  if(dense_matrix_ == NULL)
  {
    std::cerr << "Nothing to decompose..." << std::endl;
  } else {
    Eigen::BDCSVD<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > svd(dense_matrix_->getEigenMap(), Eigen::ComputeThinU | Eigen::ComputeThinV);

    U.resize(svd.matrixU().rows(),svd.matrixU().cols());
    S.resize(svd.singularValues().rows(),svd.singularValues().rows());
    V.resize(svd.matrixV().rows(),svd.matrixV().cols());
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > Ubuf(U.data(),
                                                                             U.rows(),
                                                                             U.cols());
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > Sbuf(S.data(),
                                                                             S.rows(),
                                                                             S.cols());
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > Vbuf(V.data(),
                                                                             V.rows(),
                                                                             V.cols());
    Ubuf = svd.matrixU();
    Sbuf = svd.singularValues().asDiagonal();
    Vbuf = svd.matrixV();
  }

}

template<class serialType>
void denseFactorizationFactory<serialType>::HouseholderQR()
{

  if(dense_matrix_ == NULL)
  {
    std::cerr << "Nothing to decompose..." << std::endl;
  } else {
    Eigen::HouseholderQR<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > qr(dense_matrix_->rows(), dense_matrix_->cols());
    qr.compute(dense_matrix_->getEigenMap());

    Q.resize(qr.householderQ().rows(),qr.householderQ().cols());
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > Qbuf(Q.data(),
                                                                             Q.rows(),
                                                                             Q.cols());
    Qbuf = qr.householderQ();
  }

}

template<class serialType>
void denseFactorizationFactory<serialType>::PartialPivLU()
{

  if(dense_matrix_ == NULL)
  {
    std::cerr << "Nothing to decompose..." << std::endl;
  } else {
	if (dense_matrix_->rows() != dense_matrix_->cols()) {
		std::cerr << "ERROR: LU Decomposition is only for square matrices" << std::endl;
	} else {
      Eigen::PartialPivLU<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > PPLU(dense_matrix_->getEigenMap());

/*       Q.resize(qr.householderQ().rows(),qr.householderQ().cols());
      Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > Qbuf(Q.data(),
                                                                             Q.rows(),
                                                                             Q.cols());
      Qbuf = qr.householderQ(); */
	}
  }

}
