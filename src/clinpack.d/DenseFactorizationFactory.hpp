#ifndef DENSEFACTORIZATIONFACTORY_HPP
#define DENSEFACTORIZATIONFACTORY_HPP

#include <iostream>

#include <boost/python.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

template <class serialType>
class DenseFactorizationFactory
{
public:

  DenseFactorizationFactory();

  DenseFactorizationFactory(serialType &mat);

  void reset(serialType &mat);

  void computeThinSVD();

  void computeQR();

  serialType getU() { return U; }

  serialType getSingularValues() { return S; }
 
  serialType getV() { return V; }

  serialType getQ() { return Q; }

private:

  serialType *dense_matrix_ = NULL;

  // containers to store factorizations
  serialType U;
  serialType S;
  serialType V;
  serialType Q;

};

#include "DenseFactorizationFactory_impl.hpp"

#endif /*DENSEFACTORIZATIONFACTORY_HPP*/
