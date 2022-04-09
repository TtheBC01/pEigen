#ifndef DENSEFACTORIZATIONFACTORY_HPP
#define DENSEFACTORIZATIONFACTORY_HPP

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

template <class serialType>
class denseFactorizationFactory
{
public:

  denseFactorizationFactory();

  denseFactorizationFactory(serialType &mat);

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

#include "denseFactorizationFactory_impl.hpp"

#endif /*DENSEFACTORIZATIONFACTORY_HPP*/
