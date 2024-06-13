#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <wrapped_eigen.d/dense.d/denseMatrix.hpp>
#include <wrapped_eigen.d/sparse.d/sparseMatrix.hpp>
#include <wrapped_eigen.d/factorization.d/denseFactorizationFactory.hpp>

// Copies a C++ vector into a python list container
template <class T>
boost::python::list vectorToPythonList(std::vector<T> vector)
{
  typename std::vector<T>::iterator iter;
  boost::python::list list;
  for (iter = vector.begin(); iter != vector.end(); ++iter)
  {
    list.append(*iter);
  }
  return list;
}

// Converts a denseMatrix class to a python list
template <class T>
boost::python::list denseMatrixToPythonList(denseMatrix<T> matrix)
{
  boost::python::list list;
  std::vector<T> data = matrix.container();
  list = vectorToPythonList(data);
  return list;
}

// python wrapper of libpeigen module
BOOST_PYTHON_MODULE(libpeigen)
{
  using namespace boost::python;

  def("doubleVecToList", &vectorToPythonList<double>);

  def("denseMatrixToList", &denseMatrixToPythonList<double>);

  class_<std::vector<double>>("doubleVec")
      .def(vector_indexing_suite<std::vector<double>>());

  class_<std::vector<float>>("floatVec")
      .def(vector_indexing_suite<std::vector<float>>());

  typedef denseMatrix<double> dMDouble;
  typedef sparseMatrix<double> sMDouble;

  class_<dMDouble>("denseMatrixDouble")
      .def(init<>())
      .def(init<int, int>())
      .def(init<std::vector<double> &, int, int>())
      .def("resize", &dMDouble::resize)
      .def("setRandom", &dMDouble::setRandom)
      .def("assign", &dMDouble::assign) // operator= not accessible in python
      .def("setElem", &dMDouble::setElem)
      .def("getElem", &dMDouble::getElem)
      .def("norm", &dMDouble::norm)
      .def("trace", &dMDouble::trace)
      .def("transpose", &dMDouble::transpose)
      .def(self += self)
      .def(self *= int())
      .def(self * int())
      .def(self *= float())
      .def(self * float())
      .def(self + self)
      .def(self - self)
      .def(self * self)
      .def(self * sMDouble())
      .def("show", &dMDouble::print) // print() is a reserved syntax in python
      .def("rows", static_cast<int (dMDouble::*)()>(&dMDouble::get_rows))
      .def("cols", static_cast<int (dMDouble::*)()>(&dMDouble::get_cols))
      .def("row", &dMDouble::get_row)
      .def("col", &dMDouble::get_col)
      .def("diagonal", &dMDouble::get_diagonal)
      .def("block", &dMDouble::get_block)
      .def("setRow", &dMDouble::set_row)
      .def("setCol", &dMDouble::set_col)
      .def("setDiagonal", &dMDouble::set_diagonal)
      .def("setBlock", &dMDouble::set_block)
      .def("save", &dMDouble::save)
      .def("load", &dMDouble::load);

  class_<sMDouble>("sparseMatrixDouble")
      .def(init<>())
      .def(init<int, int>())
      .def(init<int, int, int>())
      .def(init<std::vector<double> &, std::vector<int> &, std::vector<int> &, int, int>())
      .def("resize", &sMDouble::resize)
      .def("reserve", &sMDouble::reserve)
      .def("assign", &sMDouble::assign)   // operator= not accessible in python
      .def("setElem", &sMDouble::setElem) 
      .def("norm", &sMDouble::norm)
      .def("nnz", static_cast<int (sMDouble::*)()>(&sMDouble::nnz))
      .def("rows", static_cast<int (sMDouble::*)()>(&sMDouble::get_rows))
      .def("cols", static_cast<int (sMDouble::*)()>(&sMDouble::get_cols))
      .def("col", &sMDouble::get_col)
      .def("transpose", &sMDouble::transpose)
      .def(self += self)
      .def(self -= self)
      .def(self *= int())
      .def(self * int())
      .def(self *= float())
      .def(self * float())
      .def(self + self)
      .def(self - self)
      .def(self + dMDouble())
      .def(self * self)
      .def(self * dMDouble())
      .def("show", &sMDouble::print) // print() is a reserved syntax in python
      .def("show_block", &sMDouble::print_block)
      .def("save", &sMDouble::save)
      .def("load", &sMDouble::load);

  typedef denseFactorizationFactory<dMDouble> dFFactory;
  class_<dFFactory>("denseDecomposition")
      .def(init<>())
      .def(init<dMDouble &>())
      .def("reset", &dFFactory::reset)
      .def("BDCSVD", &dFFactory::BDCSVD)
      .def("HouseholderQR", &dFFactory::HouseholderQR)
      .def("PartialPivLU", &dFFactory::PartialPivLU)
      .def("getU", &dFFactory::getU)
      .def("getSingularValues", &dFFactory::getSingularValues)
      .def("getV", &dFFactory::getV)
      .def("getQ", &dFFactory::getQ);
}