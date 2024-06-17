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

  def("double_vec_to_list", &vectorToPythonList<double>);

  def("dense_matrix_to_list", &denseMatrixToPythonList<double>);

  class_<std::vector<double>>("double_vec")
      .def(vector_indexing_suite<std::vector<double>>());

  class_<std::vector<float>>("float_vec")
      .def(vector_indexing_suite<std::vector<float>>());

  typedef denseMatrix<double> dMDouble;
  typedef sparseMatrix<double> sMDouble;

  class_<dMDouble>("dense_matrix")
      .def(init<>())
      .def(init<int, int>())
      .def(init<std::vector<double> &, int, int>())
      .def("resize", &dMDouble::resize)
      .def("set_random", &dMDouble::setRandom)
      .def("assign", &dMDouble::assign) // operator= not accessible in python
      .def("set_elem", &dMDouble::setElem)
      .def("get_elem", &dMDouble::getElem)
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
      .def(self + sMDouble())
      .def(self - sMDouble())
      .def(self * self)
      .def(self * sMDouble())
      .def("show", &dMDouble::print) // print() is a reserved syntax in python
      .def("rows", static_cast<int (dMDouble::*)()>(&dMDouble::get_rows)) // we do this do disambiguate between const and non-const implementations
      .def("cols", static_cast<int (dMDouble::*)()>(&dMDouble::get_cols))
      .def("row", &dMDouble::get_row)
      .def("col", &dMDouble::get_col)
      .def("diagonal", &dMDouble::get_diagonal)
      .def("block", &dMDouble::get_block)
      .def("setRow", &dMDouble::set_row)
      .def("setCol", &dMDouble::set_col)
      .def("set_diagonal", &dMDouble::set_diagonal)
      .def("set_block", &dMDouble::set_block)
      .def("save", &dMDouble::save)
      .def("load", &dMDouble::load);

  class_<sMDouble>("sparse_matrix")
      .def(init<>())
      .def(init<int, int>())
      .def(init<int, int, int>())
      .def(init<std::vector<double> &, std::vector<int> &, std::vector<int> &, int, int>())
      .def("resize", &sMDouble::resize)
      .def("reserve", &sMDouble::reserve)
      .def("assign", &sMDouble::assign)   // operator= not accessible in python
      .def("set_elem", &sMDouble::setElem) 
      .def("get_elem", &sMDouble::getElem)
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
      .def(self - dMDouble())
      .def(self * self)
      .def(self * dMDouble())
      .def("show", &sMDouble::print) // print() is a reserved syntax in python
      .def("show_block", &sMDouble::print_block)
      .def("save", &sMDouble::save)
      .def("load", &sMDouble::load);

  typedef denseFactorizationFactory<dMDouble> dFFactory;
  class_<dFFactory>("factorizer")
      .def(init<>())
      .def(init<dMDouble &>())
      .def("reset", &dFFactory::reset)
      .def("bdcsvd", &dFFactory::BDCSVD)
      .def("householder_qr", &dFFactory::HouseholderQR)
      .def("partial_piv_lu", &dFFactory::PartialPivLU)
      .def("get_u", &dFFactory::getU)
      .def("get_singular_values", &dFFactory::getSingularValues)
      .def("get_v", &dFFactory::getV)
      .def("get_q", &dFFactory::getQ);
}