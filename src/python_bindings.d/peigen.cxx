#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <wrapped_eigen.d/dense.d/denseMatrix.hpp>
#include <wrapped_eigen.d/sparse.d/sparseMatrix.hpp>
#include <wrapped_eigen.d/factorization.d/denseFactorizationFactory.hpp>

// python wrapper of libpeigen module
BOOST_PYTHON_MODULE(libpeigen)
{
  using namespace boost::python;

  typedef denseMatrix<double> dMDouble;
  typedef sparseMatrix<double> sMDouble;

  class_<dMDouble>("dense_matrix")
      .def(init<>())
      .def(init<int, int>())
      .def(init<boost::python::list &, int, int>())
      .def("__repr__", &dMDouble::repr)
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
      .def("rows", static_cast<int (dMDouble::*)()>(&dMDouble::getRows)) // we do this do disambiguate between const and non-const implementations
      .def("cols", static_cast<int (dMDouble::*)()>(&dMDouble::getCols))
      .def("row", &dMDouble::getRow)
      .def("col", &dMDouble::getCol)
      .def("__getitem__", &dMDouble::getRowOrCol)
      .def("diagonal", &dMDouble::getDiagonal)
      .def("block", &dMDouble::getBlock)
      .def("setRow", &dMDouble::setRow)
      .def("setCol", &dMDouble::setCol)
      .def("set_diagonal", &dMDouble::setDiagonal)
      .def("set_block", &dMDouble::setBlock)
      .def("save", &dMDouble::save)
      .def("load", &dMDouble::load)
      .def("to_list", &dMDouble::toList);

  class_<sMDouble>("sparse_matrix")
      .def(init<>())
      .def(init<int, int>())
      .def(init<boost::python::list &, boost::python::list &, boost::python::list &, int, int>())
      .def("__repr__", &sMDouble::repr)
      .def("resize", &sMDouble::resize)
      .def("assign", &sMDouble::assign)   // operator= not accessible in python
      .def("set_elem", &sMDouble::setElem) 
      .def("get_elem", &sMDouble::getElem)
      .def("norm", &sMDouble::norm)
      .def("nnz", static_cast<int (sMDouble::*)()>(&sMDouble::nnz))
      .def("rows", static_cast<int (sMDouble::*)()>(&sMDouble::getRows))
      .def("cols", static_cast<int (sMDouble::*)()>(&sMDouble::getCols))
      .def("col", &sMDouble::getCol)
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
      .def("show_block", &sMDouble::printBlock)
      .def("save", &sMDouble::save)
      .def("load", &sMDouble::load)
      .def("data_to_list", &sMDouble::dataToList)
      .def("inner_to_list", &sMDouble::innerToList)
      .def("outer_to_list", &sMDouble::outerToList);

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