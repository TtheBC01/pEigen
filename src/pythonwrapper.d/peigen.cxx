#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <clinpack.d/denseMatrix.hpp>
#include <clinpack.d/sparseMatrix.hpp>

// Copies a C++ vector into a python list container
template <class T>
boost::python::list vectorToPythonList(std::vector<T> vector) 
{
  typename std::vector<T>::iterator iter;
  boost::python::list list;
  for (iter = vector.begin(); iter != vector.end(); ++iter) {
      list.append(*iter);
  }
  return list;
}

// python wrapper of libpeigen module
BOOST_PYTHON_MODULE(libpeigen)
{
  using namespace boost::python;
  
  // this function converts c++ vector to a python list structure
  def("doubleVecToList", &vectorToPythonList<double>);
  
  class_<std::vector<double> >("doubleVec")
    .def(vector_indexing_suite<std::vector<double> >())
  ;
  
  class_<std::vector<float> >("floatVec")
    .def(vector_indexing_suite<std::vector<float> >())
  ;
  
  typedef denseMatrix<double> dMDouble;
  class_<dMDouble>("denseMatrixDouble")
    .def(init< >())
    .def(init<int,int>())
    .def(init<std::vector<double>&,int,int>())
    .def("resize", &dMDouble::resize)
    .def("setRandom", &dMDouble::setRandom)
    .def("assign", &dMDouble::assign)  // operator= not accessable in python
    .def("setElem", &dMDouble::setElem) 
    .def("getElem", &dMDouble::getElem)
    .def("norm", &dMDouble::norm)
    .def("trace", &dMDouble::trace)
    .def("transpose",  &dMDouble::transpose)
    .def(self += self)
    .def(self *= int())
    .def(self + self)
    .def(self - self)
    .def(self * self)
    .def("show", &dMDouble::print) // print() is a reserved sytax in python
    .def("rows", &dMDouble::get_rows)
    .def("row", &dMDouble::get_row)
    .def("cols", &dMDouble::get_cols)
    .def("col", &dMDouble::get_col)
    .def("diagonal", &dMDouble::get_diagonal)
    .def("block", &dMDouble::get_block)
    .def("setRow", &dMDouble::set_row)
    .def("setCol", &dMDouble::set_col)
    .def("setDiagonal", &dMDouble::set_diagonal)
    .def("setBlock", &dMDouble::set_block)
    .def("save", &dMDouble::save)
    .def("load", &dMDouble::load)
  ;
  
  typedef sparseMatrix<double> sMDouble;
  class_<sMDouble>("sparseMatrixDouble")
    .def(init< >())
    .def(init<int,int>())
    .def(init<int,int,int>())
    .def(init<std::vector<double>&,std::vector<int>&,std::vector<int>&,int,int>())
    .def("resize", &sMDouble::resize)
    .def("reserve", &sMDouble::reserve)
    .def("assign", &sMDouble::assign)  // operator= not accessable in python
    .def("setElem", &sMDouble::setElem) // 
    .def(self += self)
    .def(self *= int())
    .def(self + self)
    .def(self * self)
    .def("show", &sMDouble::print) // print() is a reserved sytax in python
    .def("save", &sMDouble::save)
    .def("load", &sMDouble::load)
  ;
}