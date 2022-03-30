#include <clinpack.d/HyperMatrixDense.hpp>
#include <clinpack.d/HyperMatrixSparse.hpp>
#include <clinpack.d/DenseFactorizationFactory.hpp>

// Converts a C++ vector to a python list
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

// Converts a HyperMatrixDense class to a python list
template <class T>
boost::python::list hyperMatrixToPythonList(HyperMatrixDense<T> matrix) 
{
  boost::python::list list;
  std::vector<T> data = matrix.container();
  list = vectorToPythonList(data);
  return list;
}

// python wrapper of libhypernet module
BOOST_PYTHON_MODULE(libhypernet)
{
  using namespace boost::python;

  // this function converts c++ vector to a python list structure
  def("HyperStateDoubleToList", &vectorToPythonList<double>);

  // wrapper for vector classes to interface with c++ protocols
  class_<std::string>("HyperStateString")
    .def(vector_indexing_suite<std::string>());
  ;

  class_<std::vector<int> >("HyperStateInt")
    .def(vector_indexing_suite<std::vector<int> >());
  ;

  class_<std::vector<double> >("HyperStateDouble")
    .def(vector_indexing_suite<std::vector<double> >());
  ;

  class_<std::vector<long double> >("HyperStateLongDouble")
    .def(vector_indexing_suite<std::vector<long double> >());
  ;

  class_<std::vector<std::complex<double> > >("HyperStateComplexDouble")
    .def(vector_indexing_suite<std::vector<std::complex<double> > >());
  ;

  class_<std::vector<std::complex<long double> > >("HyperStateComplexLongDouble")
    .def(vector_indexing_suite<std::vector<std::complex<long double> > >());
  ;

  typedef HyperMatrixDense<double> HMDDouble;
  class_<HMDDouble>("HyperMatrixDense")
    .def(init< >())
    .def(init<int,int>())
    .def(init<std::vector<double>&,int,int>())
    .def("resize", &HMDDouble::resize)
    .def("setRandom", &HMDDouble::setRandom)
    .def("assign", &HMDDouble::assign)  // operator= not accessable in python
    .def("setElem", &HMDDouble::setElem) 
    .def("getElem", &HMDDouble::getElem)
    .def("norm", &HMDDouble::norm)
    .def("trace", &HMDDouble::trace)
    .def("transpose",  &HMDDouble::transpose)
    .def(self += self)
    .def(self *= int())
    .def(self + self)
    .def(self - self)
    .def(self * self)
    .def("show", &HMDDouble::print) // print() is a reserved sytax in python
    .def("rows", &HMDDouble::get_rows)
    .def("row", &HMDDouble::get_row)
    .def("cols", &HMDDouble::get_cols)
    .def("col", &HMDDouble::get_col)
    .def("diagonal", &HMDDouble::get_diagonal)
    .def("block", &HMDDouble::get_block)
    .def("setRow", &HMDDouble::set_row)
    .def("setCol", &HMDDouble::set_col)
    .def("setDiagonal", &HMDDouble::set_diagonal)
    .def("setBlock", &HMDDouble::set_block)
    .def("save", &HMDDouble::save)
    .def("load", &HMDDouble::load)
  ;

  // this function converts a HyperMatrixDense to a python list structure
  def("HyperMatrixDenseToList", &hyperMatrixToPythonList<double>);

  typedef HyperMatrixSparse<double> HMSDouble;
  class_<HMSDouble>("HyperMatrixSparse")
    .def(init< >())
    .def(init<int,int>())
    .def(init<int,int,int>())
    .def(init<std::vector<double>&,std::vector<int>&,std::vector<int>&,int,int>())
    .def("resize", &HMSDouble::resize)
    .def("reserve", &HMSDouble::reserve)
    .def("assign", &HMSDouble::assign)  // operator= not accessable in python
    .def("setElem", &HMSDouble::setElem) // 
    .def(self += self)
    .def(self *= int())
    .def(self + self)
    .def(self * self)
    .def("show", &HMSDouble::print) // print() is a reserved sytax in python
    .def("save", &HMSDouble::save)
    .def("load", &HMSDouble::load)
  ;

  typedef DenseFactorizationFactory<HMDDouble> HFFactory;
  class_<HFFactory>("HyperDenseDecomposer")
    .def(init< >())
    .def(init<HMDDouble&>())
    .def("reset", &HFFactory::reset)
    .def("computeThinSVD", &HFFactory::computeThinSVD)
    .def("computeQR", &HFFactory::computeQR)
    .def("getU", &HFFactory::getU)
    .def("getSingularValues", &HFFactory::getSingularValues)
    .def("getV", &HFFactory::getV)
    .def("getQ", &HFFactory::getQ)
  ;
}
