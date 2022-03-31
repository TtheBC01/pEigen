#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

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
}