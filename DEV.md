# Developing

## Docker Environment

This repo comes with a Docker [definition](/Dockerfile) file to construct a ubuntu-based development environment that
is guaranteed to compile. You don't need to install dependencies on you local environment, just install the
[Docker engine](https://www.docker.com/) and build the pEigen environment:

```
git clone https://github.com/TtheBC01/pEigen.git
cd pEigen 
docker build -t peigen .
docker run -it --rm --entrypoint bash -v /path/to/pEigen:/pEigen peigen 
```
This will start an interactive shell session that comes with gcc, cmake, vim, eigen, and boost dev libraries. 

## Compiling

Once you are in the interactive Docker shell, build the python bindings like this:

```
cd src
cmake .
make
```

This will produced a shared object library, `libpeigen.so`, in the `/pEigen/src/lib` directory. Import this library into 
a python session as you would any python module.

```
import sys
sys.path.append('/pEigen/src/lib')
import libpeigen
```

## Extending exposed classes and methods

To expose a new object or method in the pEigen library, you must declare it in 
[`/pEigen/src/python_bindings.d/peigen.cxx`](/src/python_bindings.d/peigen.cxx). Specifically, the 
declaration must appear in the `BOOST_PYTHON_MODULE` block. See 
[Boost.Python](https://www.boost.org/doc/libs/1_76_0/libs/python/doc/html/tutorial/tutorial/exposing.html) 
documentation for more details about exposing classes and functions. Place new class definitions in the
[`/pEigen/src/wrapped_eigen.d/`](/src/wrapped_eigen.d) if you are wrapping a new Eigen object. 

## PyPI

```
python -m build
python -m twine upload --repository testpypi dist/*
```