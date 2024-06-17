# Developing

## Docker Environment

This repo comes with a Docker [definition](/Dockerfile) file to construct a ubuntu-based development environment that
is guaranteed to compile. You don't need to install dependencies on you local environment, just install the
[Docker engine](https://www.docker.com/) and build the pEigen environment:

```
git clone https://github.com/TtheBC01/pEigen.git
cd pEigen 
git submodule update --init --recursive
docker build -t peigen .
docker run -it --rm --name peigen -v /path/to/pEigen/:/pEigen --entrypoint bash peigen
```
This will start an interactive shell session that comes with gcc, cmake, vim, eigen, and boost dev libraries. 

## Building

Once you are in the interactive Docker shell, build the python bindings like this:

```
pip install . --verbose
```

## Running Unit Tests

pEigen uses Python's [`unittest`](https://docs.python.org/3/library/unittest.html#module-unittest) package for unit testing. 
Test cases are in the `/test` folder. You can run them with Python's built-in CLI:

```shell
python -m unittest test/testDenseMatrix.py
```

## Extending exposed classes and methods

To expose a new object or method in the pEigen library, you must declare it in 
[`/pEigen/src/python_bindings.d/peigen.cxx`](/src/python_bindings.d/peigen.cxx). Specifically, the 
declaration must appear in the `BOOST_PYTHON_MODULE` block. See 
[Boost.Python](https://www.boost.org/doc/libs/1_76_0/libs/python/doc/html/tutorial/tutorial/exposing.html) 
documentation for more details about exposing classes and functions. Place new class definitions in the 
[`/pEigen/src/wrapped_eigen.d/`](/src/wrapped_eigen.d) if you are wrapping a new Eigen object. 

## PyPI

To publish pEigen to the PyPI package repository, use [twine](https://twine.readthedocs.io/en/stable/):

```
python -m build
python -m twine upload --repository testpypi dist/*
```