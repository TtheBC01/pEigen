# pEigen: Eigen for Python
Python bindings for [Eigen Tux](https://eigen.tuxfamily.org/) library using [Boost.Python](https://github.com/boostorg/python).

## Developing

This repo comes with a Docker definition file to construct a ubuntu-based development environment that
is guaranteed to compile. You don't need to install dependencies on you local environment, just install the
[Docker engine](https://www.docker.com/) and build the pEigen environment:

```
git clone https://github.com/TtheBC01/pEigen.git
cd pEigen 
docker build -t peigen .
docker run -it --rm --entrypoint bash -v /path/to/pEigen/src:/pEigen peigen 
```