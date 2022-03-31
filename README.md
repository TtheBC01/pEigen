# pEigen
Python bindings for Eigen Tux library using Boost Python

## build Docker dev environment

```
git clone https://github.com/TtheBC01/pEigen.git
cd pEigen 
docker build -t peigen .
docker run -it --rm --entrypoint bash -v /path/to/pEigen/src:/pEigen peigen 
```