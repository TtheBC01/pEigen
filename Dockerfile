FROM python:3.9

RUN apt update && \
    apt install -y cmake \
	               build-essential \
				   vim \ 
				   libboost-all-dev\ 
				   clang-tidy
RUN git clone https://gitlab.com/libeigen/eigen.git

RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade build
RUN python -m pip install --upgrade twine

RUN mkdir /pEigen
WORKDIR /pEigen