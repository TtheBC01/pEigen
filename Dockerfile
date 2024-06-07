FROM python:3.12

RUN apt update && \
    apt install -y cmake \
	               build-essential \
				   vim \ 
				   libboost-all-dev\ 
				   clang-tidy

RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade build
RUN python -m pip install --upgrade twine

RUN mkdir /pEigen
WORKDIR /pEigen