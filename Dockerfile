FROM python:3.9

RUN mkdir /pEigen
WORKDIR /pEigen

COPY /src .

RUN apt update && apt install -y cmake build-essential vim libboost-all-dev
RUN git clone https://gitlab.com/libeigen/eigen.git