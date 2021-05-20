FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

# Install cmake: https://apt.kitware.com/
RUN apt-get update -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-get update -y
RUN apt-get install -y cmake

# Install other dependencies
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y build-essential git libomp-dev git cmake python3 python3-pip libboost1.71-all-dev
RUN pip3 install tensorflow-gpu==2.4 numpy==1.19.2

# Install semantic-meshes
RUN git clone https://github.com/fferflo/semantic-meshes
RUN mkdir /semantic-meshes/build
WORKDIR /semantic-meshes/build
RUN cmake -DCLASSES_NUMS=19 -DBUILD_PYTHON_INTERFACE=ON ..
RUN make -j8 && make install
RUN pip3 install ./python
