#!/bin/bash

# This script expects the following parameters:
# - the path where the library and the corresponding dependencies are installed
# - a string containing a semicolon-separated list (e.g. "2;8;19") with the number of classes to be used to compile the library
# - a path to the cuda installation to be used to compile the library [optional]

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Script expects 2 - 3 parameters, but ${#} provided!" >&2
    echo "Usage: $0 <INSTALL_DIR> <CLASSES_NUMS> [<CUDA_DIR>]"
    exit 2
fi

INSTALL_DIR=$1
CLASSES_NUMS=$2

read -r -d '' installation_parameters <<-EOF
Provided parameters:
    INSTALL_DIR: ${INSTALL_DIR}
    CLASSES_NUMS: ${CLASSES_NUMS}
    CUDA_DIR: $3
EOF
echo "$installation_parameters"

mkdir -p $INSTALL_DIR
cd $INSTALL_DIR


#####################################
# Download, build and install CPython
#####################################

CPYTHON_SOURCE_DIR="${INSTALL_DIR}/cpython"
CPYTHON_BUILD_DIR="${INSTALL_DIR}/cpython_build"
CPYTHON_INSTALL_DIR="${INSTALL_DIR}/cpython_install"
CPYTHON_CONFIGURE_FILE="${CPYTHON_SOURCE_DIR}/configure"
CPYTHON_BIN_DIR=${CPYTHON_INSTALL_DIR}/bin
CPYTHON_LIB_DIR=${CPYTHON_INSTALL_DIR}/lib

git clone --depth 1 --branch v3.7.9 https://github.com/python/cpython $CPYTHON_SOURCE_DIR

mkdir -p $CPYTHON_BUILD_DIR
cd $CPYTHON_BUILD_DIR
$CPYTHON_CONFIGURE_FILE --prefix ${CPYTHON_INSTALL_DIR} --enable-optimizations --with-ensurepip=install --enable-shared
make -j
make install -j

export LD_LIBRARY_PATH=$CPYTHON_LIB_DIR:$LD_LIBRARY_PATH
export PATH=$CPYTHON_BIN_DIR:$PATH
export PYTHONNOUSERSITE=1

python3 -m pip install --upgrade pip
pip3 install wheel
pip install numpy==1.19.2 tensorflow-gpu==2.4


###################################
# Download, build and install Boost
###################################

BOOST_BUILD_DIR="${INSTALL_DIR}/boost_build"
BOOST_INSTALL_DIR="${INSTALL_DIR}/boost_install"
BOOST_TAR_BZ2="${BOOST_BUILD_DIR}/boost_1_75_0.tar.bz2"

cd $INSTALL_DIR
mkdir -p $BOOST_BUILD_DIR

wget https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.bz2 -O $BOOST_TAR_BZ2
tar -xf $BOOST_TAR_BZ2 --directory $BOOST_BUILD_DIR --strip-components 1

export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:${CPYTHON_INSTALL_DIR}/include/python3.7m"
cd ${BOOST_BUILD_DIR}

./bootstrap.sh --with-python=python3 --prefix=${BOOST_INSTALL_DIR}
./b2 --build-type=minimal --build-dir=${BOOST_BUILD_DIR} --prefix=${BOOST_INSTALL_DIR} toolset=gcc --layout=system stage install

export BOOST_ROOT=${BOOST_INSTALL_DIR}
export BOOST_LIBRARYDIR=${BOOST_INSTALL_DIR}/lib


###############################################
# # Download, build and install Semantic Meshes
###############################################

SEMANTIC_SOURCE_DIR="${INSTALL_DIR}/semantic-meshes"
SEMANTIC_BUILD_DIR="${INSTALL_DIR}/semantic-meshes_build"
SEMANTIC_INSTALL_DIR="${INSTALL_DIR}/semantic-meshes_install"

if ! [ -z $3 ]; then
  CUDA_DIR=$3
  export LD_LIBRARY_PATH=${CUDA_DIR}/lib64:$LD_LIBRARY_PATH
  export PATH=${CUDA_DIR}/bin:$PATH
fi

cd $INSTALL_DIR
git clone https://github.com/fferflo/semantic-meshes $SEMANTIC_SOURCE_DIR

mkdir -p $SEMANTIC_BUILD_DIR
cd $SEMANTIC_BUILD_DIR
cmake -DCMAKE_INSTALL_PREFIX=${SEMANTIC_INSTALL_DIR} -DCLASSES_NUMS=${CLASSES_NUMS} $SEMANTIC_SOURCE_DIR -DCMAKE_PREFIX_PATH="${CPYTHON_INSTALL_DIR};${BOOST_INSTALL_DIR}"
make -j
make install
pip install ${SEMANTIC_BUILD_DIR}/python


##############
# Print usage
##############

read -r -d '' library_usage <<-EOF
Installed semantic-meshes to:
    ${CPYTHON_INSTALL_DIR}
Execute the following commands to run one of the python examples shipped with semantic-meshes:
    export PATH=$CPYTHON_BIN_DIR:\$PATH
    export LD_LIBRARY_PATH=$CPYTHON_LIB_DIR:\$LD_LIBRARY_PATH
    if command -v conda &> /dev/null
    	conda deactivate
    python3 $SEMANTIC_SOURCE_DIR/python/scripts/colorize_cityscapes_mesh.py --colmap /path/to/colmap/dense/sparse --input_ply /path/to/colmap/dense/meshed-delaunay.ply --images /path/to/colmap/dense/images --output_ply /path/to/colorized_mesh.ply
EOF
echo "$library_usage"