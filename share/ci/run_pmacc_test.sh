#!/bin/bash

set -e
set -o pipefail

# the default build type is Release
# if neccesary, you can rerun the pipeline with another build type-> https://docs.gitlab.com/ee/ci/pipelines.html#manually-executing-pipelines
# to change the build type, you must set the environment variable PMACC_BUILD_TYPE
if [[ ! -v PMACC_BUILD_TYPE ]] ; then
    PMACC_BUILD_TYPE=Release;
fi

###################################################
# cmake config builder
###################################################

PMACC_CONST_ARGS=""
# to save compile time reduce the isaac functor chain length to one
PMACC_CONST_ARGS="${PMACC_CONST_ARGS} -DCMAKE_BUILD_TYPE=${PMACC_BUILD_TYPE}"
CMAKE_ARGS="${PMACC_CONST_ARGS} ${PIC_CMAKE_ARGS} -DCMAKE_CXX_COMPILER=${CXX_VERSION} -DBOOST_ROOT=/opt/boost/${BOOST_VERSION}"

# workaround for clang cuda
# HDF5 from the apt sources is pulling -D_FORTIFY_SOURCE=2 into the compile flags
# this workaround is creating a warning about the double definition of _FORTIFY_SOURCE
#
# Workaround will be removed after the test container are shipped with a self compiled HDF5
if [[ $CXX_VERSION =~ ^clang && $PMACC_BACKEND =~ ^cuda ]] ; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_FLAGS=-D_FORTIFY_SOURCE=0"
fi

###################################################
# build an run tests
###################################################

# use one build directory for all build configurations
cd $HOME
mkdir buildPMaccCI
cd buildPMaccCI

export code_DIR=$CI_PROJECT_DIR

PMACC_PARALLEL_BUILDS=$(nproc)
# limit to $CI_MAX_PARALLELISM parallel builds to avoid out of memory errors
# CI_MAX_PARALLELISM is a configured variable in the CI web interface
if [ $PMACC_PARALLEL_BUILDS -gt $CI_MAX_PARALLELISM ] ; then
    PMACC_PARALLEL_BUILDS=$CI_MAX_PARALLELISM
fi
echo -e "\033[0;32m///////////////////////////////////////////////////"
echo "number of processor threads -> $(nproc)"
echo "number of parallel builds -> $PMACC_PARALLEL_BUILDS"
echo "cmake version   -> $(cmake --version | head -n 1)"
echo "build directory -> $(pwd)"
echo "CMAKE_ARGS      -> ${CMAKE_ARGS}"
echo "accelerator     -> ${PIC_BACKEND}"
echo "input set       -> ${PIC_TEST_CASE_FOLDER}"
echo -e "/////////////////////////////////////////////////// \033[0m \n\n"

if [ "$PIC_TEST_CASE_FOLDER" == "examples/" ] || [ "$PIC_TEST_CASE_FOLDER" == "tests/" ] ; then
    extended_compile_options="-l"
fi

cmake $CMAKE_ARGS $code_DIR/include/pmacc
make -j $PMACC_PARALLEL_BUILDS
make test
