#!/bin/bash

# the default build type is Release
# if neccesary, you can rerun the pipeline with another build type-> https://docs.gitlab.com/ee/ci/pipelines.html#manually-executing-pipelines
# to change the build type, you must set the environment variable CUPLA_BUILD_TYPE

if [[ ! -v CUPLA_BUILD_TYPE ]] ; then
    CUPLA_BUILD_TYPE=Release ;
fi

###################################################
# cmake config builder
###################################################

CUPLA_CONST_ARGS=""
CUPLA_CONST_ARGS="${CUPLA_CONST_ARGS} -DCMAKE_BUILD_TYPE=${CUPLA_BUILD_TYPE}"
CUPLA_CONST_ARGS="${CUPLA_CONST_ARGS} ${CUPLA_CMAKE_ARGS}"

CMAKE_CONFIGS=()
for CXX_VERSION in $CUPLA_CXX; do
    for BOOST_VERSION in ${CUPLA_BOOST_VERSIONS}; do
	for ACC in ${ALPAKA_ACCS}; do
	    CMAKE_CONFIGS+=("${CUPLA_CONST_ARGS} -DCMAKE_CXX_COMPILER=${CXX_VERSION} -DBOOST_ROOT=/opt/boost/${BOOST_VERSION} -D${ACC}=ON")
	done
    done
done

###################################################
# build an run tests
###################################################

# use one build directory for all build configurations
mkdir build
cd build

export cupla_DIR=$CI_PROJECT_DIR

# ALPAKA_ACCS contains the backends, which are used for each build
# the backends are set in the sepcialized base jobs .base_gcc,.base_clang and.base_cuda
for CONFIG in $(seq 0 $((${#CMAKE_CONFIGS[*]} - 1))); do
    CMAKE_ARGS=${CMAKE_CONFIGS[$CONFIG]}
    echo -e "\033[0;32m///////////////////////////////////////////////////"
    echo "number of processor threads -> $(nproc)"
    cmake --version | head -n 1
    echo "CMAKE_ARGS -> ${CMAKE_ARGS}"
    echo -e "/////////////////////////////////////////////////// \033[0m \n\n"

    echo "###################################################"
    echo "# Example Matrix Multiplication (adapted original)"
    echo "###################################################"
    echo "can not run with CPU_B_SEQ_T_SEQ due to missing elements layer in original SDK example"
    echo "CPU_B_SEQ_T_OMP2/THREADS too many threads necessary (256)"
    if [[ $CMAKE_ARGS =~ -*DALPAKA_ACC_GPU_CUDA_ENABLE=ON.* ]]; then
        cmake $cupla_DIR/example/CUDASamples/matrixMul/ \
	      $CMAKE_ARGS
        make -j
        time ./matrixMul -wA=64 -wB=64 -hA=64 -hB=64
        rm -r * ;
    fi

    echo "###################################################"
    echo "# Example Async API (adapted original)"
    echo "###################################################"
    echo "can not run with CPU_B_SEQ_T_SEQ due to missing elements layer in original SDK example"
    echo "CPU_B_SEQ_T_OMP2/THREADS too many threads necessary (512)"
    if [[ $CMAKE_ARGS =~ -*DALPAKA_ACC_GPU_CUDA_ENABLE=ON.* ]]; then
        cmake $cupla_DIR/example/CUDASamples/asyncAPI/ \
	      $CMAKE_ARGS
        make -j
        time ./asyncAPI
        rm -r * ;
    fi

    echo "###################################################"
    echo "# Example Async API (added elements layer)"
    echo "###################################################"
    cmake $cupla_DIR/example/CUDASamples/asyncAPI_tuned/ \
	  $CMAKE_ARGS
    make -j
    time ./asyncAPI_tuned
    rm -r *

    echo "###################################################"
    echo "Example vectorAdd (added elements layer)"
    echo "###################################################"
    cmake $cupla_DIR/example/CUDASamples/vectorAdd/ \
	  $CMAKE_ARGS
    make -j
    time ./vectorAdd 100000
    rm -r * ;
done
