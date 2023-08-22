#!/bin/bash

# the default build type is Release
# if neccesary, you can rerun the pipeline with another build type-> https://docs.gitlab.com/ee/ci/pipelines.html#manually-executing-pipelines
# to change the build type, you must set the environment variable CUPLA_BUILD_TYPE

if [[ ! -v CUPLA_BUILD_TYPE ]] ; then
    CUPLA_BUILD_TYPE=Release ;
fi

###################################################
# update environment
###################################################
PATH=$(agc-manager -b cmake@3.22)/bin:$PATH

###################################################
# cmake config builder
###################################################

# create a cmake variable definition if an environment variable exists
#
# This function can not handle environment variables with spaces in its content.
#
# @param $1 cmake/environment variable name
#
# @result if $1 exists cmake variable definition else nothing is returned
#
# @code{.bash}
# FOO=ON
# echo "$(env2cmake FOO)" # returns "-DFOO=ON"
# echo "$(env2cmake BAR)" # returns nothing
# @endcode
function env2cmake()
{
    if [ ! -z "${!1}" ] ; then
        echo -n "-D$1=${!1}"
    fi
}

CUPLA_CONST_ARGS="$(env2cmake CMAKE_CUDA_ARCHITECTURES) $(env2cmake CMAKE_CXX_EXTENSIONS)  -Dcupla_BUILD_EXAMPLES=ON -Dcupla_ALPAKA_PROVIDER=internal"
CUPLA_CONST_ARGS="${CUPLA_CONST_ARGS} -DCMAKE_BUILD_TYPE=${CUPLA_BUILD_TYPE}"
CUPLA_CONST_ARGS="${CUPLA_CONST_ARGS} ${CUPLA_CMAKE_ARGS}"

###################################################
# build an run tests
###################################################

# use one build directory for all build configurations
mkdir build
cd build

export cupla_DIR=$CI_PROJECT_DIR


for CXX_COMPILER in $CUPLA_CXX; do
    if [[ "$CXX_COMPILER" =~ clang++.* ]] ; then
        source ${CI_PROJECT_DIR}/script/install_clang.sh
    fi
    for BOOST_VERSION in ${CUPLA_BOOST_VERSIONS}; do
        # ALPAKA_ACCS contains the backends, which are used for each build
        # the backends are set in the sepcialized base jobs .base_gcc,.base_clang and.base_cuda
        for ACC in ${ALPAKA_ACCS}; do
            CMAKE_ARGS=$CUPLA_CONST_ARGS
            if [ -n "$CUPLA_USE_CLANG_CUDA" ] ; then
                CMAKE_ARGS=("${CMAKE_ARGS} -DCMAKE_CUDA_COMPILER=${CXX_COMPILER}")
            fi
            CMAKE_ARGS=("${CMAKE_ARGS} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DBOOST_ROOT=/opt/boost/${BOOST_VERSION} -D${ACC}=ON")

            echo -e "\033[0;32m///////////////////////////////////////////////////"
            echo "number of processor threads -> $(nproc)"
            cmake --version | head -n 1
            echo "CMAKE_ARGS -> ${CMAKE_ARGS}"
            echo -e "/////////////////////////////////////////////////// \033[0m \n\n"

            cmake $cupla_DIR $CMAKE_ARGS
            cmake --build . -j

            echo "###################################################"
            echo "# Example Matrix Multiplication (adapted original)"
            echo "###################################################"
            echo "can not run with CPU_B_SEQ_T_SEQ due to missing elements layer in original SDK example"
            echo "CPU_B_SEQ_T_OMP2/THREADS too many threads necessary (256)"
            if [[ $CMAKE_ARGS =~ -*Dalpaka_ACC_GPU_CUDA_ENABLE=ON.* ]]; then
                time ./example/CUDASamples/matrixMul/matrixMul -wA=64 -wB=64 -hA=64 -hB=64
            fi

            echo "###################################################"
            echo "# Example Async API (adapted original)"
            echo "###################################################"
            echo "can not run with CPU_B_SEQ_T_SEQ due to missing elements layer in original SDK example"
            echo "CPU_B_SEQ_T_OMP2/THREADS too many threads necessary (512)"
            if [[ $CMAKE_ARGS =~ -*Dalpaka_ACC_GPU_CUDA_ENABLE=ON.* ]]; then
                time ./example/CUDASamples/asyncAPI/asyncAPI
            fi

            echo "###################################################"
            echo "# Example Async API (added elements layer)"
            echo "###################################################"
            time ./example/CUDASamples/asyncAPI_tuned/asyncAPI_tuned

            echo "###################################################"
            echo "Example vectorAdd (added elements layer)"
            echo "###################################################"
            time ./example/CUDASamples/vectorAdd/vectorAdd 100000

            echo "###################################################"
            echo "Example cupla vectorAdd (added elements layer)"
            echo "###################################################"
            time ./example/CUDASamples/cuplaVectorAdd/cuplaVectorAdd 100000


            echo "###################################################"
            echo "Example blackSchloles"
            echo "###################################################"
            if [[ $CMAKE_ARGS =~ -*Dalpaka_ACC_GPU_CUDA_ENABLE=ON.* ]]; then
                time ./example/CUDASamples/blackScholes/blackScholes
            fi

            rm -r *
        done
    done
    if [[ "$CXX_COMPILER" =~ clang++.* ]] ; then
        source ${CI_PROJECT_DIR}/script/deinstall_clang.sh
    fi
done
