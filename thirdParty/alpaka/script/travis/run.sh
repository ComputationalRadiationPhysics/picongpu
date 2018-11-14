#!/bin/bash

#
# Copyright 2017-2018 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

#-------------------------------------------------------------------------------
# e: exit as soon as one command returns a non-zero exit code.
set -euo pipefail

: ${ALPAKA_CI_CMAKE_DIR?"ALPAKA_CI_CMAKE_DIR must be specified"}
: ${ALPAKA_CI_ANALYSIS?"ALPAKA_CI_ANALYSIS must be specified"}
: ${CXX?"CXX must be specified"}

if [[ ! -v LD_LIBRARY_PATH ]]
then
    LD_LIBRARY_PATH=
fi

# CMake
export PATH=${ALPAKA_CI_CMAKE_DIR}/bin:${PATH}
cmake --version

if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "ON" ] || [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "ON" ] && [ "${ALPAKA_HIP_PLATFORM}" == "nvcc" ]
then
    # CUDA
    export PATH=/usr/local/cuda-${ALPAKA_CUDA_VERSION}/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-${ALPAKA_CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}
    # We have to explicitly add the stub libcuda.so to CUDA_LIB_PATH because the real one would be installed by the driver (which we can not install).
    export CUDA_LIB_PATH=/usr/local/cuda/lib64/stubs/

    if [ "${ALPAKA_CUDA_COMPILER}" == "nvcc" ]
    then
        which nvcc
        nvcc -V
    fi
fi

if [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "ON" ]
then
    # && [ "${ALPAKA_HIP_PLATFORM}" == "nvcc" ]
    # HIP
    # HIP_PATH required by HIP tools
    export HIP_PATH=${ALPAKA_CI_HIP_ROOT_DIR}
    # CUDA_PATH required by HIP tools
    export CUDA_PATH=/usr/local/cuda-${ALPAKA_CUDA_VERSION}
    export PATH=${HIP_PATH}/bin:$PATH
    export LD_LIBRARY_PATH=${HIP_PATH}/lib64:${HIP_PATH}/hiprand/lib:${LD_LIBRARY_PATH}
    export CMAKE_PREFIX_PATH=${HIP_PATH}:${HIP_PATH}/hiprand:${CMAKE_PREFIX_PATH:-}

    # calls nvcc or hcc
    which hipcc
    hipcc -V
    which hipconfig
    hipconfig -v
    # print newline as previous command does not do this
    echo
fi

if [ "${CXX}" == "clang++" ]
then
    # We have to prepend /usr/bin to the path because else the preinstalled clang from usr/bin/local/ is used.
    export PATH=${ALPAKA_CI_CLANG_DIR}/bin:${PATH}
    export LD_LIBRARY_PATH=${ALPAKA_CI_CLANG_DIR}/lib:${LD_LIBRARY_PATH}
    if [[ ! -v CPPFLAGS ]]
    then
        CPPFLAGS=
    fi
    export CPPFLAGS="-I ${ALPAKA_CI_CLANG_DIR}/include/c++/v1 ${CPPFLAGS}"
    if [[ ! -v CXXFLAGS ]]
    then
        CXXFLAGS=
    fi
    export CXXFLAGS="-lc++ ${CXXFLAGS}"
fi

which "${CXX}"
${CXX} -v

source ./script/travis/prepare_sanitizers.sh
if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;then ./script/travis/run_analysis.sh ;fi
./script/travis/run_build.sh
if [ "${ALPAKA_CI_ANALYSIS}" == "OFF" ] ;then ./script/travis/run_tests.sh ;fi
