#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis/set.sh

: "${ALPAKA_CI_CMAKE_DIR?'ALPAKA_CI_CMAKE_DIR must be specified'}"
echo "ALPAKA_CI_CMAKE_DIR: ${ALPAKA_CI_CMAKE_DIR}"
: "${ALPAKA_CI_ANALYSIS?'ALPAKA_CI_ANALYSIS must be specified'}"
echo "ALPAKA_CI_ANALYSIS: ${ALPAKA_CI_ANALYSIS}"
: "${ALPAKA_CI_INSTALL_CUDA?'ALPAKA_CI_INSTALL_CUDA must be specified'}"
: "${ALPAKA_CI_INSTALL_HIP?'ALPAKA_CI_INSTALL_HIP must be specified'}"
if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    : "${ALPAKA_CI_STDLIB?'ALPAKA_CI_STDLIB must be specified'}"
    echo "ALPAKA_CI_STDLIB: ${ALPAKA_CI_STDLIB}"
fi
: "${CXX?'CXX must be specified'}"
echo "CXX: ${CXX}"


if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    if [ -z "${LD_LIBRARY_PATH+x}" ]
    then
        LD_LIBRARY_PATH=
    fi
fi

# CMake
if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    export PATH=${ALPAKA_CI_CMAKE_DIR}/bin:${PATH}
fi
cmake --version

#TBB
if [ "$TRAVIS_OS_NAME" = "windows" ]
then
    #ALPAKA_TBB_BIN_DIR="${TBB_ROOT_DIR}/bin/ia32/vc14"
    ALPAKA_TBB_BIN_DIR="${TBB_ROOT_DIR}/bin/intel64/vc14"
    export PATH=${PATH}:"${ALPAKA_TBB_BIN_DIR}"
fi

# CUDA
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ]
then
    : "${ALPAKA_CUDA_VERSION?'ALPAKA_CUDA_VERSION must be specified'}"

    if [ "$TRAVIS_OS_NAME" = "linux" ]
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
    elif [ "$TRAVIS_OS_NAME" = "windows" ]
    then
        export PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${ALPAKA_CUDA_VERSION}\bin":$PATH
        export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${ALPAKA_CUDA_VERSION}"
    fi
fi

# HIP
if [ "${ALPAKA_CI_INSTALL_HIP}" == "ON" ]
then
: "${ALPAKA_CI_HIP_ROOT_DIR?'ALPAKA_CI_HIP_ROOT_DIR must be specified'}"

    # HIP
    # HIP_PATH required by HIP tools
    export HIP_PATH=${ALPAKA_CI_HIP_ROOT_DIR}
    # CUDA_PATH required by HIP tools
    if [ -n "$(command -v nvcc)" ]
    then
        export CUDA_PATH=$(dirname $(which nvcc))/../
    else
        export CUDA_PATH=/usr/local/cuda-${ALPAKA_CUDA_VERSION}
    fi

    export PATH=${HIP_PATH}/bin:$PATH
    export LD_LIBRARY_PATH=${HIP_PATH}/lib64:${HIP_PATH}/hiprand/lib:${LD_LIBRARY_PATH}
    export CMAKE_PREFIX_PATH=${HIP_PATH}:${HIP_PATH}/hiprand:${CMAKE_PREFIX_PATH:-}
    # to avoid "use of uninitialized value .." warnings in perl script hipcc
    # TODO: rely on CI vars for platform and architecture
    export HIP_PLATFORM=nvcc
    export HIP_RUNTIME=nvcc
    # calls nvcc or hcc
    which hipcc
    hipcc -V
    which hipconfig
    hipconfig --platform
    hipconfig -v
    # print newline as previous command does not do this
    echo

fi

# clang
if [ "${CXX}" == "clang++" ]
then
    # We have to prepend /usr/bin to the path because else the preinstalled clang from usr/bin/local/ is used.
    export PATH=${ALPAKA_CI_CLANG_DIR}/bin:${PATH}
    export LD_LIBRARY_PATH=${ALPAKA_CI_CLANG_DIR}/lib:${LD_LIBRARY_PATH}
    if [ -z "${CPPFLAGS+x}" ]
    then
        CPPFLAGS=
    fi
    export CPPFLAGS="-I ${ALPAKA_CI_CLANG_DIR}/include/c++/v1 ${CPPFLAGS}"
fi

# stdlib
if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
    then
        if [ -z "${CMAKE_CXX_FLAGS+x}" ]
        then
            export CMAKE_CXX_FLAGS=
        fi
        CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -stdlib=libc++"

        if [ -z "${CMAKE_EXE_LINKER_FLAGS+x}" ]
        then
            export CMAKE_EXE_LINKER_FLAGS=
        fi
        CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS} -lc++ -lc++abi"
    fi

    which "${CXX}"
    ${CXX} -v

    source ./script/travis/prepare_sanitizers.sh
    if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;then ./script/travis/run_analysis.sh ;fi
fi

./script/travis/run_build.sh

if [ "${ALPAKA_CI_ANALYSIS}" == "OFF" ] ;then ./script/travis/run_tests.sh ;fi
