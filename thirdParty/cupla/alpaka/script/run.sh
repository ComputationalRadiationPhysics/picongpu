#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

: "${ALPAKA_CI_CMAKE_DIR?'ALPAKA_CI_CMAKE_DIR must be specified'}"
echo "ALPAKA_CI_CMAKE_DIR: ${ALPAKA_CI_CMAKE_DIR}"
: "${ALPAKA_CI_ANALYSIS?'ALPAKA_CI_ANALYSIS must be specified'}"
echo "ALPAKA_CI_ANALYSIS: ${ALPAKA_CI_ANALYSIS}"
: "${ALPAKA_CI_INSTALL_CUDA?'ALPAKA_CI_INSTALL_CUDA must be specified'}"
: "${ALPAKA_CI_INSTALL_HIP?'ALPAKA_CI_INSTALL_HIP must be specified'}"
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    : "${ALPAKA_CI_STDLIB?'ALPAKA_CI_STDLIB must be specified'}"
    echo "ALPAKA_CI_STDLIB: ${ALPAKA_CI_STDLIB}"
fi
: "${CXX?'CXX must be specified'}"
echo "CXX: ${CXX}"


if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    if [ -z "${LD_LIBRARY_PATH+x}" ]
    then
        LD_LIBRARY_PATH=
    fi
    if [ "${CXX}" = "clang++" ]
    then
        if [ "${ALPAKA_CI_CLANG_VER}" -ge "10" ]
        then
            export LD_LIBRARY_PATH="/usr/lib/llvm-${ALPAKA_CI_CLANG_VER}/lib/:${LD_LIBRARY_PATH}"
        fi
    fi
fi

# CMake
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    export PATH=${ALPAKA_CI_CMAKE_DIR}/bin:${PATH}
fi
cmake --version

#TBB
if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    ALPAKA_TBB_BIN_DIR="${TBB_ROOT}/bin/intel64/vc14"
    export PATH=${PATH}:"${ALPAKA_TBB_BIN_DIR}"
fi

# CUDA
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ]
then
    : "${ALPAKA_CUDA_VERSION?'ALPAKA_CUDA_VERSION must be specified'}"

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
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
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
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
    export HIP_PATH=/opt/rocm

    export PATH=${HIP_PATH}/bin:$PATH
    export LD_LIBRARY_PATH=${HIP_PATH}/lib64:${HIP_PATH}/hiprand/lib:${LD_LIBRARY_PATH}
    export CMAKE_PREFIX_PATH=${HIP_PATH}:${HIP_PATH}/hiprand:${CMAKE_PREFIX_PATH:-}
    export CMAKE_MODULE_PATH=${HIP_PATH}/hip/cmake
    # calls nvcc or clang
    which hipcc
    hipcc --version
    which hipconfig
    hipconfig --platform
    hipconfig -v
    # print newline as previous command does not do this
    echo

fi

# stdlib
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
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

    source ./script/prepare_sanitizers.sh
fi

if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    : ${ALPAKA_CI_CL_VER?"ALPAKA_CI_CL_VER must be specified"}

    # Use the 64 bit compiler
    # FIXME: Path not found but does not seem to be necessary anymore
    #"./C/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat" amd64

    # Add msbuild to the path
    if [ "$ALPAKA_CI_CL_VER" = "2017" ]
    then
        export MSBUILD_EXECUTABLE="/C/Program Files (x86)/Microsoft Visual Studio/2017/Enterprise/MSBuild/15.0/Bin/MSBuild.exe"
    elif [ "$ALPAKA_CI_CL_VER" = "2019" ]
    then
        export MSBUILD_EXECUTABLE=$(vswhere.exe -latest -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe")
    fi
    "$MSBUILD_EXECUTABLE" -version
fi

./script/run_generate.sh
./script/run_build.sh
if [ "${ALPAKA_CI_ANALYSIS}" == "OFF" ] ;then ./script/run_tests.sh ;fi
if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;then ./script/run_analysis.sh ;fi
