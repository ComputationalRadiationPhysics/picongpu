#!/bin/bash

#
# Copyright 2021 Benjamin Worpitz, Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0
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
    if [[ "${CXX}" = "clang++"* ]]
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
    export PATH=${PATH}:"${TBB_ROOT}/redist/intel64/vc14"
fi

# CUDA
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ]
then
    : "${ALPAKA_CI_CUDA_VERSION?'ALPAKA_CI_CUDA_VERSION must be specified'}"

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
    then
        # CUDA
        export PATH=/usr/local/cuda-${ALPAKA_CI_CUDA_VERSION}/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-${ALPAKA_CI_CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}
        # We have to explicitly add the stub libcuda.so to CUDA_LIB_PATH because the real one would be installed by the driver (which we can not install).
        export CUDA_LIB_PATH=/usr/local/cuda/lib64/stubs/

        if [ "${CMAKE_CUDA_COMPILER}" == "nvcc" ]
        then
            which nvcc
            nvcc -V
        fi
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${ALPAKA_CI_CUDA_VERSION}/bin":$PATH
        export CUDA_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${ALPAKA_CI_CUDA_VERSION}"

        which nvcc
        nvcc -V

        export CUDA_PATH_V${ALPAKA_CI_CUDA_VERSION/./_}="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${ALPAKA_CI_CUDA_VERSION}"
    fi
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

        if [[ "${CXX}" == "clang++"* ]]
        then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -stdlib=libc++"
        fi
    fi

    if [ "${CXX}" == "icpc" ]
    then
        set +eu
        which ${CXX} || source /opt/intel/oneapi/setvars.sh
        set -eu
    fi

    which "${CXX}"
    ${CXX} --version
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    source ./script/prepare_sanitizers.sh
fi

if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    : ${ALPAKA_CI_CL_VER?"ALPAKA_CI_CL_VER must be specified"}

    # Add msbuild to the path
    if [ "$ALPAKA_CI_CL_VER" = "2017" ]
    then
        export MSBUILD_EXECUTABLE="/C/Program Files (x86)/Microsoft Visual Studio/2017/Enterprise/MSBuild/15.0/Bin/MSBuild.exe"
    elif [ "$ALPAKA_CI_CL_VER" = "2019" ] || [ "$ALPAKA_CI_CL_VER" = "2022" ]
    then
        export MSBUILD_EXECUTABLE=$(vswhere.exe -latest -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe")
    fi
    "$MSBUILD_EXECUTABLE" -version

    if [ "$ALPAKA_CI_CL_VER" = "2022" ]
    then
        VCVARS_BAT="/C/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build/vcvars64.bat"
        "$VCVARS_BAT"
    fi
fi

if [ -z "${ALPAKA_TEST_MDSPAN+x}" ];
then
    export alpaka_USE_MDSPAN=OFF
else
    if [ "${ALPAKA_TEST_MDSPAN}" == "ON" ];
    then
        export alpaka_USE_MDSPAN=FETCH
    else
        export alpaka_USE_MDSPAN=OFF
    fi
fi

./script/run_generate.sh
./script/run_build.sh

if [ "${ALPAKA_CI_RUN_TESTS}" == "ON" ] ; then ./script/run_tests.sh; fi
if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;
then
    ./script/run_analysis.sh
    ./script/run_install.sh
fi
