#!/bin/bash

#
# Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#
set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: before_install>"

# because of the strict abort conditions, a variable needs to be defined, if we read from
# this statement avoids additional checks later in the scripts
if [ -z "${LD_LIBRARY_PATH+x}" ]
then
    export LD_LIBRARY_PATH=""
fi

#-------------------------------------------------------------------------------
# gcc
# TODO(sehrig): remove me, if the job generator is used
if [ ! -z "${ALPAKA_CI_GCC_VER+x}" ]
then
    ALPAKA_CI_GCC_VER_SEMANTIC=( ${ALPAKA_CI_GCC_VER//./ } )
    export ALPAKA_CI_GCC_VER_MAJOR="${ALPAKA_CI_GCC_VER_SEMANTIC[0]}"
    echo ALPAKA_CI_GCC_VER_MAJOR: "${ALPAKA_CI_GCC_VER_MAJOR}"
fi

export ALPAKA_CI_INSTALL_ATOMIC="OFF"
# If the variable is not set, the backend will most probably be used by default so we install Boost.Atomic
if [ "${alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE-ON}" == "ON" ] ||
    [ "${alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE-ON}" == "ON" ] ||
    [ "${alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE-ON}" == "ON" ] ||
    [ "${alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE-ON}" == "ON" ] ||
    [ "${alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE-ON}" == "ON" ]
then
  export ALPAKA_CI_INSTALL_ATOMIC="ON"
fi

#-------------------------------------------------------------------------------
# CUDA
export ALPAKA_CI_INSTALL_CUDA="OFF"
if [[ "${alpaka_ACC_GPU_CUDA_ENABLE}" == "ON" ]]
then
    export ALPAKA_CI_INSTALL_CUDA="ON"
else
    echo_yellow "<SETENV: set cuda environment variables for disabled backend>"
    export alpaka_RELOCATABLE_DEVICE_CODE=${alpaka_RELOCATABLE_DEVICE_CODE:=""}
    export CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES:=""}
    export CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER:=""}
    export alpaka_CUDA_SHOW_REGISTER=${alpaka_CUDA_SHOW_REGISTER:=""}
    export alpaka_CUDA_KEEP_FILES=${alpaka_CUDA_KEEP_FILES:=""}
    export alpaka_CUDA_EXPT_EXTENDED_LAMBDA=${alpaka_CUDA_EXPT_EXTENDED_LAMBDA:=""}
fi

#-------------------------------------------------------------------------------
# HIP
export ALPAKA_CI_INSTALL_HIP="OFF"
if [ "${alpaka_ACC_GPU_HIP_ENABLE}" == "ON" ]
then
    export ALPAKA_CI_INSTALL_HIP="ON"
else
    echo_yellow "<DEFAULT: hip environment variables for disabled backend>"
    export CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES:=""}
    export CMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER:=""}
fi

#-------------------------------------------------------------------------------
# TBB
export ALPAKA_CI_INSTALL_TBB="OFF"
if [ ! -z "${alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE+x}" ]
then
    if [ "${alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE}" = "ON" ]
    then
        export ALPAKA_CI_INSTALL_TBB="ON"
    fi
else
    # If the variable is not set, the backend will most probably be used by default so we install it.
    export ALPAKA_CI_INSTALL_TBB="ON"
fi

#-------------------------------------------------------------------------------
# OPENMP
export ALPAKA_CI_INSTALL_OMP="OFF"
if [ "$ALPAKA_CI_OS_NAME" = "macOS"  ]
then
    export ALPAKA_CI_INSTALL_OMP="ON"
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
    then
        if [ ! -z "${alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE+x}" ]
        then
            if [ "${alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE}" = "ON" ]
            then
                 echo "libc++ is not compatible with TBB."
                 exit 1
            fi
        fi
    fi
fi

#-------------------------------------------------------------------------------
# ONEAPI
if [[ "${ALPAKA_CI_CXX}" == "icpx" ]]; then
    if [[ "${alpaka_ACC_SYCL_ENABLE}" != "ON" ]]; then
        echo_red "alpaka_ACC_SYCL_ENABLE needs to be enabled, if the C++ compiler is icpx"
        exit 1
    fi
fi

if [ "${alpaka_ACC_SYCL_ENABLE}" == "OFF" ]; then
    echo_yellow "<DEFAULT: SYCL environment variables for disabled backend>"
    export alpaka_SYCL_ONEAPI_CPU=${alpaka_SYCL_ONEAPI_CPU:=""}
    export alpaka_SYCL_ONEAPI_CPU_ISA=${alpaka_SYCL_ONEAPI_CPU_ISA:=""}
    export alpaka_SYCL_ONEAPI_GPU=${alpaka_SYCL_ONEAPI_GPU:=""}
    export alpaka_SYCL_ONEAPI_GPU_DEVICES=${alpaka_SYCL_ONEAPI_GPU_DEVICES:=""}
    export alpaka_SYCL_ONEAPI_FPGA=${alpaka_SYCL_ONEAPI_FPGA:=""}
    export alpaka_SYCL_ONEAPI_FPGA_MODE=${alpaka_SYCL_ONEAPI_FPGA_MODE:=""}
    export alpaka_SYCL_ONEAPI_FPGA_BOARD=${alpaka_SYCL_ONEAPI_FPGA_BOARD:=""}
    export alpaka_SYCL_ONEAPI_FPGA_BSP=${alpaka_SYCL_ONEAPI_FPGA_BSP:=""}
else
    if !( [ "$alpaka_SYCL_ONEAPI_CPU" == "ON" ] || [ "$alpaka_SYCL_ONEAPI_GPU" == "ON" ] || [ "$alpaka_SYCL_ONEAPI_FPGA" == "ON" ] ); then
        echo_red "ERROR: the SYCL CPU, GPU or FPGA device needs to be enabled"
        exit 1
    fi

    if [ "$alpaka_SYCL_ONEAPI_CPU" == "ON" ] && [ "$alpaka_SYCL_ONEAPI_FPGA" == "ON" ]; then
        echo_red "ERROR: the SYCL CPU or FPGA device cannot be enabled at the same time"
        exit 1
    fi

    if [ "$alpaka_SYCL_ONEAPI_CPU" == "ON" ] && [ "$alpaka_SYCL_ONEAPI_GPU" == "ON" ]; then
        echo_red "ERROR: the SYCL CPU or GPU device cannot be enabled at the same time"
        exit 1
    fi

    if [ "$alpaka_SYCL_ONEAPI_GPU" == "ON" ] && [ "$alpaka_SYCL_ONEAPI_FPGA" == "ON" ]; then
        echo_red "ERROR: the SYCL GPU or FPGA device cannot be enabled at the same time"
        exit 1
    fi

    if [ "$alpaka_SYCL_ONEAPI_CPU" == "OFF" ]; then
        echo_yellow "<DEFAULT: SYCL environment variables for enabled backend and disabled CPU device>"
        export alpaka_SYCL_ONEAPI_CPU_ISA=""
    fi

    if [ "$alpaka_SYCL_ONEAPI_GPU" == "OFF" ]; then
        echo_yellow "<DEFAULT: SYCL environment variables for enabled backend and disabled GPU device>"
        export alpaka_SYCL_ONEAPI_GPU_DEVICES=""
    fi

    if [ "$alpaka_SYCL_ONEAPI_FPGA" == "OFF" ]; then
        echo_yellow "<DEFAULT: SYCL environment variables for enabled backend and disabled FPGA device>"
        export alpaka_SYCL_ONEAPI_FPGA_MODE=""
        export alpaka_SYCL_ONEAPI_FPGA_BOARD=""
        export alpaka_SYCL_ONEAPI_FPGA_BSP=""
    fi
fi

#-------------------------------------------------------------------------------
# Install paths
export ALPAKA_CI_CMAKE_DIR=$HOME/cmake
export ALPAKA_CI_CUDA_DIR=$HOME/cuda
#-------------------------------------------------------------------------------
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
    then
        if [[ "${ALPAKA_CI_CXX}" == "g++"* ]]
        then
            echo "using libc++ with g++ not yet supported."
            exit 1
        fi
    fi
fi

if [ "$ALPAKA_CI_OS_NAME" = "Windows" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    export CMAKE_CXX_COMPILER=$ALPAKA_CI_CXX
fi
