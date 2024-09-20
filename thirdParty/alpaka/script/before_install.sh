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
fi

#-------------------------------------------------------------------------------
# HIP
export ALPAKA_CI_INSTALL_HIP="OFF"
if [ "${alpaka_ACC_GPU_HIP_ENABLE}" == "ON" ]
then
    export ALPAKA_CI_INSTALL_HIP="ON"
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
