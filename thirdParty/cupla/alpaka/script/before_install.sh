#!/bin/bash

#
# Copyright 2021 Benjamin Worpitz, Bernhard Manfred Gruber
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

#-------------------------------------------------------------------------------
# gcc
if [ ! -z "${ALPAKA_CI_GCC_VER+x}" ]
then
    ALPAKA_CI_GCC_VER_SEMANTIC=( ${ALPAKA_CI_GCC_VER//./ } )
    export ALPAKA_CI_GCC_VER_MAJOR="${ALPAKA_CI_GCC_VER_SEMANTIC[0]}"
    echo ALPAKA_CI_GCC_VER_MAJOR: "${ALPAKA_CI_GCC_VER_MAJOR}"

    if [[ "$(cat /etc/os-release)" == *"20.04"* ]]
    then
        if (( "${ALPAKA_CI_GCC_VER_MAJOR}" <= 6 ))
        then
            echo "Ubuntu 20.04 does not provide gcc-6 and older anymore."
            exit 1
        fi
    fi
fi

#-------------------------------------------------------------------------------
# Boost.
echo $ALPAKA_CI_BOOST_BRANCH
ALPAKA_CI_BOOST_BRANCH_MAJOR=${ALPAKA_CI_BOOST_BRANCH:6:1}
echo ALPAKA_CI_BOOST_BRANCH_MAJOR: "${ALPAKA_CI_BOOST_BRANCH_MAJOR}"
ALPAKA_CI_BOOST_BRANCH_MINOR=${ALPAKA_CI_BOOST_BRANCH:8:2}
echo ALPAKA_CI_BOOST_BRANCH_MINOR: "${ALPAKA_CI_BOOST_BRANCH_MINOR}"

export ALPAKA_CI_INSTALL_ATOMIC="OFF"
# If the variable is not set, the backend will most probably be used by default so we install Boost.Atomic
if [ "${alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE-ON}" == "ON" ] ||
    [ "${alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE-ON}" == "ON" ] ||
    [ "${alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE-ON}" == "ON" ] ||
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

#-------------------------------------------------------------------------------
# Fibers
export ALPAKA_CI_INSTALL_FIBERS="OFF"
if [ ! -z "${alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE+x}" ]
then
    if [ "${alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}" = "ON" ]
    then
        export ALPAKA_CI_INSTALL_FIBERS="ON"
    fi
else
    # If the variable is not set, the backend will most probably be used by default so we install it.
    export ALPAKA_CI_INSTALL_FIBERS="ON"
fi


# GCC-5.5 has broken avx512vlintrin.h in Release mode with NVCC 9.X
#   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=76731
#   https://github.com/tensorflow/tensorflow/issues/10220
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON"  ]
then
    if [[ "${CXX}" == "g++"* ]]
    then
        if (( "${ALPAKA_CI_GCC_VER_MAJOR}" == 5 ))
        then
            if [ "${CMAKE_CUDA_COMPILER}" == "nvcc" ]
            then
                if [ "${CMAKE_BUILD_TYPE}" == "Release" ]
                then
                    export CMAKE_BUILD_TYPE=Debug
                fi
            fi
        fi
    fi
fi

# nvcc does not recognize GCC-9 builtins from avx512fintrin.h in Release
#   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=76731
#   https://github.com/tensorflow/tensorflow/issues/10220
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON"  ]
then
    if [[ "${CXX}" == "g++"* ]]
    then
        if (( "${ALPAKA_CI_GCC_VER_MAJOR}" == 9 ))
        then
            if [ "${CMAKE_CUDA_COMPILER}" == "nvcc" ]
            then
                if [ "${CMAKE_BUILD_TYPE}" == "Release" ]
                then
                    export CMAKE_BUILD_TYPE=Debug
                fi
            fi
        fi
    fi
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
        if [[ "${CXX}" == "g++"* ]]
        then
            echo "using libc++ with g++ not yet supported."
            exit 1
        fi
    fi
fi

if [ ! -z "${ALPAKA_CI_CLANG_VER+x}" ]
then
    if [ "${ALPAKA_CI_CLANG_VER}" == 5 ]
    then
        if [ "${ALPAKA_CI_INSTALL_FIBERS}" == "ON" ]
        then
            # https://github.com/boostorg/fiber/issues/272
            echo "clang-5 is not compatible with boost.fibers."
            exit 1
        fi
    fi
fi

