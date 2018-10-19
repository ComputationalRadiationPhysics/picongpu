#!/bin/bash

#
# Copyright 2017 Benjamin Worpitz
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
set -eo pipefail

#-------------------------------------------------------------------------------
# CMake
ALPAKA_CI_CMAKE_VER_SEMANTIC=( ${ALPAKA_CI_CMAKE_VER//./ } )
export ALPAKA_CI_CMAKE_VER_MAJOR="${ALPAKA_CI_CMAKE_VER_SEMANTIC[0]}"
echo ALPAKA_CI_CMAKE_VER_MAJOR: "${ALPAKA_CI_CMAKE_VER_MAJOR}"
export ALPAKA_CI_CMAKE_VER_MINOR="${ALPAKA_CI_CMAKE_VER_SEMANTIC[1]}"
echo ALPAKA_CI_CMAKE_VER_MINOR: "${ALPAKA_CI_CMAKE_VER_MINOR}"

#-------------------------------------------------------------------------------
# gcc
if [ "${CXX}" == "g++" ]
then
    ALPAKA_CI_GCC_VER_SEMANTIC=( ${ALPAKA_CI_GCC_VER//./ } )
    export ALPAKA_CI_GCC_VER_MAJOR="${ALPAKA_CI_GCC_VER_SEMANTIC[0]}"
    echo ALPAKA_CI_GCC_VER_MAJOR: "${ALPAKA_CI_GCC_VER_MAJOR}"
    export ALPAKA_CI_GCC_VER_MINOR="${ALPAKA_CI_GCC_VER_SEMANTIC[1]}"
    echo ALPAKA_CI_GCC_VER_MINOR: "${ALPAKA_CI_GCC_VER_MINOR}"
fi

#-------------------------------------------------------------------------------
# clang
if [ "${CXX}" == "clang++" ]
then
    ALPAKA_CI_CLANG_VER_SEMANTIC=( ${ALPAKA_CI_CLANG_VER//./ } )
    export ALPAKA_CI_CLANG_VER_MAJOR="${ALPAKA_CI_CLANG_VER_SEMANTIC[0]}"
    echo ALPAKA_CI_CLANG_VER_MAJOR: "${ALPAKA_CI_CLANG_VER_MAJOR}"
    export ALPAKA_CI_CLANG_VER_MINOR="${ALPAKA_CI_CLANG_VER_SEMANTIC[1]}"
    echo ALPAKA_CI_CLANG_VER_MINOR: "${ALPAKA_CI_CLANG_VER_MINOR}"

    # clang versions lower than 3.7 do not support OpenMP 2.0.
    if (( (( ALPAKA_CI_CLANG_VER_MAJOR < 3 )) || ( (( ALPAKA_CI_CLANG_VER_MAJOR == 3 )) && (( ALPAKA_CI_CLANG_VER_MINOR < 7 )) ) ))
    then
        if [ "${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE}" == "ON" ]
        then
            export ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=OFF
            echo ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE} because the clang version does not support it!
        fi
        if [ "${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE}" == "ON" ]
        then
            export ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=OFF
            echo ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE} because the clang version does not support it!
        fi
        if [ "${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}" == "ON" ]
        then
            export ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=OFF
            echo ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE} because the clang version does not support it!
        fi
    fi

    # clang versions lower than 3.9 do not support OpenMP 4.0
    if (( (( ALPAKA_CI_CLANG_VER_MAJOR < 3 )) || ( (( ALPAKA_CI_CLANG_VER_MAJOR == 3 )) && (( ALPAKA_CI_CLANG_VER_MINOR < 9 )) ) ))
    then
        if [ "${ALPAKA_ACC_CPU_BT_OMP4_ENABLE}" == "ON" ]
        then
            export ALPAKA_ACC_CPU_BT_OMP4_ENABLE=OFF
            echo ALPAKA_ACC_CPU_BT_OMP4_ENABLE=${ALPAKA_ACC_CPU_BT_OMP4_ENABLE} because the clang version does not support it!
        fi
    fi
fi

#-------------------------------------------------------------------------------
# Boost.
export ALPAKA_CI_BOOST_BRANCH_MAJOR=${ALPAKA_CI_BOOST_BRANCH:6:1}
echo ALPAKA_CI_BOOST_BRANCH_MAJOR: "${ALPAKA_CI_BOOST_BRANCH_MAJOR}"
export ALPAKA_CI_BOOST_BRANCH_MINOR=${ALPAKA_CI_BOOST_BRANCH:8:2}
echo ALPAKA_CI_BOOST_BRANCH_MINOR: "${ALPAKA_CI_BOOST_BRANCH_MINOR}"

#-------------------------------------------------------------------------------
# CUDA
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "ON" ]
then
    ALPAKA_CI_CUDA_VER_SEMANTIC=( ${ALPAKA_CUDA_VER//./ } )
    export ALPAKA_CUDA_VER_MAJOR="${ALPAKA_CI_CUDA_VER_SEMANTIC[0]}"
    echo ALPAKA_CUDA_VER_MAJOR: "${ALPAKA_CUDA_VER_MAJOR}"
    export ALPAKA_CUDA_VER_MINOR="${ALPAKA_CI_CUDA_VER_SEMANTIC[1]}"
    echo ALPAKA_CUDA_VER_MINOR: "${ALPAKA_CUDA_VER_MINOR}"

    if [ "${ALPAKA_CUDA_COMPILER}" == "nvcc" ]
    then
        # FIXME: BOOST_AUTO_TEST_CASE_TEMPLATE is not compilable with nvcc in Release mode.
        if [ "${CMAKE_BUILD_TYPE}" == "Release" ]
        then
            export CMAKE_BUILD_TYPE=Debug
        fi

        # nvcc <= 9.2 does not support boost correctly so fibers have to be disabled.
        if (( (( ALPAKA_CUDA_VER_MAJOR < 9 )) || ( (( ALPAKA_CUDA_VER_MAJOR == 9 )) && (( ALPAKA_CUDA_VER_MINOR <= 2 )) ) ))
        then
            if [ "${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}" == "ON" ]
            then
                export ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=OFF
                echo ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE} because nvcc does not support boost fibers correctly!
            fi
        fi
    fi

    if [ "${ALPAKA_CUDA_COMPILER}" == "clang" ]
    then
        # clang as native CUDA compiler does not support boost fibers
        if [ ${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE} == "ON" ]
        then
            export ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=OFF
            echo ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE} because clang as native CUDA compiler does not support boost fibers correctly!
        fi

        # clang as native CUDA compiler does not support OpenMP
        if [ "${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE}" == "ON" ]
        then
            export ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=OFF
            echo ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE} because the clang as native CUDA compiler does not support OpenMP!
        fi
        if [ "${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE}" == "ON" ]
        then
            export ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=OFF
            echo ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE} because the clang as native CUDA compiler does not support OpenMP!
        fi
        if [ "${ALPAKA_ACC_CPU_BT_OMP4_ENABLE}" == "ON" ]
        then
            export ALPAKA_ACC_CPU_BT_OMP4_ENABLE=OFF
            echo ALPAKA_ACC_CPU_BT_OMP4_ENABLE=${ALPAKA_ACC_CPU_BT_OMP4_ENABLE} because the clang as native CUDA compiler does not support OpenMP!
        fi
    fi
fi
