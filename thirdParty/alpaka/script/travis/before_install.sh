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
        # nvcc 7.x does not support gcc > 4
        # nvcc 8.x does not support gcc > 5
        # nvcc 9.x supports gcc 6 but does not compile alpaka correctly
        if [ "${CXX}" == "g++" ]
        then
            if (( ALPAKA_CUDA_VER_MAJOR < 8 ))
            then
                if (( ALPAKA_CI_GCC_VER_MAJOR > 4 ))
                then
                    echo nvcc "${ALPAKA_CUDA_VER}" does not support gcc "${ALPAKA_CI_GCC_VER}"!
                    exit 1
                fi
            elif (( ALPAKA_CUDA_VER_MAJOR < 9 ))
            then
                if (( ALPAKA_CI_GCC_VER_MAJOR > 5 ))
                then
                    echo nvcc "${ALPAKA_CUDA_VER}" does not support gcc "${ALPAKA_CI_GCC_VER}"!
                    exit 1
                fi
            elif (( ALPAKA_CUDA_VER_MAJOR < 10 ))
            then
                if (( ALPAKA_CI_GCC_VER_MAJOR > 5 ))
                then
                    echo nvcc "${ALPAKA_CUDA_VER}" does not compile alpaka correctly when using gcc "${ALPAKA_CI_GCC_VER}"!
                    exit 1
                fi
            else
                echo unknown CUDA version. Update this script!
                exit 1
            fi
        fi

        # nvcc 7.0 does not support clang on linux.
        # nvcc 7.5 does support clang 3.5-3.6 on linux.
        # nvcc 7.5 for clang is buggy and does not compile alpaka correctly.
        # nvcc 8.0 does support clang 3.8+ on linux. However it fails with errors (e.g. error: calling a __host__ function("__builtin_logl") from a __device__ function("std::log") is not allowed). clang 3.7 on the other hand works.
        # nvcc 9.0 does support clang 3.9 on linux.
        # nvcc 9.1 does support clang 4.0 on linux.
        if [ "${CXX}" == "clang++" ]
        then
            if [ "${ALPAKA_CUDA_VER}" == "7.0" ]
            then
                echo nvcc {ALPAKA_CUDA_VER} does not support clang on linux!
                exit 1
            elif [ "${ALPAKA_CUDA_VER}" == "7.5" ]
            then
                echo nvcc 7.5 clang support is too buggy for alpaka!
                exit 1
                if (( (( ALPAKA_CI_CLANG_VER_MAJOR != 3 )) || ( (( ALPAKA_CI_CLANG_VER_MINOR != 5 )) || (( ALPAKA_CI_CLANG_VER_MINOR != 6 )) ) ))
                then
                    echo clang versions other than 3.5 or 3.6 are not a supported compiler for nvcc {ALPAKA_CUDA_VER} on linux!
                    exit 1
                fi
            elif [ "${ALPAKA_CUDA_VER}" == "8.0" ]
            then
                if (( (( ALPAKA_CI_CLANG_VER_MAJOR < 3 )) || ( (( ALPAKA_CI_CLANG_VER_MAJOR == 3 )) && (( ALPAKA_CI_CLANG_VER_MINOR < 7 )) ) ))
                then
                    echo clang versions lower than 3.7 are not a supported compiler for nvcc {ALPAKA_CUDA_VER} on linux!
                    exit 1
                fi
            elif [ "${ALPAKA_CUDA_VER}" == "9.0" ]
            then
                if (( (( ALPAKA_CI_CLANG_VER_MAJOR != 3 )) || (( ALPAKA_CI_CLANG_VER_MINOR != 9 )) ))
                then
                    echo clang versions other than 3.9 are not a supported compiler for nvcc {ALPAKA_CUDA_VER} on linux!
                    exit 1
                fi
            elif [ "${ALPAKA_CUDA_VER}" == "9.1" ]
            then
                if (( (( ALPAKA_CI_CLANG_VER_MAJOR != 4 )) || (( ALPAKA_CI_CLANG_VER_MINOR != 0 )) ))
                then
                    echo clang versions other than 4.0 are not a supported compiler for nvcc {ALPAKA_CUDA_VER} on linux!
                    exit 1
                fi
            else
                echo unknown CUDA version. Update this script!
                exit 1
            fi
        fi

        if (( ALPAKA_CUDA_VER_MAJOR >= 9 ))
        then
            if (( ALPAKA_CI_BOOST_BRANCH_MINOR < 65 ))
            then
                echo nvcc "${ALPAKA_CUDA_VER}" does not support boost version prior to 1.65.1!
                exit 1
            fi
        fi

        # FIXME: BOOST_AUTO_TEST_CASE_TEMPLATE is not compilable with nvcc in Release mode.
        if [ "${CMAKE_BUILD_TYPE}" == "Release" ]
        then
            export CMAKE_BUILD_TYPE=Debug
        fi

        # nvcc <= 9.1 does not support boost correctly so fibers have to be disabled.
        if (( (( ALPAKA_CUDA_VER_MAJOR < 9 )) || ( (( ALPAKA_CUDA_VER_MAJOR == 9 )) && (( ALPAKA_CUDA_VER_MINOR <= 1 )) ) ))
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
        if [ "${CXX}" != "clang++" ]
        then
            # We can only use clang as a CUDA compiler when clang is used as the main compiler.
            # For gcc we have to use nvcc.
            echo Using clang as CUDA compiler is only possible if clang is the host compiler!
            exit 1
        fi

        # clang <= 3.7 does not support native CUDA compilation.
        # clang 3.8 used as CUDA compiler is not supported (no cuRand support).
        if (( (( ALPAKA_CI_CLANG_VER_MAJOR == 3 )) && (( ALPAKA_CI_CLANG_VER_MINOR <= 8 )) ))
        then
            echo clang "${ALPAKA_CI_CLANG_VER}" used as CUDA compiler is not supported!
            exit 1
        fi

        # clang <= 3.9 used as CUDA compiler does not support CUDA 8.0+.
        if (( (( ALPAKA_CI_CLANG_VER_MAJOR == 3 )) && (( ALPAKA_CI_CLANG_VER_MINOR <= 9 )) ))
        then
            if (( ALPAKA_CUDA_VER_MAJOR >= 8 ))
            then
                echo clang "${ALPAKA_CI_CLANG_VER}" used as CUDA compiler does not support CUDA "${ALPAKA_CUDA_VER}"!
                exit 1
            fi
        fi

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
