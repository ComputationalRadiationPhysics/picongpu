#!/bin/bash

#
# Copyright 2014-2017 Benjamin Worpitz
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

# Compiles the project within the directory given by ${1} and executes ${2} within the build folder.

#-------------------------------------------------------------------------------
# e: exit as soon as one command returns a non-zero exit code.
set -eo pipefail

#-------------------------------------------------------------------------------

# create a cmake variable definition if an environment variable exists
#
# This function can not handle environment variables with spaces in its content.
#
# @param $1 environment variable name
# @param $2 cmake variable name (optional)
#           if not defined than cmake variable name is equal to environment name
# 
# @result if $2 exists cmake variable definition else nothing is returned
#
# @code{.bash}
# FOO=ON
# echo "$(env2cmake FOO)" # returns "-DFOO=ON"
# echo "$(env2cmake FOO CMAKE_FOO_DEF)" # returns "-DCMAKE_FOO_DEF=ON"
# echo "$(env2cmake BAR)" # returns nothing
# @endcode
function env2cmake()
{
    if [ $# -ne 2 ] ; then
        cmakeName=$1
    else
        cmakeName=$2
    fi
    if [ -v "$1" ] ; then
        echo -n "-D$cmakeName=${!1}"
    fi
}

#-------------------------------------------------------------------------------
# Build and execute all tests.
echo KMP_DEVICE_THREAD_LIMIT=${KMP_DEVICE_THREAD_LIMIT}
echo KMP_ALL_THREADS=${KMP_ALL_THREADS}
echo KMP_TEAMS_THREAD_LIMIT=${KMP_TEAMS_THREAD_LIMIT}
echo OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT}
echo OMP_NUM_THREADS=${OMP_NUM_THREADS}
oldPath=${PWD}
cd "${1}"
mkdir --parents build/make/
cd build/make/
cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}" \
    -DBOOST_ROOT="${ALPAKA_CI_BOOST_ROOT_DIR}" -DBOOST_LIBRARYDIR="${ALPAKA_CI_BOOST_LIB_DIR}/lib" -DBoost_USE_STATIC_LIBS=ON -DBoost_USE_MULTITHREADED=ON -DBoost_USE_STATIC_RUNTIME=OFF \
    -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE="${ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE}" -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE="${ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE}" -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE="${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}" \
    -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE="${ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE}"\
    -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE="${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE}" -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE="${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE}" -DALPAKA_ACC_CPU_BT_OMP4_ENABLE="${ALPAKA_ACC_CPU_BT_OMP4_ENABLE}" \
    "$(env2cmake ALPAKA_ACC_GPU_CUDA_ENABLE)" "$(env2cmake ALPAKA_CUDA_VER ALPAKA_CUDA_VERSION)" "$(env2cmake ALPAKA_ACC_GPU_CUDA_ONLY_MODE)" "$(env2cmake ALPAKA_CUDA_ARCH)" "$(env2cmake ALPAKA_CUDA_COMPILER)" \
    -DALPAKA_DEBUG="${ALPAKA_DEBUG}" -DALPAKA_CI=ON \
    "../../"
make VERBOSE=1
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "OFF" ]
then
    eval "${2}"
fi

cd "${oldPath}"
