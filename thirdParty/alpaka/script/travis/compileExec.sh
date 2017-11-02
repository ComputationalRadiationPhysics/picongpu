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
# Build and execute all tests.
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
    -DALPAKA_ACC_GPU_CUDA_ENABLE="${ALPAKA_ACC_GPU_CUDA_ENABLE}" -DALPAKA_CUDA_VERSION="${ALPAKA_CUDA_VER}" -DALPAKA_CUDA_COMPILER="${ALPAKA_CUDA_COMPILER}" -DALPAKA_ACC_GPU_CUDA_ONLY_MODE="${ALPAKA_ACC_GPU_CUDA_ONLY_MODE}" -DALPAKA_CUDA_ARCH="${ALPAKA_CUDA_ARCH}"\
    -DALPAKA_DEBUG="${ALPAKA_DEBUG}" -DALPAKA_CI=ON \
    "../../"
make VERBOSE=1
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "OFF" ]
then
    eval "${2}"
fi

cd "${oldPath}"
