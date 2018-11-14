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
set -euo pipefail

# runtime and compile time options
ALPAKA_DOCKER_ENV_LIST=()
ALPAKA_DOCKER_ENV_LIST+=("--env" "CC=${CC}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "CXX=${CXX}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_ANALYSIS=${ALPAKA_CI_ANALYSIS}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_BOOST_BRANCH=${ALPAKA_CI_BOOST_BRANCH}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_BOOST_ROOT_DIR=${ALPAKA_CI_BOOST_ROOT_DIR}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_BOOST_LIB_DIR=${ALPAKA_CI_BOOST_LIB_DIR}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_DIR=${ALPAKA_CI_CLANG_DIR}")
if [ "${CXX}" == "clang++" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_VER=${ALPAKA_CI_CLANG_VER}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_VER_MAJOR=${ALPAKA_CI_CLANG_VER_MAJOR}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_VER_MINOR=${ALPAKA_CI_CLANG_VER_MINOR}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_LIBSTDCPP_VERSION=${ALPAKA_CI_CLANG_LIBSTDCPP_VERSION}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CMAKE_VER=${ALPAKA_CI_CMAKE_VER}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CMAKE_VER_MAJOR=${ALPAKA_CI_CMAKE_VER_MAJOR}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CMAKE_VER_MINOR=${ALPAKA_CI_CMAKE_VER_MINOR}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CMAKE_DIR=${ALPAKA_CI_CMAKE_DIR}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CUDA_DIR=${ALPAKA_CI_CUDA_DIR}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_HIP_ROOT_DIR=${ALPAKA_CI_HIP_ROOT_DIR}")
if [ "${CXX}" == "g++" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_GCC_VER=${ALPAKA_CI_GCC_VER}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_GCC_VER_MAJOR=${ALPAKA_CI_GCC_VER_MAJOR}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_GCC_VER_MINOR=${ALPAKA_CI_GCC_VER_MINOR}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_SANITIZERS=${ALPAKA_CI_SANITIZERS}")
if [[ -v ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE}")
fi
if [[ -v ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE}")
fi
if [[ -v ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}")
fi
if [[ -v ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE}")
fi
if [[ -v ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE}")
fi
if [[ -v ALPAKA_ACC_CPU_BT_OMP4_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_BT_OMP4_ENABLE=${ALPAKA_ACC_CPU_BT_OMP4_ENABLE}")
fi
if [[ -v ALPAKA_ACC_GPU_CUDA_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_GPU_CUDA_ENABLE=${ALPAKA_ACC_GPU_CUDA_ENABLE}")
fi
if [[ -v ALPAKA_ACC_GPU_HIP_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_GPU_HIP_ENABLE=${ALPAKA_ACC_GPU_HIP_ENABLE}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_HIP_PLATFORM=${ALPAKA_HIP_PLATFORM}")
fi
if [[ -v ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE}")
fi
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "ON" ] || [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "ON" ] && [ "${ALPAKA_HIP_PLATFORM}" == "nvcc" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_VERSION=${ALPAKA_CUDA_VERSION}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_COMPILER=${ALPAKA_CUDA_COMPILER}")
fi
if [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "ON" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_HIP_ARCH=${ALPAKA_HIP_ARCH}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_HIP_BRANCH=${ALPAKA_CI_HIP_BRANCH}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_HIP_ROOT_DIR=${ALPAKA_CI_HIP_ROOT_DIR}")
fi

# runtime only options
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI=${ALPAKA_CI}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_DEBUG=${ALPAKA_DEBUG}")
if [[ -v OMP_NUM_THREADS ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "OMP_NUM_THREADS=${OMP_NUM_THREADS}")
fi
if [[ -v ALPAKA_ACC_GPU_CUDA_ONLY_MODE ]]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_GPU_CUDA_ONLY_MODE=${ALPAKA_ACC_GPU_CUDA_ONLY_MODE}")
fi
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "ON" ] # here, no HIP querying required, as ALPAKA_HIP_ARCH is used instead
then
    if [[ -v ALPAKA_CUDA_ARCH ]]
    then
        ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_ARCH=${ALPAKA_CUDA_ARCH}")
    fi
fi
if [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "ON" ]
then
    if [[ -v ALPAKA_HIP_ARCH ]]
    then
        ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_HIP_ARCH=${ALPAKA_HIP_ARCH}")
    fi
fi

docker images
docker images -q ${ALPAKA_CI_DOCKER_IMAGE_NAME}

# If we have created the image in the current run, we do not have to load it again, because it is already available.
if [[ "$(docker images -q ${ALPAKA_CI_DOCKER_IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    gzip -dc "${ALPAKA_CI_DOCKER_CACHE_IMAGE_FILE_PATH}" | docker load
fi

docker run -v "$(pwd)":"$(pwd)" -w "$(pwd)" "${ALPAKA_DOCKER_ENV_LIST[@]}" --rm "${ALPAKA_CI_DOCKER_IMAGE_NAME}" /bin/bash ./script/travis/run.sh
