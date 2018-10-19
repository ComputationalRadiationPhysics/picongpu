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

ls "${ALPAKA_CI_DOCKER_CACHE_DIR}"

ALPAKA_DOCKER_BUILD_REQUIRED=1

if [ -f "${ALPAKA_CI_DOCKER_CACHE_IMAGE_FILE_PATH}" ]
then
    # NOTE: The image being available is not the only precondition. If anything within any of the scripts has changed in comparison to the ones that created the docker image, we might have to rebuild the image.
    ALPAKA_DOCKER_BUILD_REQUIRED=0
fi

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
if [ "${CXX}" == "g++" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_GCC_VER=${ALPAKA_CI_GCC_VER}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_GCC_VER_MAJOR=${ALPAKA_CI_GCC_VER_MAJOR}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_GCC_VER_MINOR=${ALPAKA_CI_GCC_VER_MINOR}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_SANITIZERS=${ALPAKA_CI_SANITIZERS}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_BT_OMP4_ENABLE=${ALPAKA_ACC_CPU_BT_OMP4_ENABLE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_GPU_CUDA_ENABLE=${ALPAKA_ACC_GPU_CUDA_ENABLE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE=${ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE}")
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "ON" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_VER=${ALPAKA_CUDA_VER}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_VER_MAJOR=${ALPAKA_CUDA_VER_MAJOR}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_VER_MINOR=${ALPAKA_CUDA_VER_MINOR}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CUDA_COMPILER=${ALPAKA_CUDA_COMPILER}")
fi

if [ "${ALPAKA_DOCKER_BUILD_REQUIRED}" -eq 1 ]
then
  docker run -v "$(pwd)":"$(pwd)" -w "$(pwd)" "${ALPAKA_DOCKER_ENV_LIST[@]}" "${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}" /bin/bash ./script/travis/install.sh

  ALPAKA_DOCKER_CONTAINER_NAME=$(docker ps -l -q)
  docker commit "${ALPAKA_DOCKER_CONTAINER_NAME}" "${ALPAKA_CI_DOCKER_IMAGE_NAME}"

  # delete the container and the base image to save disc space
  docker stop "${ALPAKA_DOCKER_CONTAINER_NAME}"
  docker rm "${ALPAKA_DOCKER_CONTAINER_NAME}"
  docker rmi "${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}"

  docker save "${ALPAKA_CI_DOCKER_IMAGE_NAME}" | gzip > "${ALPAKA_CI_DOCKER_CACHE_IMAGE_FILE_PATH}"

  docker images
fi
