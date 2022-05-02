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
source ./script/docker_retry.sh

# runtime and compile time options
ALPAKA_DOCKER_ENV_LIST=()
ALPAKA_DOCKER_ENV_LIST+=("--env" "CC=${CC}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "CXX=${CXX}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_OS_NAME=${ALPAKA_CI_OS_NAME}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_ANALYSIS=${ALPAKA_CI_ANALYSIS}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_TBB_VERSION=${ALPAKA_CI_TBB_VERSION}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_BOOST_BRANCH=${ALPAKA_CI_BOOST_BRANCH}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "BOOST_ROOT=${BOOST_ROOT}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_BOOST_LIB_DIR=${ALPAKA_CI_BOOST_LIB_DIR}")
if [ ! -z "${ALPAKA_CI_CLANG_VER+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CLANG_VER=${ALPAKA_CI_CLANG_VER}")
fi
if [ ! -z "${ALPAKA_CI_BUILD_JOBS+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_BUILD_JOBS=${ALPAKA_CI_BUILD_JOBS}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_STDLIB=${ALPAKA_CI_STDLIB}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CMAKE_VER=${ALPAKA_CI_CMAKE_VER}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CMAKE_DIR=${ALPAKA_CI_CMAKE_DIR}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_RUN_TESTS=${ALPAKA_CI_RUN_TESTS}")
if [ ! -z "${CMAKE_CXX_FLAGS+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
fi
if [ ! -z "${CMAKE_C_COMPILER+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
fi
if [ ! -z "${CMAKE_CXX_COMPILER+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
fi
if [ ! -z "${CMAKE_EXE_LINKER_FLAGS+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
fi
if [ ! -z "${CMAKE_CXX_EXTENSIONS+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}")
fi
if [ ! -z "${ALPAKA_CI_GCC_VER+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_GCC_VER=${ALPAKA_CI_GCC_VER}")
fi
if [ ! -z "${ALPAKA_CI_SANITIZERS+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_SANITIZERS=${ALPAKA_CI_SANITIZERS}")
fi
if [ ! -z "${ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=${alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE}")
fi
if [ ! -z "${alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE=${alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE}")
fi
if [ ! -z "${alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=${alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}")
fi
if [ ! -z "${alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE=${alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE}")
fi
if [ ! -z "${alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE=${alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE}")
fi
if [ ! -z "${alpaka_ACC_ANY_BT_OMP5_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_ANY_BT_OMP5_ENABLE=${alpaka_ACC_ANY_BT_OMP5_ENABLE}")
fi
if [ ! -z "${alpaka_ACC_ANY_BT_OACC_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_ANY_BT_OACC_ENABLE=${alpaka_ACC_ANY_BT_OACC_ENABLE}")
fi
if [ ! -z "${alpaka_OFFLOAD_MAX_BLOCK_SIZE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_OFFLOAD_MAX_BLOCK_SIZE=${alpaka_OFFLOAD_MAX_BLOCK_SIZE}")
fi
if [ ! -z "${alpaka_DEBUG_OFFLOAD_ASSUME_HOST+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_DEBUG_OFFLOAD_ASSUME_HOST=${alpaka_DEBUG_OFFLOAD_ASSUME_HOST}")
fi
if [ ! -z "${alpaka_ACC_GPU_CUDA_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_GPU_CUDA_ENABLE=${alpaka_ACC_GPU_CUDA_ENABLE}")
fi
if [ ! -z "${alpaka_ACC_GPU_HIP_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_GPU_HIP_ENABLE=${alpaka_ACC_GPU_HIP_ENABLE}")
fi
if [ ! -z "${alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE=${alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_CUDA=${ALPAKA_CI_INSTALL_CUDA}")
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CUDA_DIR=${ALPAKA_CI_CUDA_DIR}")
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_CUDA_VERSION=${ALPAKA_CI_CUDA_VERSION}")
    if [ ! -z "${CMAKE_CUDA_COMPILER+x}" ]
    then
        ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
    fi
    if [ ! -z "${CMAKE_CUDA_ARCHITECTURES+x}" ]
    then
        ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    fi
    if [ ! -z "${CMAKE_CUDA_FLAGS+x}" ]
    then
        ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
    fi
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_HIP=${ALPAKA_CI_INSTALL_HIP}")
if [ "${ALPAKA_CI_INSTALL_HIP}" == "ON" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_HIP_ROOT_DIR=${ALPAKA_CI_HIP_ROOT_DIR}")
fi
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_TBB=${ALPAKA_CI_INSTALL_TBB}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_OMP=${ALPAKA_CI_INSTALL_OMP}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_FIBERS=${ALPAKA_CI_INSTALL_FIBERS}")
ALPAKA_DOCKER_ENV_LIST+=("--env" "ALPAKA_CI_INSTALL_ATOMIC=${ALPAKA_CI_INSTALL_ATOMIC}")

# runtime only options
ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_CI=${alpaka_CI}")
if [ ! -z "${alpaka_DEBUG+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_DEBUG=${alpaka_DEBUG}")
fi
if [ ! -z "${alpaka_CXX_STANDARD+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_CXX_STANDARD=${alpaka_CXX_STANDARD}")
fi
if [ ! -z "${OMP_NUM_THREADS+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "OMP_NUM_THREADS=${OMP_NUM_THREADS}")
fi
if [ ! -z "${alpaka_ACC_GPU_CUDA_ONLY_MODE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_GPU_CUDA_ONLY_MODE=${alpaka_ACC_GPU_CUDA_ONLY_MODE}")
fi
if [ ! -z "${alpaka_ACC_GPU_HIP_ONLY_MODE+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_ACC_GPU_HIP_ONLY_MODE=${alpaka_ACC_GPU_HIP_ONLY_MODE}")
fi
if [ ! -z "${alpaka_CUDA_FAST_MATH+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_CUDA_FAST_MATH=${alpaka_CUDA_FAST_MATH}")
fi
if [ ! -z "${alpaka_CUDA_FTZ+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_CUDA_FTZ=${alpaka_CUDA_FTZ}")
fi
if [ ! -z "${alpaka_CUDA_SHOW_REGISTER+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_CUDA_SHOW_REGISTER=${alpaka_CUDA_SHOW_REGISTER}")
fi
if [ ! -z "${alpaka_CUDA_KEEP_FILES+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_CUDA_KEEP_FILES=${alpaka_CUDA_KEEP_FILES}")
fi
if [ ! -z "${alpaka_CUDA_EXPT_EXTENDED_LAMBDA+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "alpaka_CUDA_EXPT_EXTENDED_LAMBDA=${alpaka_CUDA_EXPT_EXTENDED_LAMBDA}")
fi
if [ ! -z "${CMAKE_CUDA_SEPARABLE_COMPILATION+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_CUDA_SEPARABLE_COMPILATION=${CMAKE_CUDA_SEPARABLE_COMPILATION}")
fi
if [ ! -z "${CMAKE_INSTALL_PREFIX+x}" ]
then
    ALPAKA_DOCKER_ENV_LIST+=("--env" "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
fi

docker_retry docker run -v "$(pwd)":"$(pwd)" -w "$(pwd)" "${ALPAKA_DOCKER_ENV_LIST[@]}" "${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}" /bin/bash -c "source ./script/install.sh && ./script/run.sh"
