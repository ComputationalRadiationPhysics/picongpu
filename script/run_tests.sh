#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

source ./script/set.sh

: "${alpaka_ACC_GPU_CUDA_ENABLE?'alpaka_ACC_GPU_CUDA_ENABLE must be specified'}"
: "${alpaka_ACC_GPU_HIP_ENABLE?'alpaka_ACC_GPU_HIP_ENABLE must be specified'}"

if [ ! -z "${OMP_THREAD_LIMIT+x}" ]
then
    echo "OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT}"
fi
if [ ! -z "${OMP_NUM_THREADS+x}" ]
then
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
fi

# in the GitLab CI, all runtime tests are possible
if [[ ! -z "${GITLAB_CI+x}" || ("${alpaka_ACC_GPU_CUDA_ENABLE}" == "OFF" && "${alpaka_ACC_GPU_HIP_ENABLE}" == "OFF" ) ]];
then
    cd build/

    if [ "${CMAKE_CXX_COMPILER:-}" = "nvc++" ] || [ "${alpaka_ACC_GPU_CUDA_ENABLE}" == "ON" ]
    then
        # show gpu info in gitlab CI
        nvidia-smi || true
        # # enbale CUDA API logs for offload
        # export NVCOMPILER_ACC_NOTIFY=3 # exceeds mximum log length
    fi

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
    then
        ctest -V
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        ctest -V -C ${CMAKE_BUILD_TYPE}
    fi

    cd ..
fi
