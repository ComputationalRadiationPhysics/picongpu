#!/bin/bash

#
# Copyright 2017-2026 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: run_tests>"

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

    # use a string rather than an array: with "set -u", bash 3.2 (the default on macOS) treats an empty array as unset
    CTEST_ARGS=""
    # skip the cooperative kernel (grid synchronisation) tests with HIP/ROCm:
    #   - on gfx1100 the tests should pass with ROCm 7.0 and later;
    #   - on gfx1030 the tests may randomly hang.
    if [ "${alpaka_ACC_GPU_HIP_ENABLE}" == "ON" ]
    then
        echo_yellow "<SKIP: grid synchronisation tests with HIP/ROCm ${ALPAKA_CI_HIP_VERSION:-}>"
        CTEST_ARGS="${CTEST_ARGS} -E ^(gridSyncTest|helloWorldGridSync)$"
    fi

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
    then
        ctest --output-on-failure ${CTEST_ARGS}
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        ctest --output-on-failure -C ${CMAKE_BUILD_TYPE} ${CTEST_ARGS}
    fi

    cd ..
fi
