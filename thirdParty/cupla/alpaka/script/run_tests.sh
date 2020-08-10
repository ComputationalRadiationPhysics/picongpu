#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

: "${ALPAKA_ACC_GPU_CUDA_ENABLE?'ALPAKA_ACC_GPU_CUDA_ENABLE must be specified'}"
: "${ALPAKA_ACC_GPU_HIP_ENABLE?'ALPAKA_ACC_GPU_HIP_ENABLE must be specified'}"

if [ ! -z "${OMP_THREAD_LIMIT+x}" ]
then
    echo "OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT}"
fi
if [ ! -z "${OMP_NUM_THREADS+x}" ]
then
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
fi

if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "OFF" ] && [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "OFF" ];
then
    cd build/

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
    then
        ctest -V
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        ctest -V -C ${CMAKE_BUILD_TYPE}
    fi

    cd ..
fi
