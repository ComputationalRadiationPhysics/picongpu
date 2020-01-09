#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis/set.sh

: "${ALPAKA_ACC_GPU_CUDA_ENABLE?'ALPAKA_ACC_GPU_CUDA_ENABLE must be specified'}"
: "${ALPAKA_ACC_GPU_HIP_ENABLE?'ALPAKA_ACC_GPU_HIP_ENABLE must be specified'}"

if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "OFF" ] && [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "OFF" ];
then
    cd build/

    if [ "$TRAVIS_OS_NAME" = "linux" ] || [ "$TRAVIS_OS_NAME" = "osx" ]
    then
        ctest -V
    elif [ "$TRAVIS_OS_NAME" = "windows" ]
    then
        ctest -V -C ${CMAKE_BUILD_TYPE}
    fi

    cd ..
fi
