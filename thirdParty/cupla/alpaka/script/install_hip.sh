#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

: "${ALPAKA_CI_HIP_ROOT_DIR?'ALPAKA_CI_HIP_ROOT_DIR must be specified'}"
: "${ALPAKA_CI_HIP_BRANCH?'ALPAKA_CI_HIP_BRANCH must be specified'}"
: "${CMAKE_BUILD_TYPE?'CMAKE_BUILD_TYPE must be specified'}"
: "${CXX?'CXX must be specified'}"
: "${CC?'CC must be specified'}"
: "${ALPAKA_CI_CMAKE_DIR?'ALPAKA_CI_CMAKE_DIR must be specified'}"

# CMake
export PATH=${ALPAKA_CI_CMAKE_DIR}/bin:${PATH}
cmake --version

HIP_SOURCE_DIR=${ALPAKA_CI_HIP_ROOT_DIR}/source-hip/

git clone -b "${ALPAKA_CI_HIP_BRANCH}" --quiet --recursive --single-branch https://github.com/ROCm-Developer-Tools/HIP.git "${HIP_SOURCE_DIR}"
(cd "${HIP_SOURCE_DIR}"; mkdir -p build; cd build; cmake -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${ALPAKA_CI_HIP_ROOT_DIR}" -DBUILD_TESTING=OFF .. && make && make install)


## rocRAND
export HIP_PLATFORM=nvcc
export HIP_RUNTIME=nvcc
export ROCRAND_SOURCE_DIR=${ALPAKA_CI_HIP_ROOT_DIR}/source-rocrand/
if [ ! -d "${ROCRAND_SOURCE_DIR}" ]
then
    # install it into the HIP install dir
    git clone --quiet --recursive https://github.com/ROCmSoftwarePlatform/rocRAND "${ROCRAND_SOURCE_DIR}"
    (cd "${ROCRAND_SOURCE_DIR}"; mkdir -p build; cd build; cmake -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${ALPAKA_CI_HIP_ROOT_DIR}" -DBUILD_BENCHMARK=OFF -DBUILD_TEST=OFF -DNVGPU_TARGETS="30" -DCMAKE_MODULE_PATH="${ALPAKA_CI_HIP_ROOT_DIR}/cmake" -DHIP_PLATFORM="${HIP_PLATFORM}" .. && make && make install)
fi
