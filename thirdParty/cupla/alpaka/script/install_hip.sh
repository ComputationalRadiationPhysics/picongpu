#!/bin/bash

#
# Copyright 2021 Rene Widera
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_HIP_ROOT_DIR?'ALPAKA_CI_HIP_ROOT_DIR must be specified'}"

travis_retry apt-get -y --quiet update
travis_retry apt-get -y --quiet install wget gnupg2
# AMD container keys are outdated and must be updated
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
travis_retry apt-get -y --quiet update

# AMD container are not shipped with rocrand/hiprand
travis_retry sudo apt-get -y --quiet install rocrand

# ROCM_PATH required by HIP tools
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip

export HIP_LIB_PATH=${HIP_PATH}/lib

export PATH=${ROCM_PATH}/bin:$PATH
export PATH=${ROCM_PATH}/llvm/bin:$PATH

sudo update-alternatives --install /usr/bin/clang clang ${ROCM_PATH}/llvm/bin/clang 50
sudo update-alternatives --install /usr/bin/clang++ clang++ ${ROCM_PATH}/llvm/bin/clang++ 50
sudo update-alternatives --install /usr/bin/cc cc ${ROCM_PATH}/llvm/bin/clang 50
sudo update-alternatives --install /usr/bin/c++ c++ ${ROCM_PATH}/llvm/bin/clang++ 50

export LD_LIBRARY_PATH=${ROCM_PATH}/lib64:${ROCM_PATH}/hiprand/lib:${LD_LIBRARY_PATH}:${ROCM_PATH}/llvm/lib
export CMAKE_PREFIX_PATH=${ROCM_PATH}:${ROCM_PATH}/hiprand:${CMAKE_PREFIX_PATH:-}

# environment overview
which clang++
clang++ --version
which hipconfig
hipconfig --platform
echo
hipconfig -v
echo
hipconfig
rocm-smi
# print newline as previous command does not do this
echo
