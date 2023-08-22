#!/bin/bash

#
# Copyright 2022 Rene Widera, Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_HIP_ROOT_DIR?'ALPAKA_CI_HIP_ROOT_DIR must be specified'}"
: "${ALPAKA_CI_HIP_VERSION?'ALPAKA_CI_HIP_VERSION must be specified'}"

if agc-manager -e rocm@${ALPAKA_CI_HIP_VERSION} ; then
    export ROCM_PATH=$(agc-manager -b rocm@${ALPAKA_CI_HIP_VERSION})
else
    travis_retry apt-get -y --quiet update
    travis_retry apt-get -y --quiet install wget gnupg2
    # AMD container keys are outdated and must be updated
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    travis_retry apt-get -y --quiet update

    # AMD container are not shipped with rocrand/hiprand
    travis_retry sudo apt-get -y --quiet install rocrand
    export ROCM_PATH=/opt/rocm
fi
# ROCM_PATH required by HIP tools
export HIP_PATH=${ROCM_PATH}/hip

export HIP_LIB_PATH=${HIP_PATH}/lib

export PATH=${ROCM_PATH}/bin:$PATH
export PATH=${ROCM_PATH}/llvm/bin:$PATH

sudo update-alternatives --install /usr/bin/clang clang ${ROCM_PATH}/llvm/bin/clang 50
sudo update-alternatives --install /usr/bin/clang++ clang++ ${ROCM_PATH}/llvm/bin/clang++ 50
sudo update-alternatives --install /usr/bin/cc cc ${ROCM_PATH}/llvm/bin/clang 50
sudo update-alternatives --install /usr/bin/c++ c++ ${ROCM_PATH}/llvm/bin/clang++ 50

export LD_LIBRARY_PATH=${ROCM_PATH}/lib64:${ROCM_PATH}/hiprand/lib:${LD_LIBRARY_PATH}:${ROCM_PATH}/llvm/lib
export CMAKE_PREFIX_PATH=${ROCM_PATH}:${ROCM_PATH}/hiprand:${CMAKE_PREFIX_PATH:-}

if [[ "$CI_RUNNER_TAGS" =~ .*cpuonly.* ]] ; then
    # In cases where the compile-only job is executed on a GPU runner but with different kinds of accelerators
    # we need to reset the variables to avoid compiling for the wrong architecture and accelerator.
    unset CI_GPUS
    unset CI_GPU_ARCH
fi

if ! [ -z ${CI_GPUS+x} ] && [ -n "$CI_GPUS" ] ; then
    # select randomly a device if multiple exists
    # CI_GPUS is provided by the gitlab CI runner
    HIP_SELECTED_DEVICE_ID=$((RANDOM%CI_GPUS))
    export HIP_VISIBLE_DEVICES=$HIP_SELECTED_DEVICE_ID
    echo "selected HIP device '$HIP_VISIBLE_DEVICES' of '$CI_GPUS'"
else
    echo "No GPU device selected because environment variable CI_GPUS is not set."
fi

if [ -z ${CI_GPU_ARCH+x} ] ; then
    # In case the runner is not providing a GPU architecture e.g. a CPU runner set the architecture
    # to Radeon VII or MI50/60.
    export GPU_TARGETS="gfx906"
fi

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
