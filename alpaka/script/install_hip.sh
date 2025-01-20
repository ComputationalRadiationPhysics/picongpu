#!/bin/bash

#
# Copyright 2022 Rene Widera, Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: install_hip>"

: "${ALPAKA_CI_HIP_ROOT_DIR?'ALPAKA_CI_HIP_ROOT_DIR must be specified'}"
: "${ALPAKA_CI_HIP_VERSION?'ALPAKA_CI_HIP_VERSION must be specified'}"

function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }

if agc-manager -e rocm@${ALPAKA_CI_HIP_VERSION} ; then
    echo_green "<USE: preinstalled ROCm ${ALPAKA_CI_HIP_VERSION}>"
    export ROCM_PATH=$(agc-manager -b rocm@${ALPAKA_CI_HIP_VERSION})
else
    echo_yellow "<INSTALL: ROCm ${ALPAKA_CI_HIP_VERSION}>"

    travis_retry apt-get -y --quiet update
    travis_retry apt-get -y --quiet install wget gnupg2
    # AMD container keys are outdated and must be updated
    source /etc/os-release
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    echo "deb https://repo.radeon.com/rocm/apt/${ALPAKA_CI_HIP_VERSION} ${VERSION_CODENAME} main" | sudo tee -a /etc/apt/sources.list.d/rocm.list
    travis_retry apt-get -y --quiet update

    ALPAKA_CI_ROCM_VERSION=$ALPAKA_CI_HIP_VERSION
    # append .0 if no patch level is defined
    if ! echo $ALPAKA_CI_ROCM_VERSION | grep -Eq '[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+'; then
        ALPAKA_CI_ROCM_VERSION="${ALPAKA_CI_ROCM_VERSION}.0"
    fi

    apt install --no-install-recommends -y rocm-llvm${ALPAKA_CI_ROCM_VERSION} hip-runtime-amd${ALPAKA_CI_ROCM_VERSION} rocm-dev${ALPAKA_CI_ROCM_VERSION} rocm-utils${ALPAKA_CI_ROCM_VERSION} rocrand-dev${ALPAKA_CI_ROCM_VERSION} rocminfo${ALPAKA_CI_ROCM_VERSION} rocm-cmake${ALPAKA_CI_ROCM_VERSION} rocm-device-libs${ALPAKA_CI_ROCM_VERSION} rocm-core${ALPAKA_CI_ROCM_VERSION} rocm-smi-lib${ALPAKA_CI_ROCM_VERSION}
    if [ $(version ${ALPAKA_CI_ROCM_VERSION}) -ge $(version "6.0.0") ]; then
        apt install --no-install-recommends -y hiprand-dev${ALPAKA_CI_ROCM_VERSION}
    fi
    export ROCM_PATH=/opt/rocm
fi
# ROCM_PATH required by HIP tools
export HIP_PLATFORM="amd"
export HIP_DEVICE_LIB_PATH=${ROCM_PATH}/amdgcn/bitcode
export HSA_PATH=$ROCM_PATH

export PATH=${ROCM_PATH}/bin:$PATH
export PATH=${ROCM_PATH}/llvm/bin:$PATH

# Workaround if clang uses the stdlibc++. The stdlibc++-9 does not support C++20, therefore we install the stdlibc++-11. Clang automatically uses the latest stdlibc++ version.
if [[ "$(cat /etc/os-release)" =~ "20.04" ]] && [ "${alpaka_CXX_STANDARD}" == "20" ];
then
    travis_retry sudo apt install -y --no-install-recommends software-properties-common
    sudo apt-add-repository ppa:ubuntu-toolchain-r/test -y
    travis_retry sudo apt update
    travis_retry sudo apt install -y --no-install-recommends g++-11
fi

sudo update-alternatives --install /usr/bin/clang clang ${ROCM_PATH}/llvm/bin/clang 50
sudo update-alternatives --install /usr/bin/clang++ clang++ ${ROCM_PATH}/llvm/bin/clang++ 50
sudo update-alternatives --install /usr/bin/cc cc ${ROCM_PATH}/llvm/bin/clang 50
sudo update-alternatives --install /usr/bin/c++ c++ ${ROCM_PATH}/llvm/bin/clang++ 50

export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hiprand/lib:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=${ROCM_PATH}:${ROCM_PATH}/hiprand:${ROCM_PATH}/hip:${CMAKE_PREFIX_PATH:-}

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
    export CMAKE_HIP_ARCHITECTURES="gfx906"
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

# use the clang++ of the HIP SDK as C++ compiler
export CMAKE_CXX_COMPILER=$(which clang++)
