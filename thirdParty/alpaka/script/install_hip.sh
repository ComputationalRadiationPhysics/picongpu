#!/bin/bash

#
# Copyright 2022 Rene Widera, Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: install_hip>"

: "${ALPAKA_CI_HIP_VERSION?'ALPAKA_CI_HIP_VERSION must be specified'}"

function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }

if agc-manager -e rocm@${ALPAKA_CI_HIP_VERSION} ; then
    echo_green "<USE: preinstalled ROCm ${ALPAKA_CI_HIP_VERSION}>"
    export ROCM_PATH=$(agc-manager -b rocm@${ALPAKA_CI_HIP_VERSION})
else
    echo_yellow "<INSTALL: ROCm ${ALPAKA_CI_HIP_VERSION}>"

    retry_cmd apt-get -y --quiet update
    retry_cmd apt-get -y --quiet install wget gnupg2

    if [ "$(version "${ALPAKA_CI_HIP_VERSION}")" -lt "$(version "7.3.0")" ]; then
        # ALPAKA_CI_ROCM_REPO_VERSION is the name of the repository directory, ALPAKA_CI_ROCM_VERSION is the version
        # suffix of the packages. Use the latest patch release of each ROCm version, which has its own X.Y.Z
        # repository. For other versions use the X.Y repository, which contains the X.Y.0 packages: there are no
        # X.Y.0 directories.
        case "${ALPAKA_CI_HIP_VERSION}" in
            6.0) ALPAKA_CI_ROCM_REPO_VERSION=6.0.3 ;;
            6.1) ALPAKA_CI_ROCM_REPO_VERSION=6.1.5 ;;
            6.2) ALPAKA_CI_ROCM_REPO_VERSION=6.2.4 ;;
            6.3) ALPAKA_CI_ROCM_REPO_VERSION=6.3.4 ;;
            6.4) ALPAKA_CI_ROCM_REPO_VERSION=6.4.4 ;;
            # the 7.0.3 repository contains the 7.0.2 packages
            7.0) ALPAKA_CI_ROCM_REPO_VERSION=7.0.2 ;;
            7.1) ALPAKA_CI_ROCM_REPO_VERSION=7.1.1 ;;
            7.2) ALPAKA_CI_ROCM_REPO_VERSION=7.2.4 ;;
            *) ALPAKA_CI_ROCM_REPO_VERSION=${ALPAKA_CI_HIP_VERSION} ;;
        esac
        ALPAKA_CI_ROCM_VERSION=${ALPAKA_CI_ROCM_REPO_VERSION}
        # append .0 if no patch level is defined
        if ! echo $ALPAKA_CI_ROCM_VERSION | grep -Eq '[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+'; then
            ALPAKA_CI_ROCM_VERSION="${ALPAKA_CI_ROCM_VERSION}.0"
        fi
        echo_green "<INSTALL: ROCm ${ALPAKA_CI_ROCM_VERSION}>"

        # AMD container keys are outdated and must be updated
        source /etc/os-release
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
        echo "deb https://repo.radeon.com/rocm/apt/${ALPAKA_CI_ROCM_REPO_VERSION} ${VERSION_CODENAME} main" | sudo tee -a /etc/apt/sources.list.d/rocm.list
        retry_cmd apt-get -y --quiet update

        apt install --no-install-recommends -y rocm-llvm${ALPAKA_CI_ROCM_VERSION} hip-runtime-amd${ALPAKA_CI_ROCM_VERSION} rocm-dev${ALPAKA_CI_ROCM_VERSION} rocm-utils${ALPAKA_CI_ROCM_VERSION} rocrand-dev${ALPAKA_CI_ROCM_VERSION} rocminfo${ALPAKA_CI_ROCM_VERSION} rocm-cmake${ALPAKA_CI_ROCM_VERSION} rocm-device-libs${ALPAKA_CI_ROCM_VERSION} rocm-core${ALPAKA_CI_ROCM_VERSION} rocm-smi-lib${ALPAKA_CI_ROCM_VERSION}
        if [ $(version ${ALPAKA_CI_ROCM_VERSION}) -ge $(version "6.0.0") ]; then
            apt install --no-install-recommends -y hiprand-dev${ALPAKA_CI_ROCM_VERSION}
        fi
    elif [ "$(version "${ALPAKA_CI_HIP_VERSION}")" -ge "$(version "7.14.0")" ]; then
        # ROCm 7.14 is distributed from the multi-arch repository,
        # ROCm 10.0 and later from the stable repository.
        if [ "$(version "${ALPAKA_CI_HIP_VERSION}")" -ge "$(version "10.0.0")" ]; then
            ROCM_APT_REPO_URL=https://stable.repo.amd.com/rocm/core/packages
            ROCM_APT_KEY_URL=https://stable.repo.amd.com/rocm/gpg/packages.gpg
        else
            ROCM_APT_REPO_URL=https://repo.amd.com/rocm/packages-multi-arch
            ROCM_APT_KEY_URL=https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg
        fi

        sudo mkdir --parents --mode=0755 /etc/apt/keyrings
        wget "${ROCM_APT_KEY_URL}" -O - |
            gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg >/dev/null

        # Prevents apt warnings when the script is run a second time.
        # Delete and recreate the source list to ensure that the correct apt sources are set.
        if [[ -f /etc/apt/sources.list.d/rocm.list ]]; then
            sudo rm -rf /etc/apt/sources.list.d/rocm.list
        fi

        # require to set environment variable VERSION_ID
        source /etc/os-release

        sudo tee /etc/apt/sources.list.d/rocm.list <<EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] ${ROCM_APT_REPO_URL}/ubuntu${VERSION_ID//./} stable main
EOF

        retry_cmd sudo DEBIAN_FRONTEND=noninteractive apt update

        # If configured, install rocm only for a specific GPU architecture. Otherwise install it for all architectures.
        if [[ -n "${CMAKE_HIP_ARCHITECTURES}" ]]; then
            ROCM_PACKAGE_VERSION="${ALPAKA_CI_HIP_VERSION}-${CMAKE_HIP_ARCHITECTURES}"
        else
            ROCM_PACKAGE_VERSION="${ALPAKA_CI_HIP_VERSION}"
        fi

        # TODO: It is not the minimal installation. There are many libraries, like fft and dnn are installed, which do not require.
        sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
            "amdrocm-core-dev${ROCM_PACKAGE_VERSION}" "amdrocm-core${ROCM_PACKAGE_VERSION}"

        unset ROCM_PACKAGE_VERSION ROCM_APT_REPO_URL ROCM_APT_KEY_URL
    else
        echo_red "ERROR: Installing ROCm 7.9 - 7.13 is not supported"
        exit 1
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
    retry_cmd sudo apt install -y --no-install-recommends software-properties-common
    sudo apt-add-repository ppa:ubuntu-toolchain-r/test -y
    retry_cmd sudo apt update
    retry_cmd sudo apt install -y --no-install-recommends g++-11
fi

sudo update-alternatives --install /usr/bin/clang clang ${ROCM_PATH}/lib/llvm/bin/clang 50
sudo update-alternatives --install /usr/bin/clang++ clang++ ${ROCM_PATH}/lib/llvm/bin/clang++ 50
sudo update-alternatives --install /usr/bin/cc cc ${ROCM_PATH}/lib/llvm/bin/clang 50
sudo update-alternatives --install /usr/bin/c++ c++ ${ROCM_PATH}/lib/llvm/bin/clang++ 50

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
