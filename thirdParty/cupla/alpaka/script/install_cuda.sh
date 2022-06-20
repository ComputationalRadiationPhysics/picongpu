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

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_CUDA_VERSION?'ALPAKA_CI_CUDA_VERSION must be specified'}"

ALPAKA_CUDA_VER_SEMANTIC=( ${ALPAKA_CI_CUDA_VERSION//./ } )
ALPAKA_CUDA_VER_MAJOR="${ALPAKA_CUDA_VER_SEMANTIC[0]}"
echo ALPAKA_CUDA_VER_MAJOR: "${ALPAKA_CUDA_VER_MAJOR}"

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    : "${ALPAKA_CI_CUDA_DIR?'ALPAKA_CI_CUDA_DIR must be specified'}"
    : "${CMAKE_CUDA_COMPILER?'CMAKE_CUDA_COMPILER must be specified'}"

    # Ubuntu 18.04 requires some extra keys for verification
    if [[ "$(cat /etc/os-release)" == *"18.04"* ]]
    then
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install dirmngr gpg-agent
        travis_retry sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80
        travis_retry sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
    elif [[ "$(cat /etc/os-release)" == *"20.04"* ]]
    then
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install dirmngr gpg-agent gnupg2
        travis_retry sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80
        travis_retry sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
    fi

    # Set the correct CUDA downloads
    if [ "${ALPAKA_CI_CUDA_VERSION}" == "9.2" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1604
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_9.2.148-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "10.0" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1804-10-0-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"-10.0.130-410.48_1.0-1_amd64
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "10.1" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1804-10-1-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"-10.1.243-418.87.00_1.0-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "10.2" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1804-10-2-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"-10.2.89-440.33.01_1.0-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.0" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu2004-11-0-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.0.3-450.51.06-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.1" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu2004-11-1-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.1.1-455.32.00-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.2" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu2004-11-2-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.2.2-460.32.03-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.3" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu2004-11-3-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.3.1-465.19.01-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.4" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu2004-11-4-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.4.4-470.82.01-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.5" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu2004-11-5-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.5.2-495.29.05-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.6" ]
    then
        ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu2004-11-6-local
        ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.6.1-510.47.03-1_amd64.deb
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    else
        echo CUDA versions other than 9.2, 10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5 and 11.6 are not currently supported on linux!
    fi
    if [ -z "$(ls -A ${ALPAKA_CI_CUDA_DIR})" ]
    then
        mkdir -p "${ALPAKA_CI_CUDA_DIR}"
        travis_retry wget --no-verbose -O "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}" "${ALPAKA_CUDA_PKG_FILE_PATH}"
    fi
    sudo dpkg --install "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}"

    travis_retry sudo apt-get -y --quiet update

    # Install CUDA
    # Currently we do not install CUDA fully: sudo apt-get --quiet -y install cuda
    # We only install the minimal packages. Because of our manual partial installation we have to create a symlink at /usr/local/cuda
    if (( "${ALPAKA_CUDA_VER_MAJOR}" >= 11 ))
    then
      sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install cuda-compiler-"${ALPAKA_CI_CUDA_VERSION}" cuda-cudart-"${ALPAKA_CI_CUDA_VERSION}" cuda-cudart-dev-"${ALPAKA_CI_CUDA_VERSION}" libcurand-"${ALPAKA_CI_CUDA_VERSION}" libcurand-dev-"${ALPAKA_CI_CUDA_VERSION}"
    else
      sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install cuda-core-"${ALPAKA_CI_CUDA_VERSION}" cuda-cudart-"${ALPAKA_CI_CUDA_VERSION}" cuda-cudart-dev-"${ALPAKA_CI_CUDA_VERSION}" cuda-curand-"${ALPAKA_CI_CUDA_VERSION}" cuda-curand-dev-"${ALPAKA_CI_CUDA_VERSION}"
    fi
    sudo ln -s /usr/local/cuda-"${ALPAKA_CI_CUDA_VERSION}" /usr/local/cuda
    export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

    # Install driver for runtime execution.
    if [ -n "${GITLAB_CI+x}" ]
    then
        latestDriverPackage=$(sudo apt search nvidia-compute-utils 2>/dev/null| grep ^nvidia | sort | tail -n 1 | cut -d"/" -f1)
        sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install $latestDriverPackage
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
    fi

    if [ "${CMAKE_CUDA_COMPILER}" == "clang++" ]
    then
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install g++-multilib
    fi

    # clean up
    sudo rm -rf "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}"
    sudo dpkg --purge "${ALPAKA_CUDA_PKG_DEB_NAME}"
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    if [ "${ALPAKA_CI_CUDA_VERSION}" == "10.0" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_10.0.130_411.31_win10
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "10.1" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_10.1.243_426.00_win10.exe
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "10.2" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_10.2.89_441.22_win10.exe
        ALPAKA_CUDA_PKG_FILE_PATH=http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.0" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_11.0.3_451.82_win10.exe
        ALPAKA_CUDA_PKG_FILE_PATH=http://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.1" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_11.1.1_456.81_win10.exe
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.2" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_11.2.2_461.33_win10.exe
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.3" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_11.3.1_465.89_win10.exe
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.4" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_11.4.4_472.50_windows.exe
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.5" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_11.5.2_496.13_windows.exe
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.6" ]
    then
        ALPAKA_CUDA_PKG_FILE_NAME=cuda_11.6.1_511.65_windows.exe
        ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
    else
        echo CUDA versions other than 10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5 and 11.6 are not currently supported on Windows!
    fi

    curl -L -o cuda_installer.exe ${ALPAKA_CUDA_PKG_FILE_PATH}
    ./cuda_installer.exe -s "nvcc_${ALPAKA_CI_CUDA_VERSION}" "curand_dev_${ALPAKA_CI_CUDA_VERSION}" "cudart_${ALPAKA_CI_CUDA_VERSION}" "visual_studio_integration_${ALPAKA_CI_CUDA_VERSION}"
fi
