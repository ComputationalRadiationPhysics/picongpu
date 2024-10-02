#!/bin/bash

#
# Copyright 2023 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_CUDA_VERSION?'ALPAKA_CI_CUDA_VERSION must be specified'}"

ALPAKA_CUDA_VER_SEMANTIC=( ${ALPAKA_CI_CUDA_VERSION//./ } )
ALPAKA_CUDA_VER_MAJOR="${ALPAKA_CUDA_VER_SEMANTIC[0]}"
echo ALPAKA_CUDA_VER_MAJOR: "${ALPAKA_CUDA_VER_MAJOR}"


if agc-manager -e cuda@${ALPAKA_CI_CUDA_VERSION}
then
    ALPAKA_CI_CUDA_PATH=$(agc-manager -b cuda@${ALPAKA_CI_CUDA_VERSION})
    export PATH=${ALPAKA_CI_CUDA_PATH}/bin:${PATH}
    export LD_LIBRARY_PATH=${ALPAKA_CI_CUDA_PATH}/lib64:${LD_LIBRARY_PATH}
else
    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
    then
        : "${ALPAKA_CI_CUDA_DIR?'ALPAKA_CI_CUDA_DIR must be specified'}"
        : "${CMAKE_CUDA_COMPILER?'CMAKE_CUDA_COMPILER must be specified'}"

        if [[ "$(cat /etc/os-release)" == *"20.04"* ]]
        then
            ALPAKA_CUDA_DISTRO=ubuntu2004
        elif [[ "$(cat /etc/os-release)" == *"22.04"* ]]
        then
            ALPAKA_CUDA_DISTRO=ubuntu2204
        fi

        # Set the correct CUDA downloads
        if [ "${ALPAKA_CI_CUDA_VERSION}" == "11.0" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-0-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.0.3-450.51.06-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=true
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.1" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-1-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.1.1-455.32.00-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=true
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.2" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-2-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.2.2-460.32.03-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=true
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.3" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-3-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.3.1-465.19.01-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=true
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.4" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-4-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.4.4-470.82.01-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=true
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.5" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-5-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.5.2-495.29.05-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=true
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.6" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-6-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.6.2-510.47.03-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=true
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.7" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-7-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.7.1-515.65.01-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=false
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "11.8" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-11-8-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_11.8.0-520.61.05-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=false
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "12.0" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-12-0-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_12.0.0-525.60.13-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=false
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "12.1" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-12-1-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_12.1.1-530.30.02-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=false
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "12.2" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-12-2-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_12.2.1-535.86.10-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=false
        elif [ "${ALPAKA_CI_CUDA_VERSION}" == "12.3" ]
        then
            ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-"${ALPAKA_CUDA_DISTRO}"-12-3-local
            ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_12.3.2-545.23.08-1_amd64.deb
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
            ALPAKA_CUDA_OLD_KEYS=false
        else
            echo CUDA versions other than 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 12.0, 12.1, 12.2 and 12.3 are not currently supported on linux!
        fi

        # First install the local repository.
        if [ -z "$(ls -A ${ALPAKA_CI_CUDA_DIR})" ]
        then
            mkdir -p "${ALPAKA_CI_CUDA_DIR}"
            travis_retry wget --no-verbose -O "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}" "${ALPAKA_CUDA_PKG_FILE_PATH}"
        fi
        sudo dpkg --install "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}"

        # Ubuntu requires some extra keys for verification. However, the way of acquiring the keys differs between older and newer CUDA releases.
        if [ "${ALPAKA_CUDA_OLD_KEYS}" = true ]
        then
            # For all versions < 11.7
            travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install dirmngr gpg-agent gnupg2
            travis_retry sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80
            travis_retry sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
        else
            # Since 11.7 the keys are shipped as part of the local repository.
            sudo cp /var/${ALPAKA_CUDA_PKG_DEB_NAME}/cuda-*-keyring.gpg /usr/share/keyrings
        fi

        travis_retry sudo apt-get -y --quiet update

        # Install CUDA
        # Currently we do not install CUDA fully: sudo apt-get --quiet -y install cuda
        # We only install the minimal packages. Because of our manual partial installation we have to create a symlink at /usr/local/cuda
        sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install cuda-compiler-"${ALPAKA_CI_CUDA_VERSION}" cuda-cudart-"${ALPAKA_CI_CUDA_VERSION}" cuda-cudart-dev-"${ALPAKA_CI_CUDA_VERSION}" libcurand-"${ALPAKA_CI_CUDA_VERSION}" libcurand-dev-"${ALPAKA_CI_CUDA_VERSION}"

        sudo ln -s /usr/local/cuda-"${ALPAKA_CI_CUDA_VERSION}" /usr/local/cuda
        export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
        export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

        if [ "${CMAKE_CUDA_COMPILER}" == "clang++" ]
        then
            travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install g++-multilib
        fi

        # clean up
        sudo rm -rf "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}"
        sudo dpkg --purge "${ALPAKA_CUDA_PKG_DEB_NAME}"
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        if [ "${ALPAKA_CI_CUDA_VERSION}" == "12.1" ]
        then
            ALPAKA_CUDA_PKG_FILE_NAME=cuda_12.1.1_531.14_windows.exe
            ALPAKA_CUDA_PKG_FILE_PATH=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
        else
            echo CUDA versions other than 12.1 are not currently supported on Windows!
        fi

        curl -L -o cuda_installer.exe ${ALPAKA_CUDA_PKG_FILE_PATH}
        # The 'thrust' package contains the CUDA C++ Core Compute Libraries (CCCL) which are required for cuRAND. The installer doesn't do dependency management so we have to install them manually.
        ./cuda_installer.exe -s "nvcc_${ALPAKA_CI_CUDA_VERSION}" "curand_dev_${ALPAKA_CI_CUDA_VERSION}" "cudart_${ALPAKA_CI_CUDA_VERSION}" "thrust_${ALPAKA_CI_CUDA_VERSION}" "visual_studio_integration_${ALPAKA_CI_CUDA_VERSION}"
    fi
fi
