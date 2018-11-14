#!/bin/bash

#
# Copyright 2017 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

source ./script/travis/travis_retry.sh

#-------------------------------------------------------------------------------
# e: exit as soon as one command returns a non-zero exit code.
set -euo pipefail

: ${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME?"ALPAKA_CI_DOCKER_BASE_IMAGE_NAME must be specified"}
: ${ALPAKA_CI_CUDA_DIR?"ALPAKA_CI_CUDA_DIR must be specified"}
: ${ALPAKA_CUDA_VERSION?"ALPAKA_CUDA_VERSION must be specified"}
: ${ALPAKA_CUDA_COMPILER?"ALPAKA_CUDA_COMPILER must be specified"}

# Ubuntu 18.04 requires some extra keys for verification
if [[ "${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}" == *"18.04"* ]]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install dirmngr gpg-agent
    travis_retry sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80
fi

# Set the correct CUDA downloads
if [ "${ALPAKA_CUDA_VERSION}" == "8.0" ]
then
    ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1404-8-0-local
    ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_8.0.44-1_amd64-deb
    ALPAKA_CUDA_PKG_FILE_PATH=https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
elif [ "${ALPAKA_CUDA_VERSION}" == "9.0" ]
then
    ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1604-9-0-local
    ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_9.0.176-1_amd64-deb
    ALPAKA_CUDA_PKG_FILE_PATH=https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
elif [ "${ALPAKA_CUDA_VERSION}" == "9.1" ]
then
    ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1604-9-1-local
    ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_9.1.85-1_amd64
    ALPAKA_CUDA_PKG_FILE_PATH=https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
elif [ "${ALPAKA_CUDA_VERSION}" == "9.2" ]
then
    ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1604-9-2-local
    ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"_9.2.88-1_amd64
    ALPAKA_CUDA_PKG_FILE_PATH=https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
elif [ "${ALPAKA_CUDA_VERSION}" == "10.0" ]
then
    ALPAKA_CUDA_PKG_DEB_NAME=cuda-repo-ubuntu1804-10-0-local
    ALPAKA_CUDA_PKG_FILE_NAME="${ALPAKA_CUDA_PKG_DEB_NAME}"-10.0.130-410.48_1.0-1_amd64
    ALPAKA_CUDA_PKG_FILE_PATH=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/${ALPAKA_CUDA_PKG_FILE_NAME}
else
    echo CUDA versions other than 8.0, 9.0, 9.1, 9.2 and 10.0 are not currently supported!
fi
if [ -z "$(ls -A "${ALPAKA_CI_CUDA_DIR}")" ]
then
    mkdir -p "${ALPAKA_CI_CUDA_DIR}"
    travis_retry wget --no-verbose -O "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}" "${ALPAKA_CUDA_PKG_FILE_PATH}"
fi
sudo dpkg --install "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}"

travis_retry sudo apt-get -y --quiet update

# Install CUDA
# Currently we do not install CUDA fully: sudo apt-get --quiet -y install cuda
# We only install the minimal packages. Because of our manual partial installation we have to create a symlink at /usr/local/cuda
sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install cuda-core-"${ALPAKA_CUDA_VERSION}" cuda-cudart-"${ALPAKA_CUDA_VERSION}" cuda-cudart-dev-"${ALPAKA_CUDA_VERSION}" cuda-curand-"${ALPAKA_CUDA_VERSION}" cuda-curand-dev-"${ALPAKA_CUDA_VERSION}"
sudo ln -s /usr/local/cuda-"${ALPAKA_CUDA_VERSION}" /usr/local/cuda

if [ "${ALPAKA_CUDA_COMPILER}" == "clang" ]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install g++-multilib
fi

# clean up
sudo rm -rf "${ALPAKA_CI_CUDA_DIR}"/"${ALPAKA_CUDA_PKG_FILE_NAME}"
sudo dpkg --purge "${ALPAKA_CUDA_PKG_DEB_NAME}"
