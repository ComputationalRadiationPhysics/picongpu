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
set -e

: ${ALPAKA_CI_CMAKE_DIR?"ALPAKA_CI_CMAKE_DIR must be specified"}
: ${ALPAKA_CI_CMAKE_VER?"ALPAKA_CI_CMAKE_VER must be specified"}

#-------------------------------------------------------------------------------
# Remove the old CMake version.
if [ "${TRAVIS}" == "true" ] ;then sudo apt-get -y --quiet remove cmake ;fi

# Download the selected version.
if [ -z "$(ls -A "${ALPAKA_CI_CMAKE_DIR}")" ]
then
    ALPAKA_CMAKE_PKG_FILE_NAME_BASE=cmake-${ALPAKA_CI_CMAKE_VER}-Linux-x86_64
    ALPAKA_CMAKE_PKG_FILE_NAME=${ALPAKA_CMAKE_PKG_FILE_NAME_BASE}.tar.gz
    travis_retry wget --no-verbose https://cmake.org/files/v"${ALPAKA_CI_CMAKE_VER_MAJOR}"."${ALPAKA_CI_CMAKE_VER_MINOR}"/"${ALPAKA_CMAKE_PKG_FILE_NAME}"
    mkdir -p "${ALPAKA_CI_CMAKE_DIR}"
    tar -xzf "${ALPAKA_CMAKE_PKG_FILE_NAME}" -C "${ALPAKA_CI_CMAKE_DIR}"
    sudo cp -fR "${ALPAKA_CI_CMAKE_DIR}"/"${ALPAKA_CMAKE_PKG_FILE_NAME_BASE}"/* "${ALPAKA_CI_CMAKE_DIR}"
    sudo rm -rf "${ALPAKA_CMAKE_PKG_FILE_NAME}" "${ALPAKA_CI_CMAKE_DIR}"/"${ALPAKA_CMAKE_PKG_FILE_NAME_BASE}"
fi
