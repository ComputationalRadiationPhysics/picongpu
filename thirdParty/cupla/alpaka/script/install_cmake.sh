#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_CMAKE_DIR?'ALPAKA_CI_CMAKE_DIR must be specified'}"
: "${ALPAKA_CI_CMAKE_VER?'ALPAKA_CI_CMAKE_VER must be specified'}"

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    # Download the selected version.
    if [ -z "$(ls -A ${ALPAKA_CI_CMAKE_DIR})" ]
    then
        ALPAKA_CI_CMAKE_VER_SEMANTIC=( ${ALPAKA_CI_CMAKE_VER//./ } )
        ALPAKA_CI_CMAKE_VER_MAJOR="${ALPAKA_CI_CMAKE_VER_SEMANTIC[0]}"
        ALPAKA_CI_CMAKE_VER_MINOR="${ALPAKA_CI_CMAKE_VER_SEMANTIC[1]}"

        ALPAKA_CMAKE_PKG_FILE_NAME_BASE=cmake-${ALPAKA_CI_CMAKE_VER}-Linux-x86_64
        ALPAKA_CMAKE_PKG_FILE_NAME=${ALPAKA_CMAKE_PKG_FILE_NAME_BASE}.tar.gz
        travis_retry wget --no-verbose https://cmake.org/files/v"${ALPAKA_CI_CMAKE_VER_MAJOR}"."${ALPAKA_CI_CMAKE_VER_MINOR}"/"${ALPAKA_CMAKE_PKG_FILE_NAME}"
        mkdir -p "${ALPAKA_CI_CMAKE_DIR}"
        tar -xzf "${ALPAKA_CMAKE_PKG_FILE_NAME}" -C "${ALPAKA_CI_CMAKE_DIR}"
        sudo cp -fR "${ALPAKA_CI_CMAKE_DIR}"/"${ALPAKA_CMAKE_PKG_FILE_NAME_BASE}"/* "${ALPAKA_CI_CMAKE_DIR}"
        sudo rm -rf "${ALPAKA_CMAKE_PKG_FILE_NAME}" "${ALPAKA_CI_CMAKE_DIR}"/"${ALPAKA_CMAKE_PKG_FILE_NAME_BASE}"
    fi
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    choco uninstall cmake.install
    choco install cmake.install --no-progress --version ${ALPAKA_CI_CMAKE_VER}
fi
