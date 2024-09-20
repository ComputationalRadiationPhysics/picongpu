#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: install_cmake>"

: "${ALPAKA_CI_CMAKE_DIR?'ALPAKA_CI_CMAKE_DIR must be specified'}"
: "${ALPAKA_CI_CMAKE_VER?'ALPAKA_CI_CMAKE_VER must be specified'}"

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    if agc-manager -e cmake@${ALPAKA_CI_CMAKE_VER} ; then
        echo_green "<USE: preinstalled CMake ${ALPAKA_CI_CMAKE_VER}>"
        export ALPAKA_CI_CMAKE_DIR=$(agc-manager -b cmake@${ALPAKA_CI_CMAKE_VER})
    else
        echo_yellow "<INSTALL: CMake ${ALPAKA_CI_CMAKE_VER}>"
        if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
        then
            # Download the selected version.
            if [ -z "$(ls -A ${ALPAKA_CI_CMAKE_DIR})" ]
            then
                ALPAKA_CI_CMAKE_VER_SEMANTIC=( ${ALPAKA_CI_CMAKE_VER//./ } )
                ALPAKA_CI_CMAKE_VER_MAJOR="${ALPAKA_CI_CMAKE_VER_SEMANTIC[0]}"
                ALPAKA_CI_CMAKE_VER_MINOR="${ALPAKA_CI_CMAKE_VER_SEMANTIC[1]}"

                ALPAKA_CMAKE_PKG_FILE_NAME_BASE=
                if (( ( ( "${ALPAKA_CI_CMAKE_VER_MAJOR}" == 3 ) && ( "${ALPAKA_CI_CMAKE_VER_MINOR}" > 19 ) ) || ( "${ALPAKA_CI_CMAKE_VER_MAJOR}" > 3 ) ))
                then
                    ALPAKA_CMAKE_PKG_FILE_NAME_BASE=cmake-${ALPAKA_CI_CMAKE_VER}-linux-x86_64
                else
                    ALPAKA_CMAKE_PKG_FILE_NAME_BASE=cmake-${ALPAKA_CI_CMAKE_VER}-Linux-x86_64
                fi
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
    fi
fi

echo "ALPAKA_CI_CMAKE_DIR: ${ALPAKA_CI_CMAKE_DIR}"
