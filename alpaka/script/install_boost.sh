#!/bin/bash

#
# Copyright 2022 Benjamin Worpitz, René Widera, Axel Huebl, Bernhard Manfred Gruber, Andrea Bocci
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis_retry.sh
source ./script/set.sh

: "${BOOST_ROOT?'BOOST_ROOT must be specified'}"
: "${ALPAKA_CI_BOOST_LIB_DIR?'ALPAKA_CI_BOOST_LIB_DIR must be specified'}"
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    : "${ALPAKA_CI_STDLIB?'ALPAKA_CI_STDLIB must be specified'}"
fi
: "${CMAKE_BUILD_TYPE?'CMAKE_BUILD_TYPE must be specified'}"
: "${CXX?'CXX must be specified'}"
: "${CC?'CC must be specified'}"
: "${ALPAKA_CI_INSTALL_FIBERS?'ALPAKA_CI_INSTALL_FIBERS must be specified'}"
: "${ALPAKA_CI_INSTALL_ATOMIC?'ALPAKA_CI_INSTALL_ATOMIC must be specified'}"
if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    : "${ALPAKA_CI_CL_VER?'ALPAKA_CI_CL_VER must be specified'}"
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    # To speed up installing boost we use the tar ball, downloading the source via a shallow git clone took ~4 minutes, with wget ~20 seconds.
    BOOST_BASE_NAME=$(echo -n ${ALPAKA_CI_BOOST_BRANCH} | tr "." "_" | tr "-" "_")
    BOOST_BASE_VERSION=$(echo -n ${ALPAKA_CI_BOOST_BRANCH} | sed 's/boost-//')
    BOOST_TAR_FILE_NAME="${BOOST_BASE_NAME}.tar.gz"
    BOOST_DOWNLOAD_LINK="https://boostorg.jfrog.io/artifactory/main/release/${BOOST_BASE_VERSION}/source/${BOOST_TAR_FILE_NAME}"
    rm -rf ${BOOST_ROOT}
    travis_retry wget -q ${BOOST_DOWNLOAD_LINK}
    tar -xf "$BOOST_TAR_FILE_NAME"
    rm "$BOOST_TAR_FILE_NAME"

    BOOST_UNCOMPRESSED_FOLDER_NAME=${BOOST_BASE_NAME}
    mv "$BOOST_UNCOMPRESSED_FOLDER_NAME" "${BOOST_ROOT}"
else
    travis_retry rm -rf ${BOOST_ROOT} && git clone -b "${ALPAKA_CI_BOOST_BRANCH}" --quiet --recursive --single-branch --depth 1 https://github.com/boostorg/boost.git "${BOOST_ROOT}"
fi

# Set the toolset based on the compiler
if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    if [ "$ALPAKA_CI_CL_VER" = "2017" ]
    then
        TOOLSET="msvc-14.1"
    elif [ "$ALPAKA_CI_CL_VER" = "2019" ]
    then
        TOOLSET="msvc-14.2"
    elif [ "$ALPAKA_CI_CL_VER" = "2022" ]
    then
        TOOLSET="msvc-14.3"
    fi
    # Add new versions as needed
elif [ "${CXX}" == "icpc" ]
then
    TOOLSET="intel-linux"
else
    TOOLSET="${CC}"
fi

# Bootstrap boost.
if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    (cd "${BOOST_ROOT}"; ./bootstrap.bat --with-toolset="${TOOLSET}")
elif [ "${TOOLSET}" == "intel-linux" ]
then
    (cd "${BOOST_ROOT}"; sudo $SHELL -c "source /opt/intel/oneapi/setvars.sh; ./bootstrap.sh --with-toolset='${TOOLSET}'" || cat bootstrap.log)
else
    (cd "${BOOST_ROOT}"; sudo ./bootstrap.sh --with-toolset="${TOOLSET}" || cat bootstrap.log)
fi

# Create file links.
if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    (cd "${BOOST_ROOT}"; ./b2 headers)
elif [ "${TOOLSET}" == "intel-linux" ]
then
    (cd "${BOOST_ROOT}"; sudo $SHELL -c "source /opt/intel/oneapi/setvars.sh; ./b2 headers")
else
    (cd "${BOOST_ROOT}"; sudo ./b2 headers)
fi

# Only build boost if we need some of the non-header-only libraries
if [ "${ALPAKA_CI_INSTALL_FIBERS}" == "ON" ] || [ "${ALPAKA_CI_INSTALL_ATOMIC}" == "ON" ]
then
    # Prepare the library destination directory.
    mkdir -p "${ALPAKA_CI_BOOST_LIB_DIR}"

    # Create the boost build command.
    ALPAKA_BOOST_B2=""
    ALPAKA_BOOST_B2_CFLAGS=""
    ALPAKA_BOOST_B2_CXXFLAGS=""

    ALPAKA_BOOST_B2+="./b2 -j1"

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
    then
        ALPAKA_BOOST_B2_CFLAGS+="-fPIC"
        ALPAKA_BOOST_B2_CXXFLAGS+="-fPIC"
    fi

    if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        ALPAKA_BOOST_B2+=" --layout=versioned"
    else
        ALPAKA_BOOST_B2+=" --layout=tagged"
    fi

    if [ "${TOOLSET}" ]
    then
        ALPAKA_BOOST_B2+=" --toolset=${TOOLSET}"
    fi

    ALPAKA_BOOST_B2+=" architecture=x86 address-model=64 link=static threading=multi runtime-link=shared"

    if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        ALPAKA_BOOST_B2+=" define=_CRT_NONSTDC_NO_DEPRECATE define=_CRT_SECURE_NO_DEPRECATE define=_SCL_SECURE_NO_DEPRECAT define=BOOST_USE_WINFIBERS define=_ENABLE_EXTENDED_ALIGNED_STORAGE"
    fi

    if [ "${CMAKE_BUILD_TYPE}" == "Debug" ]
    then
      ALPAKA_BOOST_B2+=" variant=debug"
    else
      ALPAKA_BOOST_B2+=" variant=release"
    fi

    # Clang is not supported by the FindBoost script.
    # boost (especially old versions) produces too much warnings when using clang (newer versions) so that the 4 MiB log is too short.
    if [[ "${CXX}" == "clang++"* ]]
    then
        ALPAKA_BOOST_B2_CXXFLAGS+=" -Wunused-private-field -Wno-unused-local-typedef -Wno-c99-extensions -Wno-variadic-macros"
    fi
    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
    then
        ALPAKA_BOOST_B2_CXXFLAGS+=" -std=c++17"
    fi
    ALPAKA_BOOST_B2+=" --with-fiber --with-context --with-thread --with-atomic --with-system --with-chrono --with-date_time"

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
    then
        if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
        then
            ALPAKA_BOOST_B2_CXXFLAGS+=" -stdlib=libc++"
        fi
    fi
    if [ "${ALPAKA_BOOST_B2_CFLAGS}" != "" ]
    then
        ALPAKA_BOOST_B2+=' cflags="'
        ALPAKA_BOOST_B2+="${ALPAKA_BOOST_B2_CFLAGS}"
        ALPAKA_BOOST_B2+='"'
    fi
    if [ "${ALPAKA_BOOST_B2_CXXFLAGS}" != "" ]
    then
        ALPAKA_BOOST_B2+=' cxxflags="'
        ALPAKA_BOOST_B2+="${ALPAKA_BOOST_B2_CXXFLAGS}"
        ALPAKA_BOOST_B2+='"'
    fi

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
    then
        if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
        then
            ALPAKA_BOOST_B2+=' linkflags="-stdlib=libc++"'
        fi
    fi

    # The extra quotes are necessary to protect \ characters in Windows paths in the eval
    ALPAKA_BOOST_B2+=" --stagedir=\"${ALPAKA_CI_BOOST_LIB_DIR}\" stage"

    # Build boost.
    if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        (cd "${BOOST_ROOT}"; eval "${ALPAKA_BOOST_B2}")
    elif [ "${TOOLSET}" == "intel-linux" ]
    then
        (cd "${BOOST_ROOT}"; sudo $SHELL -c "source /opt/intel/oneapi/setvars.sh; ${ALPAKA_BOOST_B2}")
    else
        (cd "${BOOST_ROOT}"; eval "sudo ${ALPAKA_BOOST_B2}")
    fi

    # Clean the intermediate build files.
    if [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        rm -rf bin.v2
    else
        sudo rm -rf bin.v2
    fi
fi
