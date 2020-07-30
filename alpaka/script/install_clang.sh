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

: "${ALPAKA_CI_CLANG_DIR?'ALPAKA_CI_CLANG_DIR must be specified'}"
: "${ALPAKA_CI_CLANG_VER?'ALPAKA_CI_CLANG_VER must be specified'}"
: "${ALPAKA_CI_CLANG_LIBSTDCPP_VERSION?'ALPAKA_CI_CLANG_LIBSTDCPP_VERSION must be specified'}"
: "${ALPAKA_CI_STDLIB?'ALPAKA_CI_STDLIB must be specified'}"
: "${CXX?'CXX must be specified'}"

ALPAKA_CI_CLANG_VER_SEMANTIC=( ${ALPAKA_CI_CLANG_VER//./ } )
ALPAKA_CI_CLANG_VER_MAJOR="${ALPAKA_CI_CLANG_VER_SEMANTIC[0]}"

if [ -z "$(ls -A "${ALPAKA_CI_CLANG_DIR}")" ]
then

    if (( "${ALPAKA_CI_CLANG_VER_MAJOR}" >= 10 ))
    then
        ALPAKA_CLANG_PKG_FILE_NAME=clang+llvm-${ALPAKA_CI_CLANG_VER}-x86_64-linux-gnu-ubuntu-18.04.tar.xz
        travis_retry wget --no-verbose "https://github.com/llvm/llvm-project/releases/download/llvmorg-${ALPAKA_CI_CLANG_VER}/${ALPAKA_CLANG_PKG_FILE_NAME}"
    else
        ALPAKA_CLANG_PKG_FILE_NAME=clang+llvm-${ALPAKA_CI_CLANG_VER}-x86_64-linux-gnu-ubuntu-16.04.tar.xz
        travis_retry wget --no-verbose "http://llvm.org/releases/${ALPAKA_CI_CLANG_VER}/${ALPAKA_CLANG_PKG_FILE_NAME}"
    fi
    mkdir -p "${ALPAKA_CI_CLANG_DIR}"
    xzcat "${ALPAKA_CLANG_PKG_FILE_NAME}" | tar -xf - --strip 1 -C "${ALPAKA_CI_CLANG_DIR}"
    sudo rm -rf "${ALPAKA_CLANG_PKG_FILE_NAME}"
fi
if [[ "${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}" == *"20.04"* ]]
then
    if (( ("${ALPAKA_CI_CLANG_VER_MAJOR}" >= 9) && ("${ALPAKA_CI_CLANG_VER_MAJOR}" <= 10) ))
    then
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libtinfo5
    fi
fi
"${ALPAKA_CI_CLANG_DIR}/bin/llvm-config" --version
export LLVM_CONFIG="${ALPAKA_CI_CLANG_DIR}/bin/llvm-config"

travis_retry sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
travis_retry sudo apt-get -y --quiet update

travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libstdc++-"${ALPAKA_CI_CLANG_LIBSTDCPP_VERSION}"-dev
if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++-dev
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++abi-dev
fi
sudo update-alternatives --install /usr/bin/clang clang "${ALPAKA_CI_CLANG_DIR}"/bin/clang 50
sudo update-alternatives --install /usr/bin/clang++ clang++ "${ALPAKA_CI_CLANG_DIR}"/bin/clang++ 50
sudo update-alternatives --install /usr/bin/cc cc "${ALPAKA_CI_CLANG_DIR}"/bin/clang 50
sudo update-alternatives --install /usr/bin/c++ c++ "${ALPAKA_CI_CLANG_DIR}"/bin/clang++ 50
# We have to prepend /usr/bin to the path because else the preinstalled clang from usr/bin/local/ is used.
export PATH=${ALPAKA_CI_CLANG_DIR}/bin:${PATH}
if [ -z ${LD_LIBRARY_PATH+x} ]
then
    LD_LIBRARY_PATH=
fi
export LD_LIBRARY_PATH=${ALPAKA_CI_CLANG_DIR}/lib:${LD_LIBRARY_PATH}
if [ -z ${CPPFLAGS+x} ]
then
    CPPFLAGS=
fi
export CPPFLAGS="-I ${ALPAKA_CI_CLANG_DIR}/include/c++/v1 ${CPPFLAGS}"

which "${CXX}"
${CXX} -v
