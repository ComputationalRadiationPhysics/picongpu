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

: "${ALPAKA_CI_CLANG_VER?'ALPAKA_CI_CLANG_VER must be specified'}"
: "${ALPAKA_CI_STDLIB?'ALPAKA_CI_STDLIB must be specified'}"
: "${CXX?'CXX must be specified'}"

travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install clang-${ALPAKA_CI_CLANG_VER}

if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
then
    travis_retry sudo apt-get -y --quiet update
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++-dev
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++abi-dev
fi

if [ "${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE}" = "ON" ] || [ "${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE}" = "ON" ] || [ "${ALPAKA_ACC_ANY_BT_OMP5_ENABLE}" = "ON" ]
then
    if [[ "${ALPAKA_CI_CLANG_VER}" =~ ^[0-9]+$ ]] && [ "${ALPAKA_CI_CLANG_VER}" -ge 8 ]
    then
        LIBOMP_PACKAGE=libomp-${ALPAKA_CI_CLANG_VER}-dev
    else
        LIBOMP_PACKAGE=libomp-dev
    fi
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install "${LIBOMP_PACKAGE}"
    if [ "${ALPAKA_ACC_ANY_BT_OMP5_ENABLE}" = "ON" ]
    then
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install \
            clang-tools-${ALPAKA_CI_CLANG_VER} llvm-${ALPAKA_CI_CLANG_VER}
    fi
fi

sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-"${ALPAKA_CI_CLANG_VER}" 50
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-"${ALPAKA_CI_CLANG_VER}" 50
sudo update-alternatives --install /usr/bin/cc cc /usr/bin/clang-"${ALPAKA_CI_CLANG_VER}" 50
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-"${ALPAKA_CI_CLANG_VER}" 50

which "${CXX}"
${CXX} -v
