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

: "${ALPAKA_CI_CLANG_VER?'ALPAKA_CI_CLANG_VER must be specified'}"
: "${ALPAKA_CI_STDLIB?'ALPAKA_CI_STDLIB must be specified'}"
: "${CXX?'CXX must be specified'}"

# add clang-11 repository for ubuntu 18.04
if [[ "$(cat /etc/os-release)" == *"18.04"* && "${ALPAKA_CI_CLANG_VER}" -eq 11 ]]
then
    travis_retry sudo DEBIAN_FRONTEND=noninteractive apt-get -y --quiet --allow-unauthenticated --no-install-recommends install tzdata
    travis_retry apt-get -y --quiet install wget gnupg2
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main" | sudo tee /etc/apt/sources.list.d/clang11.list
    echo "deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main" | sudo tee -a  /etc/apt/sources.list.d/clang11.list
    travis_retry apt-get -y --quiet update
fi

# add clang-13 repository for ubuntu 20.04
if [[ "$(cat /etc/os-release)" == *"20.04"* && "${ALPAKA_CI_CLANG_VER}" -eq 13 ]]
then
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    sudo add-apt-repository 'deb http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main'
fi

travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install clang-${ALPAKA_CI_CLANG_VER}

if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
then
    travis_retry sudo apt-get -y --quiet update
    if [ "${ALPAKA_CI_CLANG_VER}" -gt 6 ]
    then
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++-${ALPAKA_CI_CLANG_VER}-dev
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++abi-${ALPAKA_CI_CLANG_VER}-dev
    else
        # Ubuntu started numbering libc++ with version 7. If we got to this point, we need to install the 
        # default libc++ and hope for the best
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++-dev
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++abi-dev
    fi
fi

if [ "${alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE}" = "ON" ] || [ "${alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE}" = "ON" ] || [ "${alpaka_ACC_ANY_BT_OMP5_ENABLE}" = "ON" ]
then
    if [[ "${ALPAKA_CI_CLANG_VER}" =~ ^[0-9]+$ ]] && [ "${ALPAKA_CI_CLANG_VER}" -ge 8 ]
    then
        LIBOMP_PACKAGE=libomp-${ALPAKA_CI_CLANG_VER}-dev
    else
        LIBOMP_PACKAGE=libomp-dev
    fi
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install "${LIBOMP_PACKAGE}"
    if [ "${alpaka_ACC_ANY_BT_OMP5_ENABLE}" = "ON" ]
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
