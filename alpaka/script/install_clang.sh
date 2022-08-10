#!/bin/bash

#
# Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
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

# Install from LLVM repository (if available); otherwise install LLVM from official Ubuntu repositories
ALPAKA_CI_UBUNTU_NAME=`lsb_release -c | awk '{print $2}'`
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -

# bionic = 18.04; focal = 20.04; jammy = 22.04
if { [ "${ALPAKA_CI_UBUNTU_NAME}" == "bionic" ] && [ "${ALPAKA_CI_CLANG_VER}" -ge 7 ]; } || \
   { [ "${ALPAKA_CI_UBUNTU_NAME}" == "focal" ] && [ "${ALPAKA_CI_CLANG_VER}" -ge 9 ]; } || \
   { [ "${ALPAKA_CI_UBUNTU_NAME}" == "jammy" ] && [ "${ALPAKA_CI_CLANG_VER}" -ge 13 ]; }
then
    sudo add-apt-repository "deb http://apt.llvm.org/${ALPAKA_CI_UBUNTU_NAME}/ llvm-toolchain-${ALPAKA_CI_UBUNTU_NAME}-$ALPAKA_CI_CLANG_VER main"
fi

travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install clang-${ALPAKA_CI_CLANG_VER}

if [ -n "${ALPAKA_CI_SANITIZERS}" ]
then
    # llvm-symbolizer is required for meaningful output. This is part of the llvm base package which we don't install by default.
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install llvm-${ALPAKA_CI_CLANG_VER}
fi

if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
then
    travis_retry sudo apt-get -y --quiet update
    if [ "${ALPAKA_CI_CLANG_VER}" -ge 7 ]
    then
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++-${ALPAKA_CI_CLANG_VER}-dev
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++abi-${ALPAKA_CI_CLANG_VER}-dev
        if [ "${ALPAKA_CI_CLANG_VER}" -ge 12 ]
        then
            # Starting from LLVM 12 libunwind is required when using libc++. For some reason this isn't installed by default
            travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libunwind-${ALPAKA_CI_CLANG_VER}-dev
        fi
    else
        # Ubuntu started numbering libc++ with version 7. If we got to this point, we need to install the 
        # default libc++ and hope for the best
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++-dev
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++abi-dev
    fi
fi

if [ "${alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE}" = "ON" ] || [ "${alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE}" = "ON" ] || [ "${alpaka_ACC_ANY_BT_OMP5_ENABLE}" = "ON" ]
then
    if [[ "${ALPAKA_CI_CLANG_VER}" =~ ^[0-9]+$ ]] && [ "${ALPAKA_CI_CLANG_VER}" -ge 7 ]
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
