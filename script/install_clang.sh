#!/bin/bash

#
# Copyright 2023 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_CLANG_VER?'ALPAKA_CI_CLANG_VER must be specified'}"
: "${ALPAKA_CI_STDLIB?'ALPAKA_CI_STDLIB must be specified'}"
: "${CXX?'CXX must be specified'}"

#TODO(SimeonEhrig): remove this statement, if ppa's are fixed in alpaka-group-container
if [[ -f "/etc/apt/sources.list.d/llvm.list" ]];
then
    sudo rm /etc/apt/sources.list.d/llvm.list
fi

if ! agc-manager -e clang@${ALPAKA_CI_CLANG_VER}
then
    # Install from LLVM repository (if available); otherwise install LLVM from official Ubuntu repositories
    ALPAKA_CI_UBUNTU_NAME=`lsb_release -c | awk '{print $2}'`
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -

    # focal = 20.04; jammy = 22.04
    if { [ "${ALPAKA_CI_UBUNTU_NAME}" == "focal" ] && [ "${ALPAKA_CI_CLANG_VER}" -ge 13 ]; } || \
    { [ "${ALPAKA_CI_UBUNTU_NAME}" == "jammy" ] && [ "${ALPAKA_CI_CLANG_VER}" -ge 15 ]; }
    then
        sudo add-apt-repository "deb http://apt.llvm.org/${ALPAKA_CI_UBUNTU_NAME}/ llvm-toolchain-${ALPAKA_CI_UBUNTU_NAME}-$ALPAKA_CI_CLANG_VER main"
    fi

    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install clang-${ALPAKA_CI_CLANG_VER}

    if [ -n "${ALPAKA_CI_SANITIZERS}" ]
    then
        # llvm-symbolizer is required for meaningful output. This is part of the llvm base package which we don't install by default.
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install llvm-${ALPAKA_CI_CLANG_VER}

        # The sanitizer libraries are part of libclang-rt-${ALPAKA_CI_CLANG_VER}-dev
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libclang-rt-${ALPAKA_CI_CLANG_VER}-dev
    fi

    if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
    then
        travis_retry sudo apt-get -y --quiet update
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++-${ALPAKA_CI_CLANG_VER}-dev
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libc++abi-${ALPAKA_CI_CLANG_VER}-dev
        if [ "${ALPAKA_CI_CLANG_VER}" -ge 12 ]
        then
            # Starting from LLVM 12 libunwind is required when using libc++. For some reason this isn't installed by default
            travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libunwind-${ALPAKA_CI_CLANG_VER}-dev
        fi
    fi

    if [ "${alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE}" = "ON" ] || [ "${alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE}" = "ON" ]
    then
        LIBOMP_PACKAGE=libomp-${ALPAKA_CI_CLANG_VER}-dev
        travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install "${LIBOMP_PACKAGE}"
    fi

    sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-"${ALPAKA_CI_CLANG_VER}" 50
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-"${ALPAKA_CI_CLANG_VER}" 50
    sudo update-alternatives --install /usr/bin/cc cc /usr/bin/clang-"${ALPAKA_CI_CLANG_VER}" 50
    sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-"${ALPAKA_CI_CLANG_VER}" 50
fi

which "${CXX}"
${CXX} --version
