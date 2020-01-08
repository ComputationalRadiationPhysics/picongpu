#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis/travis_retry.sh

source ./script/travis/set.sh

: ${ALPAKA_CI_ANALYSIS?"ALPAKA_CI_ANALYSIS must be specified"}
: ${ALPAKA_CI_INSTALL_CUDA?"ALPAKA_CI_INSTALL_CUDA must be specified"}
: ${ALPAKA_CI_INSTALL_HIP?"ALPAKA_CI_INSTALL_HIP must be specified"}
: ${ALPAKA_CI_INSTALL_TBB?"ALPAKA_CI_INSTALL_TBB must be specified"}

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    travis_retry apt-get -y --quiet update
    travis_retry apt-get -y install sudo

    # software-properties-common: 'add-apt-repository' and certificates for wget https download
    # binutils: ld
    # xz-utils: xzcat
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install software-properties-common wget git make binutils xz-utils
fi

if [ "$TRAVIS_OS_NAME" = "linux" ] || [ "$TRAVIS_OS_NAME" = "windows" ]
then
    ./script/travis/install_cmake.sh
fi

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;then ./script/travis/install_analysis.sh ;fi
fi

# Install CUDA before installing gcc as it installs gcc-4.8 and overwrites our selected compiler
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ] ;then ./script/travis/install_cuda.sh ;fi

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    if [ "${CXX}" == "g++" ] ;then ./script/travis/install_gcc.sh ;fi
    if [ "${CXX}" == "clang++" ] ;then source ./script/travis/install_clang.sh ;fi
    if [ "${ALPAKA_CI_INSTALL_HIP}" == "ON" ] ;then ./script/travis/install_hip.sh ;fi
fi

if [ "${ALPAKA_CI_INSTALL_TBB}" = "ON" ]
then
    ./script/travis/install_tbb.sh
fi

./script/travis/install_boost.sh

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    # Minimize docker image size
    sudo apt-get --quiet --purge autoremove
    sudo apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
fi
