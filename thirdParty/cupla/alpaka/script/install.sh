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

: ${ALPAKA_CI_ANALYSIS?"ALPAKA_CI_ANALYSIS must be specified"}
: ${ALPAKA_CI_INSTALL_CUDA?"ALPAKA_CI_INSTALL_CUDA must be specified"}
: ${ALPAKA_CI_INSTALL_HIP?"ALPAKA_CI_INSTALL_HIP must be specified"}
: ${ALPAKA_CI_INSTALL_TBB?"ALPAKA_CI_INSTALL_TBB must be specified"}

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    travis_retry apt-get -y --quiet update
    travis_retry apt-get -y install sudo

    # tzdata is installed by software-properties-common but it requires some special handling
    if [[ "$(cat /etc/os-release)" == *"20.04"* ]]
    then
        export DEBIAN_FRONTEND=noninteractive
        travis_retry sudo apt-get --quiet --allow-unauthenticated --no-install-recommends install tzdata
    fi

    # software-properties-common: 'add-apt-repository' and certificates for wget https download
    # binutils: ld
    # xz-utils: xzcat
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install software-properties-common wget git make binutils xz-utils
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    ./script/install_cmake.sh
fi

if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;then ./script/install_analysis.sh ;fi

# Install CUDA before installing gcc as it installs gcc-4.8 and overwrites our selected compiler
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ] ;then ./script/install_cuda.sh ;fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    if [ "${CXX}" == "g++" ] ;then ./script/install_gcc.sh ;fi
    if [ "${CXX}" == "clang++" ] ;then source ./script/install_clang.sh ;fi
elif [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    echo "### list all applications ###"
    ls "/Applications/"
    echo "### end list all applications ###"
    sudo xcode-select -s "/Applications/Xcode_${ALPAKA_CI_XCODE_VER}.app/Contents/Developer"
fi

if [ "${ALPAKA_CI_INSTALL_TBB}" = "ON" ]
then
    ./script/install_tbb.sh
fi

./script/install_boost.sh

