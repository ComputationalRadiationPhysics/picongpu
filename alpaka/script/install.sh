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
        travis_retry sudo DEBIAN_FRONTEND=noninteractive apt-get -y --quiet --allow-unauthenticated --no-install-recommends install tzdata
    fi

    # software-properties-common: 'add-apt-repository' and certificates for wget https download
    # binutils: ld
    # xz-utils: xzcat
    travis_retry sudo DEBIAN_FRONTEND=noninteractive apt-get -y --quiet --allow-unauthenticated --no-install-recommends install software-properties-common wget git make binutils xz-utils gnupg2
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    source ./script/install_cmake.sh
fi

if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;then source ./script/install_analysis.sh ;fi

# Install CUDA before installing gcc as it installs gcc-4.8 and overwrites our selected compiler
if [ "${ALPAKA_CI_INSTALL_CUDA}" == "ON" ] ;then source ./script/install_cuda.sh ;fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    if [[ "${CXX}" == "g++"* ]] ;then source ./script/install_gcc.sh ;fi
    # do not install clang if we use HIP, HIP/ROCm is shipping an own clang version
    if [[ "${CXX}" == "clang++" ]] && [ "${ALPAKA_CI_INSTALL_HIP}" != "ON" ] ;then source ./script/install_clang.sh ;fi
    if [[ "${CXX}" == "icpc"* ]] ;then source ./script/install_icpc.sh ;fi
elif [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    echo "### list all applications ###"
    ls "/Applications/"
    echo "### end list all applications ###"
    sudo xcode-select -s "/Applications/Xcode_${ALPAKA_CI_XCODE_VER}.app/Contents/Developer"
fi

if [ "${ALPAKA_CI_INSTALL_TBB}" = "ON" ]
then
    source ./script/install_tbb.sh
fi

if [ "${ALPAKA_CI_INSTALL_OMP}" = "ON" ]
then
    source ./script/install_omp.sh
fi

# HIP
if [ "${ALPAKA_CI_INSTALL_HIP}" = "ON" ]
then
    source ./script/install_hip.sh
fi

source ./script/install_boost.sh

