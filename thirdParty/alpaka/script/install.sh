#!/bin/bash
#
# Copyright 2023 Benjamin Worpitz, Matthias Werner, René Widera, Antonio Di Pilato, Bernhard Manfred Gruber,
#                Simeon Ehrig, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

: ${ALPAKA_CI_ANALYSIS?"ALPAKA_CI_ANALYSIS must be specified"}
: ${ALPAKA_CI_INSTALL_CUDA?"ALPAKA_CI_INSTALL_CUDA must be specified"}
: ${ALPAKA_CI_INSTALL_HIP?"ALPAKA_CI_INSTALL_HIP must be specified"}
: ${ALPAKA_CI_INSTALL_TBB?"ALPAKA_CI_INSTALL_TBB must be specified"}

# the agc-manager only exists in the agc-container
# set alias to false, so each time if we ask the agc-manager if a software is installed, it will
# return false and the installation of software will be triggered
if [ "$ALPAKA_CI_OS_NAME" != "Linux" ] || [ ! -f "/usr/bin/agc-manager" ]
then
    echo "agc-manager is not installed"

    echo '#!/bin/bash' > agc-manager
    echo 'exit 1' >> agc-manager

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
    then
        sudo chmod +x agc-manager
        sudo mv agc-manager /usr/bin/agc-manager
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        chmod +x agc-manager
        mv agc-manager /usr/bin
    elif [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
    then
        sudo chmod +x agc-manager
        sudo mv agc-manager /usr/local/bin
    else
        echo "unknown operation system: ${ALPAKA_CI_OS_NAME}"
        exit 1
    fi
else
    echo "found agc-manager"
fi

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
    if [[ "${CXX}" == "icpx" ]] ;then source ./script/install_oneapi.sh ;fi
elif [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    echo "### list all applications ###"
    ls "/Applications/"
    echo "### end list all applications ###"
    sudo xcode-select -s "/Applications/Xcode_${ALPAKA_CI_XCODE_VER}.app/Contents/Developer"
fi

# Don't install TBB for oneAPI runners - it will be installed as part of oneAPI
if [ "${ALPAKA_CI_INSTALL_TBB}" = "ON" ] && [ "${CXX}" != "icpx" ]
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
