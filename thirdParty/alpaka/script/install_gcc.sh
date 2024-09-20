#!/bin/bash

#
# Copyright 2022 Benjamin Worpitz, Simeon Ehrig, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: install_gcc>"

: "${ALPAKA_CI_GCC_VER?'ALPAKA_CI_GCC_VER must be specified'}"
: "${ALPAKA_CI_SANITIZERS?'ALPAKA_CI_SANITIZERS must be specified'}"

if agc-manager -e gcc@${ALPAKA_CI_GCC_VER}
then
    echo_green "<USE: preinstalled GCC ${ALPAKA_CI_GCC_VER}>"
else
    echo_yellow "<INSTALL: GCC ${ALPAKA_CI_GCC_VER}>"

    travis_retry sudo add-apt-repository -y ppa:ubuntu-toolchain-r/ppa # Contains gcc 10.4 (Ubuntu 20.04)
    travis_retry sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test # Contains gcc 11 (Ubuntu 20.04)
    travis_retry sudo apt-get -y --quiet update
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install g++-"${ALPAKA_CI_GCC_VER}"
fi

which g++-${ALPAKA_CI_GCC_VER}
export CMAKE_CXX_COMPILER=$(which g++-${ALPAKA_CI_GCC_VER})

# the g++ executalbe is required for compiling boost
# if it does not exist, create symbolic link to the install g++-${ALPAKA_CI_GCC_VER}
if ! command -v g++ >/dev/null; then
    echo_yellow "No g++ executable found."
    ln -s $(which g++-${ALPAKA_CI_GCC_VER}) $(dirname $(which g++-${ALPAKA_CI_GCC_VER}))/g++
fi

if [[ "${ALPAKA_CI_SANITIZERS}" == *"TSan"* ]]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libtsan0
fi

which "${CMAKE_CXX_COMPILER}"
${CMAKE_CXX_COMPILER} -v
