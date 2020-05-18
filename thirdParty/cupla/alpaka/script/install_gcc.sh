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

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_GCC_VER?'ALPAKA_CI_GCC_VER must be specified'}"
: "${ALPAKA_CI_SANITIZERS?'ALPAKA_CI_SANITIZERS must be specified'}"
: "${CXX?'CXX must be specified'}"

travis_retry sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
travis_retry sudo apt-get -y --quiet update

travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install g++-"${ALPAKA_CI_GCC_VER}"
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-"${ALPAKA_CI_GCC_VER}" 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-"${ALPAKA_CI_GCC_VER}" 50
if [[ "${ALPAKA_CI_SANITIZERS}" == *"TSan"* ]]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libtsan0
fi

which "${CXX}"
${CXX} -v
