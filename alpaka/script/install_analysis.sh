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

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    #-------------------------------------------------------------------------------
    # Install sloc
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install sloccount
    sloccount --version

    #-------------------------------------------------------------------------------
    # Install shellcheck
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install shellcheck
    shellcheck --version

elif [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    #-------------------------------------------------------------------------------
    # Install sloc
    brew install sloccount
    sloccount --version

    #-------------------------------------------------------------------------------
    # Install shellcheck
    brew install shellcheck
    shellcheck --version

fi
