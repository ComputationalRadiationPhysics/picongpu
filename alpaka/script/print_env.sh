#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: print_env>"

#-------------------------------------------------------------------------------
if [ "$alpaka_CI" = "GITHUB" ]
then
    echo GITHUB_WORKSPACE: "${GITHUB_WORKSPACE}"
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    # Show all running services
    sudo service --status-all

    # Show memory stats
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install smem
    sudo smem
    sudo free -m -t
fi
