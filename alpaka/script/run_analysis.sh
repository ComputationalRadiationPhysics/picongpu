#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: run_analysis>"

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    #-------------------------------------------------------------------------------
    # sloc
    sloccount .

    #-------------------------------------------------------------------------------
    # TODO/FIXME/HACK
    grep -r HACK ./* || true
    grep -r FIXME ./* || true
    grep -r TODO ./* || true

    #-------------------------------------------------------------------------------
    # check shell script with shellcheck
    find . -type f -name "*.sh" -exec shellcheck {} \;
fi
