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

source ./script/set.sh

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
