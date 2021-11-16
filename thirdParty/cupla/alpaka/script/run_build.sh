#!/bin/bash

#
# Copyright 2014-2021 Benjamin Worpitz, Simeon Ehrig
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

cd build/

if [ -z "${ALPAKA_CI_BUILD_JOBS+x}" ]
then
    ALPAKA_CI_BUILD_JOBS=1
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    make VERBOSE=1 -j${ALPAKA_CI_BUILD_JOBS}
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
then
    "$MSBUILD_EXECUTABLE" "alpaka.sln" -p:Configuration=${CMAKE_BUILD_TYPE} -maxcpucount:${ALPAKA_CI_BUILD_JOBS} -verbosity:minimal
fi

cd ..
