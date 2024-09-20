#!/bin/bash

#
# Copyright 2014-2021 Benjamin Worpitz, Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#
set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: run_build>"

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
