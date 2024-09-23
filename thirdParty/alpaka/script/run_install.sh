#!/bin/bash

#
# Copyright 2021 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: run_install>"

ALPAKA_CI_CMAKE_EXECUTABLE=cmake
if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
    ALPAKA_CI_CMAKE_EXECUTABLE="${ALPAKA_CI_CMAKE_DIR}/bin/cmake"
fi

"${ALPAKA_CI_CMAKE_EXECUTABLE}" --install build --config ${CMAKE_BUILD_TYPE}
