#!/bin/bash
#
# Copyright 2023 Antonio Di Pilato, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

if [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    brew reinstall --build-from-source --formula ./script/homebrew/${ALPAKA_CI_XCODE_VER}/libomp.rb
fi
