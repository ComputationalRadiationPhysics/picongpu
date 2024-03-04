#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

source ./script/set.sh

./script/print_env.sh
source ./script/before_install.sh

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
  ./script/docker_ci.sh
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
  source ./script/install.sh
  ./script/run.sh
fi
