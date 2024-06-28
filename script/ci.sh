#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

source ./script/set.sh

./script/print_env.sh
source ./script/before_install.sh
if [ -n "$GITHUB_ACTIONS" ] && [ "$ALPAKA_CI_OS_NAME" = "Linux" ]; then
  # Workaround for the error: ThreadSanitizer: unexpected memory mapping
  # change the configuration of the address space layout randomization
  sudo sysctl vm.mmap_rnd_bits=28
fi

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
  ./script/docker_ci.sh
elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ] || [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
  source ./script/install.sh
  ./script/run.sh
fi
