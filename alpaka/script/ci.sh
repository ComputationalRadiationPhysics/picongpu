#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: ci>"

./script/print_env.sh
source ./script/before_install.sh
if [ -n "$GITHUB_ACTIONS" ] && [ "$ALPAKA_CI_OS_NAME" = "Linux" ]; then
  # Workaround for the error: ThreadSanitizer: unexpected memory mapping
  # change the configuration of the address space layout randomization
  sudo sysctl vm.mmap_rnd_bits=28
fi

source ./script/install.sh
./script/run.sh
