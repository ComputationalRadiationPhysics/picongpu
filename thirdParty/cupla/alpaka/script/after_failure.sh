#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

source ./script/set.sh

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
  sudo smem
  sudo free -m -t
  # show actions of the OOM killer
  sudo dmesg
fi
