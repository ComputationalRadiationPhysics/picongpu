#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
then
  sudo smem
  sudo free -m -t
  # show actions of the OOM killer
  sudo dmesg
fi
