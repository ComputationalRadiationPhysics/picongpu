#!/bin/bash

#
# Copyright 2022 Simeon Ehrig
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/set.sh

if [ "${alpaka_ACC_GPU_HIP_ENABLE}" == "ON" ];
then
    apt-get -y --quiet update || echo "ignore any errors"
    apt-get -y --quiet install wget gnupg2
    # AMD container keys are outdated and must be updated
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
fi

source ./script/before_install.sh
source ./script/install.sh
source ./script/run.sh
