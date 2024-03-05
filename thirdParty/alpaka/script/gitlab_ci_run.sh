#!/bin/bash

#
# Copyright 2022 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

source ./script/set.sh

# inside the agc-container, the user is root and does not require sudo
# to compatibility to other container, fake the missing sudo command
if ! command -v sudo &> /dev/null
then
    cp ${CI_PROJECT_DIR}/script/gitlabci/fake_sudo.sh /usr/bin/sudo
    chmod +x /usr/bin/sudo
fi

source ./script/before_install.sh
source ./script/install.sh
source ./script/run.sh
