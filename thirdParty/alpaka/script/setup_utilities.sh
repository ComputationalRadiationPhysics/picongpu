#!/usr/bin/env bash

# SPDX-License-Identifier: MPL-2.0

# serveral helper function and tools for the CI
# the script should be source everywhere, the utils are required
# if a bash script is normal called, self defined bash functions are not avaible from the calling bash instance


set -e

# disable command traces for the following scripts to avoid useless noise in the job output
source ./script/setup_utilities/travis_retry.sh
source ./script/setup_utilities/sudo.sh
source ./script/setup_utilities/agc-manager.sh
source ./script/setup_utilities/set.sh
