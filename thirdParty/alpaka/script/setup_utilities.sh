#!/usr/bin/env bash

# SPDX-License-Identifier: MPL-2.0

# serveral helper function and tools for the CI
# the script should be source everywhere, the utils are required
# if a bash script is normal called, self defined bash functions are not avaible from the calling bash instance


# exit by default if the command does not return 0
# can be deactivated by setting the environment variable alpaka_DISABLE_EXIT_FAILURE
# for example for local debugging in a Docker container
if [ -z ${alpaka_DISABLE_EXIT_FAILURE+x} ]; then
    set -e
fi

# disable command traces for the following scripts to avoid useless noise in the job output
source ./script/setup_utilities/color_echo.sh
source ./script/setup_utilities/travis_retry.sh
source ./script/setup_utilities/sudo.sh
source ./script/setup_utilities/agc-manager.sh
source ./script/setup_utilities/set.sh
