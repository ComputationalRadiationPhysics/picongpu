#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

#-------------------------------------------------------------------------------
# -e: exit as soon as one command returns a non-zero exit code
# -o pipefail: pipeline returns exit code of the rightmost command with a non-zero exit code
# -u: treat unset variables as an error
# -v: Print shell input lines as they are read
# -x: Print command traces before executing command

# exit by default if the command does not return 0
# can be deactivated by setting the environment variable alpaka_DISABLE_EXIT_FAILURE
# for example for local debugging in a Docker container
if [ -z ${alpaka_DISABLE_EXIT_FAILURE+x} ]; then
    set -e
fi

set -ouvx pipefail
