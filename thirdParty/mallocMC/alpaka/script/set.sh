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

#-------------------------------------------------------------------------------
# -e: exit as soon as one command returns a non-zero exit code
# -o pipefail: pipeline returns exit code of the rightmost command with a non-zero exit code
# -u: treat unset variables as an error
# -v: Print shell input lines as they are read
# -x: Print command traces before executing command
set -eouvx pipefail
