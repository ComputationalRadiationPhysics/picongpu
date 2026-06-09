#!/bin/bash

# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

cmake_version="cmake@3.28"
echo "CMake-version: $cmake_version"
if agc-manager -e $cmake_version ; then
  export PATH=$(agc-manager -b cmake@3.28)/bin:$PATH
else
    # throw only a warning because for pypicongpu test we do not need CMake
    echo "WARNING: CMake 3.28 is not available" >&2
fi
