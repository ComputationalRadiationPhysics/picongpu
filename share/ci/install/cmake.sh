#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

cmake_version="cmake@3.25"
echo "CMake-version: $cmake_version"
if agc-manager -e $cmake_version ; then
  export PATH=$(agc-manager -b cmake@3.25)/bin:$PATH
else
    # throw only a warning because for pypicongpu test we do not need CMake
    echo "WARNING: CMake 3.25 is not available" >&2
fi
