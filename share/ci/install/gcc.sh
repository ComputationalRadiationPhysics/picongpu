#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

gcc_version=$(echo $CXX_VERSION | tr -d "g++-")
echo "GCC-version: $gcc_version"

if ! agc-manager -e gcc@${gcc_version}; then
    apt install -y gcc-${gcc_version} g++-${gcc_version}
else
    GCC_BASE_PATH="$(agc-manager -b gcc@${gcc_version})"
    export PATH=$GCC_BASE_PATH/bin:$PATH
fi
