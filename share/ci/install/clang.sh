#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

clang_version=$(echo $CXX_VERSION | tr -d "clang++-")
echo "Clang-version: $clang_version"

if ! agc-manager -e clang@${clang_version}; then
    apt install -y clang-${clang_version}
    if [[ "$PIC_BACKEND" =~ omp2b.* ]] ; then
        apt install -y libomp-${clang_version}-dev
    fi
else
    CLANG_BASE_PATH="$(agc-manager -b clang@${clang_version})"
    export PATH=$CLANG_BASE_PATH/bin:$PATH
fi
