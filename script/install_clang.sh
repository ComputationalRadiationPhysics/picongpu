#!/bin/bash

set -e
set -o pipefail

# install all required clang version including OpenMP
clang_version=$(echo $CXX_COMPILER | tr -d "clang++-")
echo "Clang-version: $clang_version"

if ! agc-manager -e clang@${clang_version}; then
    apt update
    apt install -y clang-${clang_version}
    apt install -y libomp-${clang_version}-dev
else
    CLANG_BASE_PATH="$(agc-manager -b clang@${clang_version})"
    export PATH=$CLANG_BASE_PATH/bin:$PATH
fi
