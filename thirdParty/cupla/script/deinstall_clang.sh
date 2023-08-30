#!/bin/bash

set -e
set -o pipefail


# deinstall clang because different clang OpenMP versions can not stay side by side
clang_version=$(echo $CXX_COMPILER | tr -d "clang++-")
echo "deinstall Clang-version: $clang_version"

if ! agc-manager -e clang@${clang_version}; then
    apt remove -y clang-${clang_version} libomp-${clang_version}-dev
fi
