#!/bin/bash

# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

# provide sources for clang versions missing in the containers version 4.0
# the next line is a template to enable alternative clang versions which are not part of the release
# echo -e "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main\ndeb-src http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main" >> /etc/apt/sources.list.d/llvm.list

apt -y update

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
