#!/bin/bash
#
# Copyright 2023 Axel Hübl, Simeon Ehrig, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${CXX?'CXX must be specified'}"


if ! agc-manager -e oneapi
then
    # Ref.: https://github.com/rscohn2/oneapi-ci
    # intel-basekit intel-hpckit are too large in size

    travis_retry sudo apt-get -qqq update
    travis_retry sudo apt-get install -y wget ca-certificates gnupg
    travis_retry sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB

    sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
    echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

    travis_retry sudo apt-get update

    #  See a list of oneAPI packages available for install
    echo "################################"
    sudo -E apt-cache pkgnames intel
    echo "################################"

    # The compiler will automatically pull in OpenMP and TBB as dependencies
    components=(
        intel-oneapi-common-vars                                      # Contains /opt/intel/oneapi/setvars.sh - has no version number
        intel-oneapi-compiler-dpcpp-cpp-"${ALPAKA_CI_ONEAPI_VERSION}" # Contains icpx compiler and SYCL runtime
        intel-oneapi-runtime-opencl                                   # Required to run SYCL tests on the CPU - has no version number
    )
    travis_retry sudo apt-get install -y "${components[@]}"

    set +eu
    source /opt/intel/oneapi/setvars.sh
    set -eu
fi

which "${CXX}"
${CXX} --version
which "${CC}"
${CC} --version
sycl-ls
