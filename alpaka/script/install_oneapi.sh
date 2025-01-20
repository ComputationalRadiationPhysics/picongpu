#!/bin/bash
#
# Copyright 2023 Axel Hübl, Simeon Ehrig, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: install_oneapi>"

if agc-manager -e oneapi
then
    echo_green "<USE: preinstalled OneAPI ${ALPAKA_CI_ONEAPI_VERSION}>"
else
    echo_yellow "<INSTALL: Intel OneAPI ${ALPAKA_CI_ONEAPI_VERSION}>"

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

    # Workaround if icpx uses the stdlibc++. The stdlibc++-9 does not support C++20, therefore we install the stdlibc++-11. Clang automatically uses the latest stdlibc++ version.
    if [[ "$(cat /etc/os-release)" =~ "20.04" ]] && [ "${alpaka_CXX_STANDARD}" == "20" ];
    then
        travis_retry sudo apt install -y --no-install-recommends software-properties-common
        sudo apt-add-repository ppa:ubuntu-toolchain-r/test -y
        travis_retry sudo apt update
        travis_retry sudo apt install -y --no-install-recommends g++-11
    fi

    # path depends on the SDK version
    export CMAKE_CXX_COMPILER=$(which icpx)
fi

which "${CMAKE_CXX_COMPILER}"
${CMAKE_CXX_COMPILER} --version
sycl-ls
