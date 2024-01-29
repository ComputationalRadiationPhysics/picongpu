#!/bin/bash

#
# Copyright 2023 Benjamin Worpitz, Bernhard Manfred Gruber, Simeon Ehrig, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_TBB_VERSION?'ALPAKA_CI_TBB_VERSION must be specified'}"

if ! agc-manager -e tbb@${ALPAKA_CI_TBB_VERSION}
then
    # Install TBB
    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
    then
        travis_retry sudo apt-get -qqq update
        travis_retry sudo apt-get install -y wget ca-certificates gnupg

        travis_retry sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

        travis_retry sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
        travis_retry sudo apt-get install -y intel-oneapi-tbb-devel-${ALPAKA_CI_TBB_VERSION}
    elif [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
    then
        brew install tbb
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]
    then
        echo "${TBB_ROOT}"
        TBB_DOWNLOAD_URL="https://github.com/oneapi-src/oneTBB/releases/download/v${ALPAKA_CI_TBB_VERSION}/oneapi-tbb-${ALPAKA_CI_TBB_VERSION}-win.zip"
        TBB_ZIP="tbb.zip"
        powershell.exe -Command '[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; Invoke-WebRequest "'${TBB_DOWNLOAD_URL}'" -OutFile "'${TBB_ZIP}'"'
        unzip -q "${TBB_ZIP}" -d "${TBB_ROOT}"
        rm "${TBB_ZIP}"
        INNER_FOLDER="${TBB_ROOT}/$(ls ${TBB_ROOT})"
        # run the mv in PS, because INNER_FOLDER contains backslashes (from github.workspace variable) and bash fails to glob
        powershell.exe -Command mv "${INNER_FOLDER}/*" "${TBB_ROOT}/"
        rm -r "${INNER_FOLDER}"
        export TBB_DIR="${TBB_ROOT}/lib/cmake/tbb"
    fi
fi
