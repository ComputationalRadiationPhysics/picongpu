#!/bin/bash

#
# Copyright 2023 Benjamin Worpitz, Bernhard Manfred Gruber, Simeon Ehrig, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: install_tbb>"

: "${ALPAKA_CI_TBB_VERSION?'ALPAKA_CI_TBB_VERSION must be specified'}"

if agc-manager -e tbb@${ALPAKA_CI_TBB_VERSION}
then
    echo_green "<USE: preinstalled Intel TBB ${ALPAKA_CI_TBB_VERSION}>"
else
    echo_yellow "<INSTALL: Intel TBB ${ALPAKA_CI_TBB_VERSION}>"

    # Install TBB
    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]
    then
        retry_cmd sudo apt-get -qqq update
        retry_cmd sudo apt-get install -y wget ca-certificates gnupg

        retry_cmd sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

        retry_cmd sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
        retry_cmd sudo apt-get install -y intel-oneapi-tbb-devel-${ALPAKA_CI_TBB_VERSION}
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
        export ALPAKA_CI_TBB_DIR="${TBB_ROOT}/lib/cmake/tbb"
    fi
fi
