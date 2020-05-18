#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis_retry.sh

source ./script/set.sh

# Install TBB
if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install libtbb-dev
elif [ "$TRAVIS_OS_NAME" = "osx" ]
then
    brew unlink python@2
    brew install tbb
elif [ "$TRAVIS_OS_NAME" = "windows" ]
then
    TBB_ARCHIVE_VER="tbb44_20160526oss"
    TBB_DOWNLOAD_URL="https://github.com/intel/tbb/releases/download/4.4.5/${TBB_ARCHIVE_VER}_win.zip"
    TBB_DST_PATH="tbb.zip"
    powershell.exe -Command '[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; Invoke-WebRequest "'${TBB_DOWNLOAD_URL}'" -OutFile "'${TBB_DST_PATH}'"'
    mkdir "${TBB_ROOT}"
    unzip -q "${TBB_DST_PATH}" -d "${TBB_ROOT}"
    rm "${TBB_DST_PATH}"
    TBB_UNZIP_DIR="${TBB_ROOT}/${TBB_ARCHIVE_VER}"
    mv ${TBB_UNZIP_DIR}/* "${TBB_ROOT}/"
    rmdir "${TBB_UNZIP_DIR}"
fi
