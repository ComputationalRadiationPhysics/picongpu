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

source ./script/set.sh

#-------------------------------------------------------------------------------
if [ "$ALPAKA_CI" = "TRAVIS" ]
then
    # Print the travis environment variables: http://docs.travis-ci.com/user/ci-environment/
    echo TRAVIS_BRANCH: "${TRAVIS_BRANCH}"
    echo TRAVIS_BUILD_DIR: "${TRAVIS_BUILD_DIR}"
    echo TRAVIS_BUILD_ID: "${TRAVIS_BUILD_ID}"
    echo TRAVIS_BUILD_NUMBER: "${TRAVIS_BUILD_NUMBER}"
    echo TRAVIS_COMMIT: "${TRAVIS_COMMIT}"
    echo TRAVIS_COMMIT_RANGE: "${TRAVIS_COMMIT_RANGE}"
    echo TRAVIS_JOB_ID: "${TRAVIS_JOB_ID}"
    echo TRAVIS_JOB_NUMBER: "${TRAVIS_JOB_NUMBER}"
    echo TRAVIS_PULL_REQUEST: "${TRAVIS_PULL_REQUEST}"
    echo TRAVIS_SECURE_ENV_VARS: "${TRAVIS_SECURE_ENV_VARS}"
    echo TRAVIS_REPO_SLUG: "${TRAVIS_REPO_SLUG}"
    echo TRAVIS_OS_NAME: "${TRAVIS_OS_NAME}"
    echo TRAVIS_TAG: "${TRAVIS_TAG}"
elif [ "$ALPAKA_CI" = "GITHUB" ]
then
    echo GITHUB_WORKSPACE: "${GITHUB_WORKSPACE}"
fi

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
    # Show all running services
    sudo service --status-all

    # Stop some unnecessary services to save memory
    sudo /etc/init.d/mysql stop
    sudo /etc/init.d/postgresql stop
    sudo /etc/init.d/redis-server stop

    # Show memory stats
    sudo smem
    sudo free -m -t
fi
