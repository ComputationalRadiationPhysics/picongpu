#!/bin/bash

#
# Copyright 2018-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis/set.sh

./script/travis/print_travisEnv.sh
source ./script/travis/before_install.sh

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
  ./script/travis/docker_install.sh
  ./script/travis/docker_run.sh
elif [ "$TRAVIS_OS_NAME" = "windows" ] || [ "$TRAVIS_OS_NAME" = "osx" ]
then
  ./script/travis/install.sh
  ./script/travis/run.sh
fi
