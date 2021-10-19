#!/bin/bash

#
# Copyright 2021 Rene Widera
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis_retry.sh

source ./script/set.sh

: "${ALPAKA_CI_HIP_ROOT_DIR?'ALPAKA_CI_HIP_ROOT_DIR must be specified'}"

travis_retry apt-get -y --quiet update
travis_retry apt-get -y --quiet wget gnupg2
# AMD container keys are outdated and must be updated
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
travis_retry apt-get -y --quiet update

# AMD container are not shipped with rocrand/hiprand
travis_retry sudo apt-get -y --quiet install rocrand
