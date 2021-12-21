#!/bin/bash

#
# Copyright 2020 Benjamin Worpitz
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis_retry.sh

source ./script/set.sh

cd docs/doxygen/html

git config --global user.email "action@github.com"
git config --global user.name "GitHub Action"

git add -f .

git log -n 3

git diff --quiet && git diff --staged --quiet || (git commit -m "Update documentation skip-checks: true"; git push origin gh-pages)

cd ../../../
