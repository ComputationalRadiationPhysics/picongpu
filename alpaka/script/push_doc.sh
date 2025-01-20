#!/bin/bash

#
# Copyright 2020 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

echo_green "<SCRIPT: push_doc>"

cd docs/doxygen/html

git config --global user.email "action@github.com"
git config --global user.name "GitHub Action"

git add -f .

git log -n 3

git diff --quiet && git diff --staged --quiet || (git commit -m "Update documentation skip-checks: true"; git push origin gh-pages)

cd ../../../
