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

#To deploy the doxygen documentation a copy of the repository is created inside the deployed folder.
#This copy is always in the gh-pages branch consisting only of the containing files.
#This folder is ignored in all other branches.
#*NOTE:* This has already been done once and does not have to be repeated!
#On working branch:
#- Add deploy directory to `.gitignore` (if not already done)
#- Create the `gh-pages` branch: `git checkout --orphan gh-pages`
#- Clean the branch: `git rm -rf .`
#- Commit and push the branch: `git add --all`, `git commit -m"add gh-pages branch"`, `git push`

# Clone the gh-pages branch into the docs/doxygen/html folder.
git clone -b gh-pages https://x-access-token:${2}@github.com/${1}.git docs/doxygen/html

cd docs/

rm -rf doxygen/html/*
rm -rf doxygen/xml/*

doxygen Doxyfile

cd ../
