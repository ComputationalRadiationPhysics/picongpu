#!/usr/bin/env bash
#
# Copyright 2017-2021 Axel Huebl
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

# This file is a maintainer tool to bump the versions inside PIConGPU's
# source directory at all places where necessary.

# Maintainer Inputs ###########################################################

echo "Hi there, this is a PIConGPU maintainer tool to update the source"
echo "code of PIConGPU to a new version number on all places where"
echo "necessary."
echo "For it to work, you need write access on the source directory and"
echo "you should be working in a clean git branch without ongoing"
echo "rebase/merge/conflict resolves and without unstaged changes."

# check source dir
REPO_DIR=$(cd $(dirname $BASH_SOURCE)/../../../ && pwd)
echo
echo "Your current source directory is: $REPO_DIR"
echo

read -p "Are you sure you want to continue? [y/N] " -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "You did not confirm with 'y', aborting."
    exit 1
fi

echo "We will now run a few sed commands on your source directory."
echo "Please answer the following questions about the version number"
echo "you want to set first:"
echo

read -p "MAJOR version? (e.g. 1) " -r
MAJOR=$REPLY
echo
read -p "MINOR version? (e.g. 2) " -r
MINOR=$REPLY
echo
read -p "PATCH version? (e.g. 3) " -r
PATCH=$REPLY
echo
read -p "SUFFIX? (e.g. rc2, dev, ... or empty) " -r
SUFFIX=$REPLY
echo

if [[ -n "$SUFFIX" ]]
then
    SUFFIX_STR="-$SUFFIX"
fi

VERSION_STR="$MAJOR.$MINOR.$PATCH$SUFFIX_STR"

echo
echo "Your new version is: $VERSION_STR"
echo

read -p "Is this information correct? Will now start updating! [y/N] " -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "You did not confirm with 'y', aborting."
    exit 1
fi


# Updates #####################################################################

# PIConGPU version.hpp
#   include/picongpu/version.hpp
sed -i 's/'\
'[[:blank:]]*#[[:blank:]]*define[[:blank:]]\+PICONGPU_VERSION_MAJOR[[:blank:]]\+.*/'\
'#define PICONGPU_VERSION_MAJOR '$MAJOR'/g' \
    $REPO_DIR/include/picongpu/version.hpp
sed -i 's/'\
'[[:blank:]]*#[[:blank:]]*define[[:blank:]]\+PICONGPU_VERSION_MINOR[[:blank:]]\+.*/'\
'#define PICONGPU_VERSION_MINOR '$MINOR'/g' \
    $REPO_DIR/include/picongpu/version.hpp
sed -i 's/'\
'[[:blank:]]*#[[:blank:]]*define[[:blank:]]\+PICONGPU_VERSION_PATCH[[:blank:]]\+.*/'\
'#define PICONGPU_VERSION_PATCH '$PATCH'/g' \
    $REPO_DIR/include/picongpu/version.hpp
sed -i 's/'\
'[[:blank:]]*#[[:blank:]]*define[[:blank:]]\+PICONGPU_VERSION_LABEL[[:blank:]]\+.*/'\
'#define PICONGPU_VERSION_LABEL "'$SUFFIX'"/g' \
    $REPO_DIR/include/picongpu/version.hpp

# sphinx / RTD
#   docs/source/conf.py
sed -i "s/"\
"[[:blank:]]*version[[:blank:]]*=[[:blank:]]*u.*/"\
"version = u'$MAJOR.$MINOR.$PATCH'/g" \
    $REPO_DIR/docs/source/conf.py
sed -i "s/"\
"[[:blank:]]*release[[:blank:]]*=[[:blank:]]*u.*/"\
"release = u'$VERSION_STR'/g" \
    $REPO_DIR/docs/source/conf.py

# containers
#   share/picongpu/dockerfiles
sed -i 's/'\
'\/picongpu:[0-9]\+\.[0-9]\+\.[0-9]\+\(-.\+\)*/'\
'\/picongpu:'$VERSION_STR'/g' \
    $REPO_DIR/share/picongpu/dockerfiles/README.rst
sed -i 's/'\
'--tag [0-9]\+\.[0-9]\+\.[0-9]\+\(-.\+\)*/'\
'--tag '$VERSION_STR'/g' \
    $REPO_DIR/share/picongpu/dockerfiles/README.rst

sed -i 's/'\
'picongpu@[0-9]\+\.[0-9]\+\.[0-9]\+\(-.\+\)*/'\
'picongpu@'$VERSION_STR'/g' \
    $REPO_DIR/share/picongpu/dockerfiles/ubuntu-2004/Dockerfile

sed -i 's/'\
'\/picongpu:[0-9]\+\.[0-9]\+\.[0-9]\+\(-.\+\)*/'\
'\/picongpu:'$VERSION_STR'/g' \
    $REPO_DIR/share/picongpu/dockerfiles/ubuntu-2004/Singularity
sed -i 's/'\
'Version [0-9]\+\.[0-9]\+\.[0-9]\+\(-.\+\)*/'\
'Version '$VERSION_STR'/g' \
    $REPO_DIR/share/picongpu/dockerfiles/ubuntu-2004/Singularity

# @todo `project(...)` version in CMakeLists.txt (future)


# Epilog ######################################################################

echo
echo "Done. Please check your source, e.g. via"
echo "  git diff"
echo "now and commit the changes if no errors occured."
