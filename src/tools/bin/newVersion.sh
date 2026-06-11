#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Axel Huebl
#
# SPDX-License-Identifier: GPL-3.0-or-later
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

# @todo `project(...)` version in CMakeLists.txt (future)


# Epilog ######################################################################

echo
echo "Done. Please check your source, e.g. via"
echo "  git diff"
echo "now and commit the changes if no errors occured."
