#!/bin/bash

#
# Copyright 2017 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

#-------------------------------------------------------------------------------
# e: exit as soon as one command returns a non-zero exit code.
#set -e

: ${ALPAKA_CI_BOOST_ROOT_DIR?"ALPAKA_CI_BOOST_ROOT_DIR must be specified"}
: ${ALPAKA_CI_BOOST_LIB_DIR?"ALPAKA_CI_BOOST_LIB_DIR must be specified"}
: ${CXX?"CXX must be specified"}
: ${CC?"CC must be specified"}

# --depth 1 does not necessarily always work.
# There seem to be problems when the super-project references a non-HEAD commit
# as the submodules are also cloned with --depth 1.
git clone -b "${ALPAKA_CI_BOOST_BRANCH}" --quiet --recursive --single-branch https://github.com/boostorg/boost.git "${ALPAKA_CI_BOOST_ROOT_DIR}"

# Prepare building of boost.
(cd "${ALPAKA_CI_BOOST_ROOT_DIR}"; sudo ./bootstrap.sh --with-toolset="${CC}")
(cd "${ALPAKA_CI_BOOST_ROOT_DIR}"; cat ./bootstrap.log)

# Create file links.
(cd "${ALPAKA_CI_BOOST_ROOT_DIR}"; sudo ./b2 headers)

# Prepare the library destination directory.
mkdir --parents "${ALPAKA_CI_BOOST_LIB_DIR}"

# Create the boost build command.
#  --layout=versioned
ALPAKA_BOOST_B2_CFLAGS="-fPIC"
ALPAKA_BOOST_B2_CXXFLAGS="-fPIC"
ALPAKA_BOOST_B2="sudo ./b2 -j1 --layout=tagged --toolset=${CC}"
ALPAKA_BOOST_B2+=" architecture=x86 address-model=64 variant=debug,release link=static threading=multi runtime-link=shared"
# Clang is not supported by the FindBoost script.
# boost (especially old versions) produces too much warnings when using clang (newer versions) so that the 4 MiB log is too short.
if [ "${CXX}" == "clang++" ]
then
    ALPAKA_BOOST_B2_CXXFLAGS+=" -Wunused-private-field -Wno-unused-local-typedef -Wno-c99-extensions -Wno-variadic-macros"
    if ( (( ALPAKA_CI_CLANG_VER_MAJOR >= 4 )) || ( (( ALPAKA_CI_CLANG_VER_MAJOR == 3 )) && (( ALPAKA_CI_CLANG_VER_MINOR >= 6 )) ) )
    then
        ALPAKA_BOOST_B2_CXXFLAGS+=" -Wno-unused-local-typedef"
    fi
fi
# Select the libraries required.
ALPAKA_BOOST_B2+=" --with-test"
if [ "${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE}" == "ON" ]
then
    ALPAKA_BOOST_B2_CXXFLAGS+=" -std=c++11"
    ALPAKA_BOOST_B2+=" --with-fiber --with-context --with-thread --with-system --with-atomic --with-chrono --with-date_time"
fi
if [ "${ALPAKA_BOOST_B2_CFLAGS}" != "" ]
then
    ALPAKA_BOOST_B2+=' cflags="'
    ALPAKA_BOOST_B2+="${ALPAKA_BOOST_B2_CFLAGS}"
    ALPAKA_BOOST_B2+='"'
fi
if [ "${ALPAKA_BOOST_B2_CXXFLAGS}" != "" ]
then
    ALPAKA_BOOST_B2+=' cxxflags="'
    ALPAKA_BOOST_B2+="${ALPAKA_BOOST_B2_CXXFLAGS}"
    ALPAKA_BOOST_B2+='"'
fi
ALPAKA_BOOST_B2+=" --stagedir=${ALPAKA_CI_BOOST_LIB_DIR} stage"
# Build boost.
echo "ALPAKA_BOOST_B2=${ALPAKA_BOOST_B2}"
(cd "${ALPAKA_CI_BOOST_ROOT_DIR}"; eval "${ALPAKA_BOOST_B2}")

# Clean the intermediate build files.
sudo rm -rf bin.v2
