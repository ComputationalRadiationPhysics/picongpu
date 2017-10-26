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

source ./script/travis/travis_retry.sh

#-------------------------------------------------------------------------------
# e: exit as soon as one command returns a non-zero exit code.
set -e

travis_retry apt-get -y --quiet update
travis_retry apt-get -y install sudo

# software-properties-common: 'add-apt-repository' and certificates for wget https download
# binutils: ld
# xz-utils: xzcat
travis_retry sudo apt-get -y --quiet --allow-unauthenticated --no-install-recommends install software-properties-common wget git make binutils xz-utils

./script/travis/install_cmake.sh
if [ "${ALPAKA_CI_ANALYSIS}" == "ON" ] ;then ./script/travis/install_analysis.sh ;fi
# Install CUDA before installing gcc as it installs gcc-4.8 and overwrites our selected compiler
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "ON" ] ;then ./script/travis/install_cuda.sh ;fi
if [ "${CXX}" == "g++" ] ;then ./script/travis/install_gcc.sh ;fi
if [ "${CXX}" == "clang++" ] ;then source ./script/travis/install_clang.sh ;fi
if [ "${ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE}" == "ON" ] ;then ./script/travis/install_tbb.sh ;fi
./script/travis/install_boost.sh

# Minimize docker image size
sudo apt-get --quiet --purge autoremove
sudo apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
