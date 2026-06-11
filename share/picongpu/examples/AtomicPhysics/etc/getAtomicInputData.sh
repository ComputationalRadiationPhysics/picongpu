#!/usr/bin/env bash

# SPDX-FileCopyrightText: Brian Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later

#

# clone repository with reference atomic input data
(cd $HOME && git clone git@github.com:ComputationalRadiationPhysics/FLYonPICAtomicTestData.git)

# copy atomic input data to setup
cp $HOME/FLYonPICAtomicTestData/* ./atomicInputData/

# remove repository again
rm -rf $HOME/FLYonPICAtomicTestData
