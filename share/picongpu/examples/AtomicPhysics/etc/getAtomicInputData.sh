#!/usr/bin/env bash
#
# Copyright 2024 Brian Marre
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

# clone repository with reference atomic input data
(cd $HOME && git clone git@github.com:ComputationalRadiationPhysics/FLYonPICAtomicTestData.git)

# copy atomic input data to setup
cp $HOME/FLYonPICAtomicTestData/* ./atomicInputData/

# remove repository again
rm -rf $HOME/FLYonPICAtomicTestData