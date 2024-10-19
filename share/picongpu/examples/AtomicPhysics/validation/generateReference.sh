#!/usr/bin/env bash
#
# Copyright 2024 Brian Marre, Tapish Narwal
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.

# assume environment with working picongpu with atomic physics

# build setup
pic-build -c "-DPARAM_FORCE_CONSTANT_ELECTRON_TEMPERATURE" 2>&1 | tee compile.result

# run test 10 simulations
for i in $(seq 1 10)
do
    ./bin/picongpu -r 4 -g 32 32 32 -d 1 1 1 --periodic 1 1 1 -s 25 --progressPeriod 1 --openPMD.period 1 --versionOnce 2>&1 | tee ./output_$i.result
    mkdir ./output_$i
    mv ./output_$i.result ./output_$i/
    mv ./binningOpenPMD/atomicStateBinning_*.bp ./output_$i/
done

# install python dependencies
pip install -r ./validation/requirements.txt

# once all test simulations have run do evaluation
python ./validation/GenerateReference.py
