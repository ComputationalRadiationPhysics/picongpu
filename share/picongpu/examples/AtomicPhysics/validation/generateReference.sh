#!/usr/bin/env bash

# SPDX-FileCopyrightText: Brian Marre, Tapish Narwal
#
# SPDX-License-Identifier: GPL-3.0-or-later

#

# assume environment with working picongpu with atomic physics

# build setup
pic-build -c "-DPARAM_FORCE_CONSTANT_ELECTRON_TEMPERATURE=true" 2>&1 | tee compile.result

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
