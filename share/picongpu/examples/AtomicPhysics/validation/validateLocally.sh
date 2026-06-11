#!/usr/bin/env bash

# SPDX-FileCopyrightText: Brian Marre, Tapish Narwal
#
# SPDX-License-Identifier: GPL-3.0-or-later

#

# assume working picongpu with atomic physics environment

# build setup
pic-build -c "-DPARAM_FORCE_CONSTANT_ELECTRON_TEMPERATURE=true" 2>&1 | tee compile.result

# run test simulation
./bin/picongpu -r 4 -g 32 32 32 -d 1 1 1 --periodic 1 1 1 -s 25 --progressPeriod 1 --versionOnce 2>&1 | tee ../output.result

# install python dependencies
pip install -r ./validation/requirements.txt

# once all test simulations have run do evaluation
python ./validation/EvaluationScript.py
