#!/usr/bin/env bash

# SPDX-FileCopyrightText: Brian Marre, Tapish Narwal
#
# SPDX-License-Identifier: GPL-3.0-or-later

#

# assume working picongpu with atomic physics environment

# once all test simulations have run do evaluation
python ./validation/EvaluationScript.py
