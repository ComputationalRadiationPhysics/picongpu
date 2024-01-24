"""
This file is part of the PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

Note: only one point of data is currently covered
      in theory, more cases have yet to be added
"""

import testsuite.Math.deviation as dv
import testsuite._checkData as cD


def _calculate(theory, simulation):
    acceptance = cD.checkVariables(variable="acceptance")

    # only 1D
    max_diff = dv.getMinDifference(theory, simulation)
    perc = dv.getDifferenceInPercentage(theory, simulation)
    acc_range = dv.getAcceptanceRange(theory, acceptance)
    result = dv.getTestResult(theory, simulation, acceptance)

    return max_diff, perc, acc_range, result
