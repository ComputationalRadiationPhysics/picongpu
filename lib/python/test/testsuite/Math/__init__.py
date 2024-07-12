"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

This library contains some functions for calculating some mathematical
and physics functions, as well as some deviation calculations. It is
based on frequently used functions and is constantly being expanded.

Functions present in testsuite.Math are listed below.

physical functions:
-------------------

    calculateV_O
    calculateTime
    calculateTimeFreq
    plasmafrequence
    calculateBeta

deviation functions:
--------------------

    getDifference
    getMaxDifference
    getDifferenceInPercentage
    getAcceptanceRange
    getTestResult

mathematical functions:
-----------------------

    growthrate

For more information, see the documentation for the individual modules
"""

# To get sub-modules
from . import math
from . import physics
from . import deviation

__all__ = ["math", "physics", "deviation"]
__all__ += math.__all__
__all__ += physics.__all__
__all__ += deviation.__all__
