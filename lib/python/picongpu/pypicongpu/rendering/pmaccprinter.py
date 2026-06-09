# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from sympy import S
from sympy.printing.cxx import cxx_code_printers

PMACC_MATH_FUNCTIONS = {
    "Abs": "abs",
}


class PMAccPrinter(cxx_code_printers["c++17"]):
    # Originally, the C++ printers use `_ns = "std::"`.
    _ns = "pmacc::math::"
    _kf = cxx_code_printers["c++17"]._kf | PMACC_MATH_FUNCTIONS
    # The original math_macros contained macros like M_PI from math.h
    # We want to use the pmacc versions, so we remove this.
    math_macros = None

    def __init__(self, settings=None):
        super().__init__(
            (settings or {})
            | {
                "math_macros": {
                    S.Pi: "pmacc::math::Pi<float_X>::value",
                    S.Pi / 2: "pmacc::math::Pi<float_X>::halfValue",
                    S.Pi / 4: "pmacc::math::Pi<float_X>::quarterValue",
                    2 / S.Pi: "pmacc::math::Pi<float_X>::doubleReciprocalValue",
                }
            }
        )
