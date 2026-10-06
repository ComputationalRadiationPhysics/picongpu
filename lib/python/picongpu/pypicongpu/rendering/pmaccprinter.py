"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from sympy import S
from sympy.printing.cxx import cxx_code_printers, reserved

PMACC_MATH_FUNCTIONS = {
    "Abs": "abs",
}

# sympy ships reserved words only up to C++17. PMacc/PIConGPU compile with C++20
# (see include/pmacc/CMakeLists.txt), so add the keywords sympy's list misses.
# ``register`` is still a reserved keyword in C++20 (only its use as a storage
# class was removed in C++17), and sympy also has a ``catch,`` typo in its C++98
# list. Keeping these on the printer makes it the single source of truth: every
# identifier it renders is escaped (e.g. ``requires`` -> ``requires_``), so
# callers no longer need their own keyword list.
_EXTRA_KEYWORDS = frozenset(
    {
        "char8_t",
        "concept",
        "consteval",
        "constinit",
        "co_await",
        "co_return",
        "co_yield",
        "import",
        "module",
        "register",
        "requires",
    }
)

_RESERVED_WORDS = (set(reserved["C++17"]) - {"catch,"}) | {"catch"} | _EXTRA_KEYWORDS


class PMAccPrinter(cxx_code_printers["c++17"]):
    # Replace the inherited C++17 set with the C++20 set PMacc actually compiles
    # with, so that ``_print_Symbol`` escapes every keyword we may emit.
    reserved_words = _RESERVED_WORDS

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
