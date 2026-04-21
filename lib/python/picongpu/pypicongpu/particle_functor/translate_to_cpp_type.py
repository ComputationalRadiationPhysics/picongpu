"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from numbers import Integral, Real


def translate_to_cpp_type(return_type):
    try:
        # Ordering is important here because issubclass(bool, int) is True in Python world
        if issubclass(return_type, bool):
            return "bool"
        if issubclass(return_type, Integral):
            return "int"
        if issubclass(return_type, Real):
            return "float_X"
    except TypeError:
        pass
    if isinstance(return_type, str):
        return return_type
    raise ValueError(f"Cannot translate {return_type=} to a C++ type.")


def translate_from_cpp_type(return_type: str) -> type:
    if return_type == "bool":
        return bool
    if "int" in return_type:
        return int
    if "float" in return_type or "double" in return_type:
        return float
    raise ValueError(f"No known translation from C++ {return_type=} to python.")
