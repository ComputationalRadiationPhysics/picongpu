"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Richard Pausch, Julian Lenz
License: GPLv3+
"""

import numpy as np

_EPSILON = 1.0e-6


class UnitDimension:
    """
    class to describe units
    """

    # number of unit dimension
    N_unit_dim = 7

    """
    The unit index map associates names with array indices.
    The name definition can be found at https://en.wikipedia.org/wiki/SI_base_unit.
    The array index order follows that of PIConGPU:
    include/picongpu/plugins/binning/UnitConversion.hpp, lines 40-48.
    """
    _unit_index_map = {
        "length": 0,
        "L": 0,
        "mass": 1,
        "M": 1,
        "time": 2,
        "T": 2,
        "electric current": 3,
        "I": 3,
        "thermodynamic temperature": 4,
        "Θ": 4,
        "amount of substance": 5,
        "N": 5,
        "luminous intensity": 6,
        "J": 6,
    }

    def __init__(self, other=None, **kwargs):
        """set unit vector either empty or by name"""
        self.unit_vector = np.zeros(self.N_unit_dim)
        if other is not None:
            if len(kwargs) > 0:
                raise ValueError(
                    "Handing `other` and `kwargs` to `UnitDimension` is mutually exclusive. You gave {other=}, {kwargs=}."
                )
            if isinstance(other, UnitDimension):
                other = other.unit_vector
            self.unit_vector = np.asarray(other, dtype=float)
        else:
            for key, val in kwargs.items():
                self.unit_vector[self._unit_index_map[key]] = val

    def __getitem__(self, name):
        """access component by name"""
        index = self._unit_index_map.get(name)
        if index is None:
            raise KeyError(f"Unknown unit name: {name}")
        return self.unit_vector[index]

    def __iter__(self):
        """return iterator of unit based on PIConGPU order: LMTIΘNJ"""
        return iter(self.unit_vector)

    def __str__(self):
        """return string representation that only outputs relevant units"""
        return " ".join(
            f"{name}^{val}"
            for name, index in self._unit_index_map.items()
            # This assumes that we have a short name of length 1 for each unit dimension.
            if len(name) == 1 and np.abs(val := self.unit_vector[index]) > _EPSILON
        )

    def __pow__(self, exponent):
        """rase unit to a power"""
        return UnitDimension(self.unit_vector * exponent)

    def __mul__(self, factor):
        """multiply units with each other"""
        return UnitDimension(self.unit_vector + factor.unit_vector)

    def __truediv__(self, divisor):
        """divide one unit by another"""
        return self * (divisor**-1)


# predefined unit dimensions
T = UnitDimension(T=1)
M = UnitDimension(M=1)
L = UnitDimension(L=1)
# ruff's rule E741 forbids ambiguous names
# "I" is undoubtedly not a great variable name
# but we clearly got a good reason here
I = UnitDimension(I=1)  # noqa: E741
