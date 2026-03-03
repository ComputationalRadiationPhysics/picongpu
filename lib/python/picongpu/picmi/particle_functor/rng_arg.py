"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from numbers import Integral
from typing import Any

import numpy as np
from pydantic import BaseModel, PrivateAttr, computed_field
from scipy.stats import norm, randint, uniform
from sympy import Symbol

from picongpu.pypicongpu.util import alt


def unpack_dist_loc_scale(dist, return_type, **kwargs):
    if dist == "uniform":
        if _is_integral(return_type):
            low, high = kwargs.get("range", _range_of(return_type))
            distribution = randint
        else:
            low, high = kwargs.get("range", (0.0, 1.0))
            distribution = uniform
        loc = low
        scale = high - low
        if scale < 0:
            raise ValueError(f"Requested reversed range of uniform random numbers. You gave {high=} < {low=}.")
    elif dist == "normal":
        if _is_integral(return_type):
            raise ValueError(
                f"PIConGPU does not support normal distributions of integral type. You gave: {return_type=}."
            )
        else:
            loc = kwargs.get("mean", 0.0)
            scale = kwargs.get("std", 1.0)
            distribution = norm
        if scale <= 0:
            raise ValueError(
                f"std={scale} must be greater than 0 for drawing from a normal distribution. You gave std={scale}."
            )
    else:
        raise ValueError(f"Unknown distribution for RNGArg. You gave: {dist=}.")
    return distribution, loc, scale


class RNGArg(BaseModel):
    _used_attributes: dict[Symbol, Any] = PrivateAttr(default_factory=dict)
    _counter: int = PrivateAttr(0)
    _dist_and_return_type: tuple[str, str | type] | None = PrivateAttr(None)

    @computed_field
    def dist(self) -> str | None:
        return alt(lambda: self._dist_and_return_type[0], None)

    @computed_field
    def return_type(self) -> str | type | None:
        return alt(lambda: self._dist_and_return_type[1], None)

    def get_attribute_map(self):
        return self._used_attributes

    def get(self, dist, return_type="float_X", shape=1, **kwargs):
        if self._dist_and_return_type is None:
            self._dist_and_return_type = (dist, return_type)
        else:
            if self._dist_and_return_type != (dist, return_type):
                raise ValueError(
                    "PIConGPU does not support drawing from multiple different distributions in one functor yet. "
                    f"You're trying to draw from {(dist, return_type)=} but previously you've drawn from {self._dist_and_return_type}."
                )

        my_symbols = []
        count = np.prod(shape)
        _, loc, scale = unpack_dist_loc_scale(dist, return_type, **kwargs)
        if count < 1:
            raise ValueError(f"You must draw at least one number but {shape=}.")
        for _ in range(count):
            my_symbol = Symbol(f"random{self._counter}")
            self._counter += 1
            self._used_attributes |= {my_symbol: ("random_number", (("loc", loc), ("scale", scale)))}
            my_symbols.append(my_symbol)
        if shape == 1:
            return my_symbol
        return np.reshape(my_symbols, shape=shape)

    def to_scipy(self, dist, return_type="float_X", **kwargs):
        distribution, loc, scale = unpack_dist_loc_scale(dist, return_type=return_type, **kwargs)
        return distribution(loc=loc, scale=scale)


def _is_integral(value):
    return isinstance(value, Integral) or (isinstance(value, str) and "int" in value)


def _range_of(value):
    # Should be extended to actually extract that information from the value.
    return (0, 2**32)
