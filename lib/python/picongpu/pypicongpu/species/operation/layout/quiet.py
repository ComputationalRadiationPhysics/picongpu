"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import partial
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, PlainSerializer


def serialise_vec(value) -> dict:
    # 2D3V: the C++ numParticlesPerDimension is built from a 3-component Int
    # (shrunk to simDim), so a 2-component offset (2D grid) is padded with z = 1.
    value = tuple(value)
    if len(value) == 2:
        value = (value[0], value[1], 1)
    return dict(zip("xyz", value))


def broadcast_validation(values, condition, message="Condition not met."):
    if not all(condition(value) for value in values):
        raise ValueError(f"{message} You gave: {values}.")
    return values


Vec3_int = Annotated[
    tuple[int, ...],
    PlainSerializer(serialise_vec),
    AfterValidator(
        partial(
            broadcast_validation,
            condition=lambda v: v > 0,
            message="Number of points must be greater than 0 in each direction.",
        )
    ),
]


class Quiet(BaseModel):
    type_quiet: Literal[True] = True
    n_points: Vec3_int = Field(default=(0, 0, 0))
    ppc: int = Field(gt=0)
    """particles per cell, >0"""
