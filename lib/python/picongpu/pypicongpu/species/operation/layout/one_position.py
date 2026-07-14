"""
This file is part of PIConGPU.
Copyright 2025-2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import partial
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, PlainSerializer


def serialise_vec(value) -> dict:
    return dict(zip("xyz", value))


def broadcast_validation(values, condition, message="Condition not met."):
    if not all(condition(value) for value in values):
        raise ValueError(f"{message} You gave: {values}.")
    return values


Vec3_float = Annotated[
    tuple[float, float, float],
    PlainSerializer(serialise_vec),
    AfterValidator(
        partial(
            broadcast_validation,
            condition=lambda v: v >= 0 and v < 1,
            message="All of in_cell_offset must be between 0 and 1.",
        )
    ),
]


class OnePosition(BaseModel):
    type_one_position: Literal[True] = True
    in_cell_offset: Vec3_float = Field(default=(0.0, 0.0, 0.0))
    """Offset inside of the cell relative to cell size, i.e., between 0 and 1"""
    ppc: int = Field(gt=0)
    """particles per cell, >0"""
