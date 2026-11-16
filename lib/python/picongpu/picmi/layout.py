"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from collections.abc import Sequence
from functools import partial
from operator import gt, le

import numpy as np
import picmistandard
from pydantic import Field, computed_field, field_validator

from ..pypicongpu.species.operation.layout import OnePosition as PyPIConGPU_OnePosition
from ..pypicongpu.species.operation.layout import Quiet, Random


class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
    n_macroparticles_per_cell: int = Field(
        gt=0,
        description="Number of macroparticles to load per cell. Either this argument or n_macroparticles "
        "should be supplied (not both).",
    )
    # PIConGPU can't handle the following separately:
    n_macroparticles: None = Field(
        None,
        description="Not supported by PIConGPU: the particle count is set per cell via "
        "n_macroparticles_per_cell, so n_macroparticles must be None.",
    )
    seed: None = Field(
        None,
        description="Not supported by PIConGPU: the pseudo-random number generator seed cannot be chosen, "
        "so seed must be None.",
    )
    grid: None = Field(
        None,
        description="Not supported by PIConGPU: the underlying grid is always used, so grid must be None.",
    )

    def get_as_pypicongpu(self):
        return Random(ppc=self.n_macroparticles_per_cell)


class GriddedLayout(picmistandard.PICMI_GriddedLayout):
    n_macroparticles_per_cell: list[int] = Field(
        min_length=1,
        max_length=3,
        description="Number of particles per cell along each axis (one entry per grid dimension, or a single "
        "number in 1D). Must be given explicitly, and every entry must be a positive integer.",
    )

    def __init__(self, *args, **kwargs):
        if "n_macroparticle_per_cell" in kwargs:
            raise TypeError(
                "The deprecated spelling 'n_macroparticle_per_cell' is no longer supported. "
                "Use 'n_macroparticles_per_cell' instead."
            )
        super().__init__(*args, **kwargs)

    # The pinned picmistandard still exposes the deprecated spelling as a
    # property with a setter. Shadow it so that both attribute access and
    # assignment raise, not only construction via ``__init__``.
    @property
    def n_macroparticle_per_cell(self):
        raise AttributeError(
            "The deprecated spelling 'n_macroparticle_per_cell' is no longer supported. "
            "Use 'n_macroparticles_per_cell' instead."
        )

    @n_macroparticle_per_cell.setter
    def n_macroparticle_per_cell(self, value):
        raise AttributeError(
            "The deprecated spelling 'n_macroparticle_per_cell' is no longer supported. "
            "Use 'n_macroparticles_per_cell' instead."
        )

    @field_validator("n_macroparticles_per_cell", mode="after")
    @classmethod
    def _require_positive(cls, value):
        if not all(v >= 1 for v in value):
            raise ValueError(f"All n_macroparticles_per_cell must be greater than or equal to 1. You gave: {value}.")
        return value

    def get_as_pypicongpu(self):
        return Quiet(ppc=np.prod(self.n_macroparticles_per_cell), n_points=self.n_macroparticles_per_cell)

    @computed_field
    def in_cell_offsets(self) -> Sequence[Sequence[float]]:
        return (np.mgrid[*map(slice, self.n_macroparticles_per_cell)] + 0.5).reshape(
            len(self.n_macroparticles_per_cell), -1
        ).T / self.n_macroparticles_per_cell


class OnePositionLayout(picmistandard.PICMI_Layout):
    n_macroparticles_per_cell: int = Field(gt=0, description="Number of particles per cell")
    in_cell_offset: tuple[float, ...] = Field(
        (0.0, 0.0, 0.0),
        description="Offset to cell origin where the particles are placed in units of cell size (between 0 and 1). "
        "Two components for a 2D grid, three for a 3D grid (the third, z, is ignored in 2D3V).",
    )
    grid: None = None

    @field_validator("in_cell_offset", mode="after")
    @classmethod
    def _validate_in_cell_offset(cls, in_cell_offset):
        if len(in_cell_offset) not in (2, 3):
            raise ValueError(
                f"in_cell_offset must have 2 (2D) or 3 (3D) components. You gave {in_cell_offset=} with {len(in_cell_offset)}."
            )
        if not (all(map(partial(le, 0.0), in_cell_offset)) and all(map(partial(gt, 1.0), in_cell_offset))):
            raise ValueError(f"All of in_cell_offset must be between 0 and 1. You gave: {in_cell_offset=}.")
        return in_cell_offset

    def get_as_pypicongpu(self):
        return PyPIConGPU_OnePosition(ppc=self.n_macroparticles_per_cell, in_cell_offset=self.in_cell_offset)


AnyLayout = PseudoRandomLayout | GriddedLayout | OnePositionLayout
