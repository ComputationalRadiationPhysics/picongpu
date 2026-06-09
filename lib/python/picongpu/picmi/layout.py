# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from functools import partial
from operator import gt, le

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from ..pypicongpu.species.operation.layout import OnePosition as PyPIConGPU_OnePosition
from ..pypicongpu.species.operation.layout import Quiet, Random


# This will inherit from PICMI_PseudoRandomLayout again
# once PICMI has switched to pydantic.
class PseudoRandomLayout(BaseModel):
    n_macroparticles_per_cell: int = Field(gt=0)
    n_macroparticles: None = None
    seed: None = None
    grid: None = None

    def get_as_pypicongpu(self):
        return Random(ppc=self.n_macroparticles_per_cell)


# This will inherit from PICMI_GriddedLayout again
# once PICMI has switched to pydantic.
class GriddedLayout(BaseModel):
    n_macroparticles_per_cell: tuple[int, int, int] = (1, 1, 1)
    grid: None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_as_pypicongpu(self):
        return Quiet(ppc=np.prod(self.n_macroparticles_per_cell), n_points=self.n_macroparticles_per_cell)

    @computed_field
    def in_cell_offsets(self) -> np.ndarray:
        return (np.mgrid[*map(slice, self.n_macroparticles_per_cell)] + 0.5).reshape(
            len(self.n_macroparticles_per_cell), -1
        ).T / self.n_macroparticles_per_cell


class OnePositionLayout(BaseModel):
    n_macroparticles_per_cell: int = Field(gt=0, description="Number of particles per cell")
    in_cell_offset: tuple[float, float, float] = Field(
        (0.0, 0.0, 0.0),
        description="Offset to cell origin where the particles are placed in units of cell size (between 0 and 1).",
    )
    grid: None = None

    @field_validator("in_cell_offset", mode="after")
    @classmethod
    def _validate_in_cell_offset(cls, in_cell_offset):
        if not (all(map(partial(le, 0.0), in_cell_offset)) and all(map(partial(gt, 1.0), in_cell_offset))):
            raise ValueError(f"All of in_cell_offset must be between 0 and 1. You gave: {in_cell_offset=}.")
        return in_cell_offset

    def get_as_pypicongpu(self):
        return PyPIConGPU_OnePosition(ppc=self.n_macroparticles_per_cell, in_cell_offset=self.in_cell_offset)


AnyLayout = PseudoRandomLayout | GriddedLayout | OnePositionLayout
