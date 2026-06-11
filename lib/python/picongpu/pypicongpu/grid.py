"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import enum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, PlainSerializer, model_validator
from typing_extensions import Self

from .rendering import RenderedObject


class BoundaryCondition(enum.Enum):
    """
    Boundary Condition of PIConGPU

    Defines how particles that pass the simulation bounding box are treated.

    TODO: implement the other methods supported by PIConGPU
    (reflecting, thermal)
    """

    PERIODIC = 1
    ABSORBING = 2

    def get_cfg_str(self) -> str:
        """
        Get string equivalent for cfg files
        :return: string for --periodic
        """
        literal_by_boundarycondition = {
            BoundaryCondition.PERIODIC: "1",
            BoundaryCondition.ABSORBING: "0",
        }
        return literal_by_boundarycondition[self]


def serialise_vec(value) -> dict:
    return dict(zip("xyz", value))


Vec3_float = Annotated[tuple[float, float, float], PlainSerializer(serialise_vec)]
Vec3_int = Annotated[tuple[int, int, int], PlainSerializer(serialise_vec)]


def serialise_grid_dist(value) -> None | dict[Literal["x", "y", "z"], list[dict[Literal["device_cells"], int]]]:
    return (
        value
        if value is None
        else {
            "x": [{"device_cells": x} for x in value[0]],
            "y": [{"device_cells": x} for x in value[1]],
            "z": [{"device_cells": x} for x in value[2]],
        }
    )


def all_gt(iterable, m):
    if all(correct := [x > m for x in iterable]):
        return iterable
    else:
        message = f"{iterable=} contains values <= {m=} while all should be greater than m. Valid are the following: {correct=}."
        raise ValueError(message)


def grid_dist_validate(grid_dist):
    if grid_dist is None:
        return None
    if all_gt(sum(grid_dist, []), 0):
        return grid_dist


class Grid3D(BaseModel, RenderedObject):
    """
    PIConGPU 3 dimensional (cartesian) grid

    Defined by the dimensions of each cell and the number of cells per axis.

    The bounding box is implicitly given as TODO.
    """

    cell_size: Annotated[Vec3_float, AfterValidator(lambda x: all_gt(x, 0))] = Field(alias="cell_size_si")
    """Width of individual cell in each direction"""

    cell_cnt: Annotated[Vec3_int, AfterValidator(lambda x: all_gt(x, 0))]
    """total number of cells in each direction"""

    boundary_condition: Annotated[
        tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
        PlainSerializer(lambda x: serialise_vec(map(BoundaryCondition.get_cfg_str, x)), return_type=dict),
    ]
    """behavior towards particles crossing each boundary"""

    gpu_cnt: Annotated[Vec3_int, AfterValidator(lambda x: all_gt(x, 0))] = Field((1, 1, 1), alias="n_gpus")
    """number of GPUs in x y and z direction as 3-integer tuple"""

    grid_dist: Annotated[
        tuple[list[int], list[int], list[int]] | None,
        PlainSerializer(serialise_grid_dist),
        AfterValidator(grid_dist_validate),
    ] = None
    """distribution of grid cells to GPUs for each axis"""

    super_cell_size: Vec3_int
    """size of super cell in x y and z direction as 3-integer tuple in cells"""

    @model_validator(mode="after")
    def check(self) -> Self:
        """serialized representation provided for RenderedObject"""
        if self.grid_dist is not None:
            assert sum(self.grid_dist[0]) == self.cell_cnt[0], "sum of grid_dists in x must be equal to number_of_cells"
            assert sum(self.grid_dist[1]) == self.cell_cnt[1], "sum of grid_dists in y must be equal to number_of_cells"
            assert sum(self.grid_dist[2]) == self.cell_cnt[2], "sum of grid_dists in z must be equal to number_of_cells"

        return self
