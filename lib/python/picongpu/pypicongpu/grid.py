"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Julian Lenz
License: GPLv3+
"""

import enum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, PlainSerializer, computed_field, model_validator
from typing_extensions import Self

from .rendering import RenderedObject


class BoundaryCondition(enum.Enum):
    """
    Boundary Condition of PIConGPU

    Maps to the ``--periodic`` grid option (``1`` = periodic, ``0`` = absorbing/open).
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


def serialise_vec3(value) -> dict:
    return dict(zip("xyz", value))


def serialise_vec2(value) -> dict:
    return dict(zip("xy", value))


Vec3_float = Annotated[tuple[float, float, float], PlainSerializer(serialise_vec3)]
Vec3_int = Annotated[tuple[int, int, int], PlainSerializer(serialise_vec3)]
Vec2_float = Annotated[tuple[float, float], PlainSerializer(serialise_vec2)]
Vec2_int = Annotated[tuple[int, int], PlainSerializer(serialise_vec2)]


def serialise_grid_dist3(value) -> None | dict[Literal["x", "y", "z"], list[dict[Literal["device_cells"], int]]]:
    return (
        value
        if value is None
        else {
            "x": [{"device_cells": x} for x in value[0]],
            "y": [{"device_cells": x} for x in value[1]],
            "z": [{"device_cells": x} for x in value[2]],
        }
    )


def serialise_grid_dist2(value) -> None | dict[Literal["x", "y"], list[dict[Literal["device_cells"], int]]]:
    return (
        value
        if value is None
        else {
            "x": [{"device_cells": x} for x in value[0]],
            "y": [{"device_cells": x} for x in value[1]],
        }
    )


def all_gt(iterable, m):
    if all(correct := [x > m for x in iterable]):
        return iterable
    else:
        message = f"{iterable=} contains values <= {m=} while all should be greater than m. Valid are the following: {correct=}."
        raise ValueError(message)


def all_ge(iterable, m):
    if all(correct := [x >= m for x in iterable]):
        return iterable
    else:
        message = f"{iterable=} contains values < {m=} while all should be greater than or equal to m. Valid are the following: {correct=}."
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

    The bounding box is implicitly given as ``cell_size * cell_cnt`` per axis.
    """

    cell_size: Annotated[Vec3_float, AfterValidator(lambda x: all_gt(x, 0))] = Field(alias="cell_size_si")
    """Width of individual cell in each direction"""

    cell_cnt: Annotated[Vec3_int, AfterValidator(lambda x: all_gt(x, 0))]
    """total number of cells in each direction"""

    boundary_condition: Annotated[
        tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
        PlainSerializer(lambda x: serialise_vec3(map(BoundaryCondition.get_cfg_str, x)), return_type=dict),
    ]
    """behavior towards particles crossing each boundary"""

    gpu_cnt: Annotated[Vec3_int, AfterValidator(lambda x: all_gt(x, 0))] = Field((1, 1, 1), alias="n_gpus")
    """number of GPUs in x y and z direction as 3-integer tuple"""

    grid_dist: Annotated[
        tuple[list[int], list[int], list[int]] | None,
        PlainSerializer(serialise_grid_dist3),
        AfterValidator(grid_dist_validate),
    ] = None
    """distribution of grid cells to GPUs for each axis"""

    super_cell_size: Vec3_int
    """size of super cell in x y and z direction as 3-integer tuple in cells"""

    guard_size: Annotated[Vec3_int | None, AfterValidator(lambda x: None if x is None else all_ge(x, 0))] = None
    """size of the guard region in x y and z direction as a 3-integer tuple in super cells"""

    @computed_field
    def has_z(self) -> bool:
        return True

    @computed_field
    def sim_dim(self) -> int:
        return 3

    @computed_field
    def cell_depth(self) -> float:
        """The Z cell length (CELL_DEPTH_SI), so the template can expand one field for both 2D and 3D grids."""
        return self.cell_size[2]

    @model_validator(mode="after")
    def check(self) -> Self:
        """serialized representation provided for RenderedObject"""
        if self.grid_dist is not None:
            assert sum(self.grid_dist[0]) == self.cell_cnt[0], "sum of grid_dists in x must be equal to number_of_cells"
            assert sum(self.grid_dist[1]) == self.cell_cnt[1], "sum of grid_dists in y must be equal to number_of_cells"
            assert sum(self.grid_dist[2]) == self.cell_cnt[2], "sum of grid_dists in z must be equal to number_of_cells"

        return self


class Grid2D(BaseModel, RenderedObject):
    """
    PIConGPU 2 dimensional (cartesian) grid

    2D3V simulation: two spatial dimensions (x and y, Z is dropped) but fields,
    momentum and velocities still carry three vector components.

    Defined by the dimensions of each cell and the number of cells per axis.
    """

    cell_size: Annotated[Vec2_float, AfterValidator(lambda x: all_gt(x, 0))] = Field(alias="cell_size_si")
    """Width of individual cell in each spatial direction"""

    cell_depth: Annotated[
        float,
        AfterValidator(lambda x: x if x > 0 else (_ for _ in ()).throw(ValueError("cell depth must be > 0"))),
    ] = Field(alias="cell_depth_si")
    """Z cell length (CELL_DEPTH_SI), the wire-particle integration length used to normalize densities"""

    cell_cnt: Annotated[Vec2_int, AfterValidator(lambda x: all_gt(x, 0))]
    """total number of cells in each spatial direction"""

    boundary_condition: Annotated[
        tuple[BoundaryCondition, BoundaryCondition],
        PlainSerializer(lambda x: serialise_vec2(map(BoundaryCondition.get_cfg_str, x)), return_type=dict),
    ]
    """behavior towards particles crossing each boundary"""

    gpu_cnt: Annotated[Vec2_int, AfterValidator(lambda x: all_gt(x, 0))] = Field((1, 1), alias="n_gpus")
    """number of GPUs in x and y direction as 2-integer tuple"""

    grid_dist: Annotated[
        tuple[list[int], list[int]] | None,
        PlainSerializer(serialise_grid_dist2),
        AfterValidator(grid_dist_validate),
    ] = None
    """distribution of grid cells to GPUs for each axis"""

    super_cell_size: Vec2_int
    """size of super cell in x and y direction as a 2-integer tuple in cells"""

    guard_size: Annotated[Vec2_int | None, AfterValidator(lambda x: None if x is None else all_ge(x, 0))] = None
    """size of the guard region in x and y direction as a 2-integer tuple in super cells"""

    @computed_field
    def has_z(self) -> bool:
        return False

    @computed_field
    def sim_dim(self) -> int:
        return 2

    @model_validator(mode="after")
    def check(self) -> Self:
        """serialized representation provided for RenderedObject"""
        if self.grid_dist is not None:
            assert sum(self.grid_dist[0]) == self.cell_cnt[0], "sum of grid_dists in x must be equal to number_of_cells"
            assert sum(self.grid_dist[1]) == self.cell_cnt[1], "sum of grid_dists in y must be equal to number_of_cells"

        return self


AnyGrid = Grid3D | Grid2D
