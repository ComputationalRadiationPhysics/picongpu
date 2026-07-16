"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Julian Lenz
License: GPLv3+
"""

from typing import Annotated
import picmistandard
from pydantic import AfterValidator, Field, computed_field

from ..pypicongpu import grid, util
from .copy_attributes import converts_to


def _normalise_type(kw, key, t):
    kw[key] = tuple(t(bound) for bound in kw[key])
    return kw


PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID = {
    "open": grid.BoundaryCondition.ABSORBING,
    "periodic": grid.BoundaryCondition.PERIODIC,
}


def _normalise_n_gpus(n_gpus) -> tuple[int, int, int]:
    picongpu_n_gpus = n_gpus
    n_gpus = tuple(n_gpus or tuple([1, 1, 1]))
    if len(n_gpus) == 1:
        n_gpus = tuple([1, n_gpus[0], 1])

    if len(n_gpus) != 3:
        raise ValueError(
            "The given number of gpus could not be mapped to a 3-component list of integers. "
            f"You gave {picongpu_n_gpus} and we interpreted this as {n_gpus=}."
        )

    if any(map(lambda x: x <= 0, n_gpus)):
        raise ValueError(
            f"Number of gpus must be positive integer(s). "
            f"You gave {picongpu_n_gpus=} and we interpreted this as {n_gpus=}."
        )

    return n_gpus


@converts_to(
    grid.Grid3D,
    preamble=lambda self: self.check(),
    conversions={
        "boundary_condition": lambda self: tuple(
            PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID[x] for x in self.lower_boundary_conditions
        ),
        "cell_cnt": "number_of_cells",
    },
    remove_prefix="picongpu_",
)
class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
    picongpu_n_gpus: Annotated[tuple[int, int, int], AfterValidator(_normalise_n_gpus)] = Field(default=(1, 1, 1))
    picongpu_grid_dist: None | list[list[int]] = Field(default=None)
    picongpu_super_cell_size: tuple[int, int, int] = Field(default=(8, 8, 4))

    @computed_field
    def picongpu_cell_size(self) -> tuple[int, int, int]:
        return (
            (self.upper_bound[0] - self.lower_bound[0]) / self.number_of_cells[0],
            (self.upper_bound[1] - self.lower_bound[1]) / self.number_of_cells[1],
            (self.upper_bound[2] - self.lower_bound[2]) / self.number_of_cells[2],
        )

    def get_cell_size(self):
        return self.picongpu_cell_size

    def check(self):
        # todo check
        if any(bound != 0.0 for bound in self.lower_bound):
            raise ValueError(
                f"A lower bound different from 0,0,0 is not supported in PIConGPU. You gave {self.lower_bound}."
            )
        if self.lower_boundary_conditions != self.upper_boundary_conditions:
            raise ValueError(
                "upper and lower boundary conditions must be equal (can only be chosen by axis, not by direction)"
            )
        util.unsupported("moving window", self.moving_window_velocity)
        util.unsupported("refined regions", self.refined_regions, [])
        util.unsupported("lower bound (particles)", self.lower_bound_particles, self.lower_bound)
        util.unsupported("upper bound (particles)", self.upper_bound_particles, self.upper_bound)
        util.unsupported(
            "lower boundary conditions (particles)",
            self.lower_boundary_conditions_particles,
            self.lower_boundary_conditions,
        )
        util.unsupported(
            "upper boundary conditions (particles)",
            self.upper_boundary_conditions_particles,
            self.upper_boundary_conditions,
        )
        util.unsupported("guard cells", self.guard_cells)
        util.unsupported("pml cells", self.pml_cells)

        if self.lower_boundary_conditions[0] not in PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID:
            raise ValueError("X: boundary condition not supported")
        if self.lower_boundary_conditions[1] not in PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID:
            raise ValueError("Y: boundary condition not supported")
        if self.lower_boundary_conditions[2] not in PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID:
            raise ValueError("Z: boundary condition not supported")

        if self.picongpu_grid_dist is not None:
            for i in range(3):
                if not all(n >= 1 for n in self.picongpu_grid_dist[i]):
                    raise ValueError("All values in grid distribution must be greater than 0.")
                if sum(self.picongpu_grid_dist[i]) != self.number_of_cells[i]:
                    raise ValueError(f"sum of grid distribution in dimension {i} must match number of cells")
                if len(self.picongpu_grid_dist[i]) != self.picongpu_n_gpus[i]:
                    raise ValueError(f"number of grid distributions in dimension {i} must match number of gpus")

        for i in range(3):
            if self.picongpu_super_cell_size[i] < 1:
                raise ValueError("super cell size must be an integer greater than 1")
        cells = [
            self.number_of_cells[0],
            self.number_of_cells[1],
            self.number_of_cells[2],
        ]
        dim_name = ["x", "y", "z"]
        for dim in range(3):
            if self.picongpu_grid_dist is None:
                if (
                    (cells[dim] // self.picongpu_n_gpus[dim]) // self.picongpu_super_cell_size[dim]
                ) * self.picongpu_n_gpus[dim] * self.picongpu_super_cell_size[dim] != cells[dim]:
                    raise ValueError(
                        "GPU- and/or super-cell-distribution in {} dimension does not match grid size".format(
                            dim_name[dim]
                        )
                    )
            else:
                # any returns true if there is at least one non zero (True) element
                if any([x % self.picongpu_super_cell_size[dim] for x in self.picongpu_grid_dist[dim]]):
                    raise ValueError(
                        f"grid distribution in {dim_name[dim]} dimension must be multiple of super cell size"
                    )
