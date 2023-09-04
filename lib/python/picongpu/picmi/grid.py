"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from ..pypicongpu import grid
from ..pypicongpu import util

import picmistandard
import typeguard


@typeguard.typechecked
class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
    def __init__(self,
                 picongpu_n_gpus: list[int] | None = None,
                 **kw):
        """overwriting PICMI init to extract gpu distribution for PIConGPU
        :param picongpu_n_gpus: number of gpus for each dimension
            None matches to a single GPU (1, 1, 1)
            a single integer assumes parallelization in y (1, N, 1)
            a 3-integer-long list is distributed directly as (Nx, Ny, Nz)
        """
        self.picongpu_n_gpus = picongpu_n_gpus

        # continue with regular init
        super().__init__(**kw)

    def get_as_pypicongpu(self):
        # todo check
        assert [0, 0, 0] == self.lower_bound, "lower bounds must be 0, 0, 0"
        assert self.lower_boundary_conditions == \
            self.upper_boundary_conditions, \
            "upper and lower boundary conditions must be equal " \
            "(can only be chosen by axis, not by direction)"

        util.unsupported("moving window", self.moving_window_velocity)
        util.unsupported("refined regions", self.refined_regions, [])
        util.unsupported("lower bound (particles)",
                         self.lower_bound_particles,
                         self.lower_bound)
        util.unsupported("upper bound (particles)",
                         self.upper_bound_particles,
                         self.upper_bound)
        util.unsupported("lower boundary conditions (particles)",
                         self.lower_boundary_conditions_particles,
                         self.lower_boundary_conditions)
        util.unsupported("upper boundary conditions (particles)",
                         self.upper_boundary_conditions_particles,
                         self.upper_boundary_conditions)
        util.unsupported("guard cells", self.guard_cells)
        util.unsupported("pml cells", self.pml_cells)

        picongpu_boundary_condition_by_picmi_id = {
            "open": grid.BoundaryCondition.ABSORBING,
            "periodic": grid.BoundaryCondition.PERIODIC,
        }

        assert self.lower_boundary_conditions[0] in \
            picongpu_boundary_condition_by_picmi_id, \
            "X: boundary condition not supported"
        assert self.lower_boundary_conditions[1] in \
            picongpu_boundary_condition_by_picmi_id, \
            "Y: boundary condition not supported"
        assert self.lower_boundary_conditions[2] in \
            picongpu_boundary_condition_by_picmi_id, \
            "Z: boundary condition not supported"

        g = grid.Grid3D()
        g.cell_size_x_si = (self.upper_bound[0] - self.lower_bound[0]) \
            / self.number_of_cells[0]
        g.cell_size_y_si = (self.upper_bound[1] - self.lower_bound[1]) \
            / self.number_of_cells[1]
        g.cell_size_z_si = (self.upper_bound[2] - self.lower_bound[2]) \
            / self.number_of_cells[2]
        g.cell_cnt_x = self.number_of_cells[0]
        g.cell_cnt_y = self.number_of_cells[1]
        g.cell_cnt_z = self.number_of_cells[2]
        g.boundary_condition_x = \
            picongpu_boundary_condition_by_picmi_id[
                self.lower_boundary_conditions[0]]
        g.boundary_condition_y = \
            picongpu_boundary_condition_by_picmi_id[
                self.lower_boundary_conditions[1]]
        g.boundary_condition_z = \
            picongpu_boundary_condition_by_picmi_id[
                self.lower_boundary_conditions[2]]

        # gpu distribution
        # convert input to 3 integer list
        if self.picongpu_n_gpus is None:
            g.n_gpus = tuple([1, 1, 1])
        elif len(self.picongpu_n_gpus) == 1:
            assert self.picongpu_n_gpus[0] > 0, \
                "number of gpus must be positive integer"
            g.n_gpus = tuple([1, self.picongpu_n_gpus[0], 1])
        elif len(self.picongpu_n_gpus) == 3:
            for dim in range(3):
                assert self.picongpu_n_gpus[dim] > 0, \
                    "number of gpus must be positive integer"
            g.n_gpus = tuple(self.picongpu_n_gpus)
        else:
            raise ValueError("picongpu_n_gpus was neither None, "
                             "a 1-integer-list or a 3-integer-list")

        # check if gpu distribution fits grid
        # TODO: super_cell_size still hard coded
        super_cell_size = [8, 8, 4]
        cells = [
            self.number_of_cells[0],
            self.number_of_cells[1],
            self.number_of_cells[2]]
        dim_name = ["x", "y", "z"]
        for dim in range(3):
            assert (((cells[dim] // g.n_gpus[dim]) // super_cell_size[dim])
                    * g.n_gpus[dim] * super_cell_size[dim] == cells[dim]), \
                "GPU- and/or super-cell-distribution in {} dimension does " \
                "not match grid size".format(dim_name[dim])

        return g
