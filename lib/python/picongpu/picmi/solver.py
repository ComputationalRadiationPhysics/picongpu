"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ..pypicongpu import util, solver

import picmistandard

from typeguard import typechecked


@typechecked
class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
    """
    PICMI Electromagnic Solver

    See PICMI spec for full documentation.
    """

    def get_as_pypicongpu(self) -> solver.Solver:
        solver_by_method = {
            "Yee": solver.YeeSolver(),
        }

        if self.method not in solver_by_method:
            raise ValueError("unkown solver: {}".format(self.method))

        # todo: stencil order, cfl
        util.unsupported("stencil order", self.stencil_order)
        util.unsupported("field smoother", self.field_smoother)
        if self.method != "Yee":
            # for yee the cfl will be respected -- this behavior is coordinated
            # at the simulation class though
            util.unsupported("cfl", self.cfl)

        util.unsupported("l_nodal", self.l_nodal)
        util.unsupported("source smoother", self.source_smoother)
        util.unsupported("level of subcycling", self.subcycling)
        util.unsupported("galilean velocity", self.galilean_velocity)

        return solver_by_method[self.method]
