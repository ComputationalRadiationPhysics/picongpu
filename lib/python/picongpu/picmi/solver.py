# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

import picmistandard
import typeguard

from picongpu.pypicongpu import util
from picongpu.pypicongpu.field_solver import AnySolver, LeheSolver, YeeSolver


@typeguard.typechecked
class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
    """
    PICMI Electromagnic Solver

    See PICMI spec for full documentation.
    """

    def get_as_pypicongpu(self) -> AnySolver:
        solver_by_method = {
            "Yee": YeeSolver(),
            "Lehe": LeheSolver(),
        }

        if self.method not in solver_by_method:
            raise ValueError("unkown solver: {}".format(self.method))

        # todo: stencil order, cfl
        util.unsupported("stencil order", self.stencil_order)
        util.unsupported("field smoother", self.field_smoother)
        if self.method != "Yee" and self.method != "Lehe":
            # for yee and Lehe the cfl will be respected -- this behavior is coordinated
            # at the simulation class though
            util.unsupported("cfl", self.cfl)

        util.unsupported("source smoother", self.source_smoother)
        util.unsupported("level of subcycling", self.subcycling)
        util.unsupported("galilean velocity", self.galilean_velocity)
        util.unsupported("divE cleaning", self.divE_cleaning)
        util.unsupported("divB cleaning", self.divB_cleaning)
        util.unsupported("pml divE cleaning", self.pml_divE_cleaning)
        util.unsupported("pml divB cleaning", self.pml_divB_cleaning)

        return solver_by_method[self.method]
