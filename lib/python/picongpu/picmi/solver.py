"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from picmistandard import PICMI_BinomialSmoother, PICMI_ElectromagneticSolver

from picongpu.pypicongpu import util
from picongpu.pypicongpu.field_solver import AnySolver, LeheSolver, YeeSolver


class BinomialSmoother(PICMI_BinomialSmoother):
    n_pass: None = None
    compensation: None = None
    stride: None = None
    alpha: None = None


class ElectromagneticSolver(PICMI_ElectromagneticSolver):
    """
    PICMI Electromagnic Solver

    See PICMI spec for full documentation.
    """

    source_smoother: BinomialSmoother | None = None
    field_smoother: None = None

    def get_as_pypicongpu(self) -> AnySolver:
        solver_by_method = {
            "Yee": YeeSolver(),
            "Lehe": LeheSolver(),
        }

        if self.method not in solver_by_method:
            raise ValueError("unkown solver: {}".format(self.method))

        # todo: stencil order, cfl
        util.unsupported("stencil order", self.stencil_order)
        if self.method != "Yee" and self.method != "Lehe":
            # for yee and Lehe the cfl will be respected -- this behavior is coordinated
            # at the simulation class though
            util.unsupported("cfl", self.cfl)

        util.unsupported("level of subcycling", self.subcycling)
        util.unsupported("galilean velocity", self.galilean_velocity)
        util.unsupported("divE cleaning", self.divE_cleaning)
        util.unsupported("divB cleaning", self.divB_cleaning)
        util.unsupported("pml divE cleaning", self.pml_divE_cleaning)
        util.unsupported("pml divB cleaning", self.pml_divB_cleaning)

        return solver_by_method[self.method]
