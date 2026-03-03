"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from typing import Literal

from picmistandard import PICMI_BinomialSmoother, PICMI_ElectromagneticSolver

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

    field_smoother: None = None
    method: Literal["Yee", "Lehe"]
    stencil_order: None = None
    subcycling: None = None
    galilean_velocity: None = None
    divE_cleaning: None = None
    divB_cleaning: None = None
    pml_divE_cleaning: None = None
    pml_divB_cleaning: None = None

    def get_as_pypicongpu(self) -> AnySolver:
        return YeeSolver() if self.method == "Yee" else LeheSolver()
