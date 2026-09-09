"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from collections.abc import Sequence
from typing import Annotated, Literal
from pydantic import Field

from picmistandard import PICMI_BinomialSmoother, PICMI_ElectromagneticSolver, PICMI_ElectrostaticSolver

from picongpu.pypicongpu import util
from picongpu.pypicongpu.field_solver import AnySolver, LeheSolver, YeeSolver
from picongpu.pypicongpu.poissonsolver import PoissonSolver


class BinomialSmoother(PICMI_BinomialSmoother):
    """
    PICMI Binomial Smoother

    PIConGPU's binomial current deposition uses fixed parameters, so all
    standard parameters except `n_pass` (which must be given by the standard
    but is not used) are rejected.
    """

    compensation: Annotated[Sequence[bool] | None, util.rejects_unsupported("binomial smoother parameters")] = None
    stride: Annotated[Sequence[int] | None, util.rejects_unsupported("binomial smoother parameters")] = None
    alpha: Annotated[Sequence[float] | None, util.rejects_unsupported("binomial smoother parameters")] = None


class ElectromagneticSolver(PICMI_ElectromagneticSolver):
    """
    PICMI Electromagnic Solver

    See PICMI spec for full documentation.

    Only the Yee and Lehe solvers are supported; solver options that PIConGPU
    does not implement are rejected at construction time.
    """

    field_smoother: Annotated[PICMI_BinomialSmoother | None, util.rejects_unsupported("field smoothers")] = None
    method: Literal["Yee", "Lehe"]
    stencil_order: Annotated[Sequence[int] | None, util.rejects_unsupported("higher order solver stencils")] = None
    subcycling: Annotated[int | None, util.rejects_unsupported("subcycling")] = None
    galilean_velocity: Annotated[Sequence[float] | None, util.rejects_unsupported("galilean velocity")] = None
    divE_cleaning: Annotated[bool | None, util.rejects_unsupported("divE cleaning")] = None
    divB_cleaning: Annotated[bool | None, util.rejects_unsupported("divB cleaning")] = None
    pml_divE_cleaning: Annotated[bool | None, util.rejects_unsupported("pml divE cleaning")] = None
    pml_divB_cleaning: Annotated[bool | None, util.rejects_unsupported("pml divB cleaning")] = None

    def get_as_pypicongpu(self) -> AnySolver:
        return YeeSolver() if self.method == "Yee" else LeheSolver()


class ElectrostaticSolver(PICMI_ElectrostaticSolver):
    """
    PICMI Electrostatic Solver

    See PICMI spec for full documentation.

    Only the Poisson solver is supported; solver options that PIConGPU
    does not implement are rejected at construction time.
    """

    method: Literal["Poisson"] = "Poisson"
    required_precision: Annotated[float, Field(..., gt=0.0)] = 1e-8
    maximum_iterations: Annotated[int, Field(..., gt=0)] = 2000
    preconditioner: Literal["default", "none"] = "default"
    preconditioner_maximum_iterations: Annotated[int, Field(..., gt=0)] = 20

    def get_as_pypicongpu(self) -> PoissonSolver:
        return PoissonSolver(
            tolerance=self.required_precision,
            max_steps=self.maximum_iterations,
            preconditioner=self.preconditioner,
            preconditioner_max_steps=self.preconditioner_maximum_iterations,
        )
