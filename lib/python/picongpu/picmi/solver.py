"""
This file is part of PIConGPU.
Copyright 2021-2026 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Edgar Marquardt
License: GPLv3+
"""

from collections.abc import Sequence
from typing import Annotated, Literal, get_args
from pydantic import Field, computed_field

from picmistandard import PICMI_BinomialSmoother, PICMI_ElectromagneticSolver
from picmistandard.base import _PICMIModel
from picmistandard.fields import PICMI_AnyGrid

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


class PICMI_ElectrostaticSolver(_PICMIModel):
    """
    Electrostatic field solver
    """

    @computed_field
    def methods_list(self) -> list[str]:
        # Retained for backwards compatibility reasons.
        # The type annotation of `method` is the ground-truth.
        return list(get_args(type(self).__annotations__["method"]))

    grid: PICMI_AnyGrid = Field(description="Grid object for the diagnostic")

    method: Literal["FFT", "Multigrid"] | None = Field(
        default=None,
        description="The advance method use to solve the poisson equation. The default method is code dependent.",
    )

    required_precision: float | None = Field(default=None, description="The required precision for iterative solvers.")

    maximum_iterations: int | None = Field(
        default=None, description="The maximum number of iterations for iterative solvers."
    )


class ElectrostaticSolver(PICMI_ElectrostaticSolver):
    """
    PICMI Electrostatic Solver

    See PICMI spec for full documentation.

    Only the Poisson solver is supported; solver options that PIConGPU
    does not implement are rejected at construction time.
    """

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
