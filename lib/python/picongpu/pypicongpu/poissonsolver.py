"""
This file is part of the PIConGPU.
Copyright 2025-2026 PIConGPU contributors
Authors: Edgar Marquardt
License: GPLv3+
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field

from .rendering import RenderedObject


class PoissonSolver(RenderedObject, BaseModel):
    """
    Bi-CGSTAB Poisson solver for the electric field in the starting condition.
    """

    max_steps: Annotated[int, Field(..., gt=0)]
    """maximum number of iterations for the Poisson solver"""

    tolerance: Annotated[float, Field(..., gt=0.0)]
    """maximum tolerance for the Poisson solver"""

    preconditioner: Literal["chebyshev", "none"]
    """preconditioner for the Poisson solver"""

    preconditioner_max_steps: Annotated[int, Field(..., gt=0)]
    """maximum number of iterations for the preconditioner"""

    boundary_condition: Literal["dirichlet", "homogeneous_dirichlet", "neumann", "homogeneous_neumann"]
    """boundary condition for the non-periodic boundaries in the Poisson solver"""

    boundary_function: Annotated[str, Field(..., min_length=1)] = "0.0"
    """function for the boundary condition in the Poisson solver"""

    @computed_field
    def preconditioner_disabled(self) -> bool:
        return self.preconditioner == "none"

    @computed_field
    def boundary_conditions_cpp(self) -> str:
        return {
            "dirichlet": "BoundaryConditionsDirichlet<BoundaryFunction>",
            "homogeneous_dirichlet": "BoundaryConditionsHomogeneousDirichlet",
            "neumann": "BoundaryConditionsNeumann<BoundaryFunction>",
            "homogeneous_neumann": "BoundaryConditionsHomogeneousNeumann",
        }[self.boundary_condition]
