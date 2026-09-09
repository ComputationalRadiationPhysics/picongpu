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
    Poisson solver for the electric field in the starting condition.
    """

    max_steps: Annotated[int, Field(..., gt=0)] = 2000
    """maximum number of iterations for the Poisson solver"""

    tolerance: Annotated[float, Field(..., gt=0.0)] = 1e-8
    """maximum tolerance for the Poisson solver"""

    preconditioner: Literal["default", "none"] = "default"
    """preconditioner for the Poisson solver"""

    preconditioner_max_steps: Annotated[int, Field(..., gt=0)] = 20
    """maximum number of iterations for the preconditioner"""

    @computed_field
    def preconditioner_disabled(self) -> bool:
        return self.preconditioner == "none"
