"""
This file is part of the PIConGPU.
Copyright 2025-2026 PIConGPU contributors
Authors: Edgar Marquardt
License: GPLv3+
"""

from typing import Annotated

from pydantic import BaseModel, Field

from .rendering import RenderedObject


class PoissonSolver(RenderedObject, BaseModel):
    """
    Poisson solver for the electric field in the starting condition.
    """

    max_steps: Annotated[int, Field(..., gt=0)] | None = None
    """maximum number of iterations for the Poisson solver"""

    tolerance: Annotated[float, Field(..., gt=0.0)] | None = None
    """maximum tolerance for the Poisson solver"""

    preconditioner_disabled: Annotated[bool, Field(...)] | None = None
    """disable preconditioner for the Poisson solver"""

    preconditioner_max_steps: Annotated[int, Field(..., gt=0)] | None = None
    """maximum number of iterations for the preconditioner"""
