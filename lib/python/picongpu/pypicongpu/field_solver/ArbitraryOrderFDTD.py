"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, Field, computed_field
from ..rendering import RenderedObject


class ArbitraryOrderFDTDSolver(RenderedObject, BaseModel):
    """
    Arbitrary-order FDTD solver as defined by PIConGPU

    Approximates spatial derivatives by finite differences with a chosen number
    of neighbors (see maxwellSolver::ArbitraryOrderFDTD in
    include/picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/ArbitraryOrderFDTD.def).

    The C++ template parameter is the number of neighbors; the order of the
    solver is twice the number of neighbors.
    """

    neighbors: int = Field(ge=1)
    """number of neighbors used for the finite difference (order = 2 * neighbors)"""

    @computed_field
    def name(self) -> str:
        return f"ArbitraryOrderFDTD<{self.neighbors}>"
