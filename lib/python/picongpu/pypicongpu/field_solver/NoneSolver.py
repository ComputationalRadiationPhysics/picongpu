"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, computed_field
from ..rendering import RenderedObject


class NoneSolver(RenderedObject, BaseModel):
    """
    None solver as defined by PIConGPU

    Disables the vacuum update of E and B (see maxwellSolver::None in
    include/picongpu/fields/MaxwellSolver/None/None.def).

    note: has no parameters
    """

    @computed_field
    def name(self) -> str:
        return "None"
