"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, computed_field
from ..rendering import RenderedObject


class CKCSolver(RenderedObject, BaseModel):
    """
    CKC solver as defined by PIConGPU

    Uses the extended Cole-Karkkainen-Cowan stencil with better dispersion
    properties (see maxwellSolver::CKC in
    include/picongpu/fields/MaxwellSolver/CKC/CKC.def).

    note: has no parameters
    """

    @computed_field
    def name(self) -> str:
        return "CKC"
