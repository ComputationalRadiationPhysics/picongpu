"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ..rendering import RenderedObject
from .DefaultSolver import Solver

import typeguard


@typeguard.typechecked
class LeheSolver(Solver, RenderedObject):
    """
    Lehe solver as defined by PIConGPU

    note: has no parameters
    """

    def _get_serialized(self) -> dict:
        # @todo + "<>" needs to be fixed later
        return {
            "name": "Lehe" + "<>",
        }
