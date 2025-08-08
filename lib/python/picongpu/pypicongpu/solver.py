"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .rendering import RenderedObject

import typeguard


@typeguard.typechecked
class Solver:
    """
    represents a field solver

    Parent class for type safety, does not contain anything.
    """

    pass


@typeguard.typechecked
class YeeSolver(Solver, RenderedObject):
    """
    Yee solver as defined by PIConGPU

    note: has no parameters
    """

    def _get_serialized(self) -> dict:
        return {
            "name": "Yee",
        }


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
