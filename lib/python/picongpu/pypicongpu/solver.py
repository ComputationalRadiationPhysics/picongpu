"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
License: GPLv3+
"""

from typeguard import typechecked
from .rendering import RenderedObject


@typechecked
class Solver:
    """
    represents a field solver

    Parent class for type safety, does not contain anything.
    """
    pass


@typechecked
class YeeSolver(Solver, RenderedObject):
    """
    Yee solver as defined by PIConGPU

    note: has no parameters
    """

    def _get_serialized(self) -> dict:
        return {
            "name": "Yee",
        }
