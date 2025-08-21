"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from .. import util
from ..species import Species
from .timestepspec import TimeStepSpec
from .plugin import Plugin

import typeguard
import typing
import warnings


@typeguard.typechecked
class PhaseSpace(Plugin):
    """
    Phase Space output plugin for PIConGPU.

    Extracts phase-space data for a given species, spatial coordinate, and momentum coordinate.
    """

    species = util.build_typesafe_property(Species)
    period = util.build_typesafe_property(TimeStepSpec)
    spatial_coordinate = util.build_typesafe_property(typing.Literal["x", "y", "z"])
    momentum_coordinate = util.build_typesafe_property(typing.Literal["px", "py", "pz"])
    min_momentum = util.build_typesafe_property(float)
    max_momentum = util.build_typesafe_property(float)

    _name = "phasespace"

    def __init__(self):
        """Do nothing"""
        pass

    def check(self):
        """Validate attributes."""
        if self.min_momentum >= self.max_momentum:
            raise ValueError("min_momentum must be less than max_momentum")

    def _get_serialized(self) -> dict:
        """Return the serialized representation of the object."""
        self.check()
        if not self.period.get_rendering_context(200).get("specs", []):
            warnings.warn("PhaseSpace is disabled because period is empty")
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(200),
            "spatial_coordinate": self.spatial_coordinate,
            "momentum_coordinate": self.momentum_coordinate,
            "min_momentum": self.min_momentum,
            "max_momentum": self.max_momentum,
        }
