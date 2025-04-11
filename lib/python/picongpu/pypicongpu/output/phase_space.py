"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from .. import util
from ..species import Species

from .plugin import Plugin
from .timestepspec import TimeStepSpec

import typeguard
import typing
from typing import Literal


@typeguard.typechecked
class PhaseSpace(Plugin):
    species = util.build_typesafe_property(Species)
    period = util.build_typesafe_property(TimeStepSpec)
    spatial_coordinate = util.build_typesafe_property(Literal["x", "y", "z"])
    momentum_coordinate = util.build_typesafe_property(Literal["px", "py", "pz"])
    min_momentum = util.build_typesafe_property(float)
    max_momentum = util.build_typesafe_property(float)

    _name = "phasespace"

    def __init__(self):
        "do nothing"

    def check(self):
        if self.min_momentum >= self.max_momentum:
            raise ValueError(
                "PhaseSpace's min_momentum should be smaller than max_momentum. "
                f"You gave: {self.min_momentum=} and {self.max_momentum=}."
            )

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""
        self.check()
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(),
            "spatial_coordinate": self.spatial_coordinate,
            "momentum_coordinate": self.momentum_coordinate,
            "min_momentum": self.min_momentum,
            "max_momentum": self.max_momentum,
        }
