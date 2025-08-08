"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from .timestepspec import TimeStepSpec
from .. import util
from ..species import Species

from .plugin import Plugin

import typeguard
import typing


@typeguard.typechecked
class MacroParticleCount(Plugin):
    species = util.build_typesafe_property(Species)
    period = util.build_typesafe_property(TimeStepSpec)

    _name = "macroparticlecount"

    def __init__(self, species: Species, period: TimeStepSpec):
        self.species = species
        self.period = period

    def check(self):
        """Validate attributes."""
        if not self.species:
            raise ValueError("species must be set")
        if not self.period:
            raise ValueError("period must be set")

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""
        self.check()
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(),
        }
