"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from .. import util
from ..species import Species
from .timestepspec import TimeStepSpec

from .plugin import Plugin

import typeguard
import typing


@typeguard.typechecked
class EnergyHistogram(Plugin):
    species = util.build_typesafe_property(Species)
    period = util.build_typesafe_property(TimeStepSpec)
    bin_count = util.build_typesafe_property(int)
    min_energy = util.build_typesafe_property(float)
    max_energy = util.build_typesafe_property(float)

    _name = "energyhistogram"

    def __init__(self):
        "do nothing"

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(),
            "bin_count": self.bin_count,
            "min_energy": self.min_energy,
            "max_energy": self.max_energy,
        }
