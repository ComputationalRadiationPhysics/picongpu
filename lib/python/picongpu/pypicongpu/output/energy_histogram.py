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

    def check(self):
        """Validate attributes."""
        if not hasattr(self, "bin_count") or self.bin_count <= 0:
            raise ValueError(f"bin_count must be positive, got {self.bin_count}")
        if not hasattr(self, "min_energy") or not hasattr(self, "max_energy") or self.min_energy >= self.max_energy:
            raise ValueError(f"min_energy must be less than max_energy, got {self.min_energy} >= {self.max_energy}")

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""
        self.check()
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(),
            "bin_count": self.bin_count,
            "min_energy": self.min_energy,
            "max_energy": self.max_energy,
        }
