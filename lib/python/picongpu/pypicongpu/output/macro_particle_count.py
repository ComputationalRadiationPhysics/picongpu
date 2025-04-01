"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
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

    def __init__(self):
        "do nothing"

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(),
        }
