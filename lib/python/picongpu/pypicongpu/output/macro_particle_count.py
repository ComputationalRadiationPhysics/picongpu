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
import warnings


@typeguard.typechecked
class MacroParticleCount(Plugin):
    """
    MacroParticleCount output plugin for PIConGPU.

    Outputs the number of macro-particles for a given species at specified time steps.
    """

    species = util.build_typesafe_property(Species)
    period = util.build_typesafe_property(TimeStepSpec)
    _name = "macroparticlecount"

    def __init__(self):
        """Initialize with no attributes set."""
        pass

    def check(self):
        """Validate attributes."""
        try:
            _ = self.species
        except AttributeError:
            raise ValueError("species must be set") from None
        try:
            _ = self.period
        except AttributeError:
            raise ValueError("period must be set") from None

    def _get_serialized(self) -> dict:
        """Return the serialized representation of the object."""
        self.check()
        if not self.period.get_rendering_context().get("specs", []):
            warnings.warn("MacroParticleCount is disabled because period is empty")
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(),
        }
