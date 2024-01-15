"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .operation import Operation
from .momentum import Temperature, Drift
from ..species import Species
from ..attribute import Momentum
from ... import util
from typeguard import typechecked
import typing


@typechecked
class SimpleMomentum(Operation):
    """
    provides momentum to a species

    specified by:

    - temperature
    - drift

    Both are optional. If both are missing, momentum **is still provided**, but
    left at 0 (default).
    """

    species = util.build_typesafe_property(Species)
    """species for which momentum will be set"""

    temperature = util.build_typesafe_property(typing.Optional[Temperature])
    """temperature of particles (if any)"""

    drift = util.build_typesafe_property(typing.Optional[Drift])
    """drift of particles (if any)"""

    def __init__(self):
        pass

    def check_preconditions(self) -> None:
        # acces species to make sure it is set -> no required constants
        assert self.species is not None

        if self.temperature is not None:
            self.temperature.check()

        if self.drift is not None:
            self.drift.check()

    def prebook_species_attributes(self) -> None:
        # always provides attribute -- might not be set (i.e. left at 0) though
        self.attributes_by_species = {self.species: [Momentum()]}

    def _get_serialized(self) -> dict:
        self.check_preconditions()

        context = {
            "species": self.species.get_rendering_context(),
            "temperature": None,
            "drift": None,
        }

        if self.temperature is not None:
            context["temperature"] = self.temperature.get_rendering_context()

        if self.drift is not None:
            context["drift"] = self.drift.get_rendering_context()

        return context
