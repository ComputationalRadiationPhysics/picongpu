"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .constant import Constant
from ..attribute import BoundElectrons
from .elementproperties import ElementProperties

import typeguard
import typing


@typeguard.typechecked
class Ionizers(Constant):
    """
    ionizers describing the ionization methods

    Currently the selected ionizers are fixed by the code generation.
    When they are selectable by the user, they can be added here.
    """

    # note: no typecheck here -- which would require circular imports
    electron_species = None
    """species to be used as electrons"""

    def __init__(self):
        # overwrite from parent
        pass

    def check(self) -> None:
        # import here to avoid circular import
        from ..species import Species

        if not isinstance(self.electron_species, Species):
            raise TypeError("electron_species must be of type pypicongpu Species")

        # electron species must not be ionizable
        if self.electron_species.has_constant_of_type(Ionizers):
            raise ValueError("used electron species {} must not be ionizable itself".format(self.electron_species.name))

        # note: do **NOT** check() electron species here
        # -> it is not fully initialized at this point in the initialization
        # (check requires attributes which are added last,
        # but constants are added first)

    def _get_serialized(self) -> dict:
        # (please resist the temptation of removing the check b/c "its not
        # needed here": checks should *always* be run before serialization,
        # so make it a habit of expecting it everywhere)
        self.check()
        return {
            "electron_species": self.electron_species.get_rendering_context(),
        }

    def get_species_dependencies(self):
        self.check()
        return [self.electron_species]

    def get_attribute_dependencies(self) -> typing.List[type]:
        return [BoundElectrons]

    def get_constant_dependencies(self) -> typing.List[type]:
        return [ElementProperties]
