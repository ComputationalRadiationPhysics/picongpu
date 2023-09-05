"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .constant import Constant
from ... import util
from typeguard import typechecked
import typing


@typechecked
class Mass(Constant):
    """
    mass of a physical particle
    """

    mass_si = util.build_typesafe_property(float)
    """mass in kg of an individual particle"""

    def __init__(self):
        # overwrite from parent
        pass

    def check(self) -> None:
        if self.mass_si <= 0:
            raise ValueError("mass must be larger than zero")

    def _get_serialized(self) -> dict:
        self.check()
        return {
            "mass_si": self.mass_si
        }

    def get_species_dependencies(self):
        return []

    def get_attribute_dependencies(self) -> typing.List[type]:
        return []

    def get_constant_dependencies(self) -> typing.List[type]:
        return []
