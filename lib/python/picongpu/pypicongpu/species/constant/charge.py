"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
License: GPLv3+
"""

from .constant import Constant
from ... import util
from typeguard import typechecked
import typing


@typechecked
class Charge(Constant):
    """
    charge of a physical particle
    """

    charge_si = util.build_typesafe_property(float)
    """charge in C of an individual particle"""

    def __init__(self):
        # overwrite from parent
        pass

    def check(self) -> None:
        pass

    def _get_serialized(self) -> dict:
        # (please resist the temptation of removing the check b/c "its not
        # needed here": checks should *always* be run before serialization,
        # so make it a habit of expecting it everywhere)
        self.check()
        return {
            "charge_si": self.charge_si,
        }

    def get_species_dependencies(self):
        return []

    def get_attribute_dependencies(self) -> typing.List[type]:
        return []

    def get_constant_dependencies(self) -> typing.List[type]:
        return []
