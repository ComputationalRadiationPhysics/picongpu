"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .constant import Constant
from ... import util

import typeguard
import typing


@typeguard.typechecked
class DensityRatio(Constant):
    """
    factor for weighting when using profiles/deriving
    """

    ratio = util.build_typesafe_property(float)
    """factor for weighting calculation"""

    def __init__(self):
        pass

    def check(self) -> None:
        if self.ratio <= 0:
            raise ValueError("density ratio must be >0")

    def _get_serialized(self) -> dict:
        self.check()
        return {
            "ratio": self.ratio,
        }

    def get_species_dependencies(self):
        return []

    def get_attribute_dependencies(self) -> typing.List[type]:
        return []

    def get_constant_dependencies(self) -> typing.List[type]:
        return []
