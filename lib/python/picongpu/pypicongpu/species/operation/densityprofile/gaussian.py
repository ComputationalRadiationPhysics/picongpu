"""
This file is part of the PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre, Afshari Masoud
License: GPLv3+
"""

from .densityprofile import DensityProfile
from .... import util
from typeguard import typechecked


@typechecked
class Gaussian(DensityProfile):
    """
    Directional guassian density profile
    """

    gasCenterLeft = util.build_typesafe_property(float)
    """  """
    gasCenterRight = util.build_typesafe_property(float)

    gasSigmaLeft = util.build_typesafe_property(float)
    """the distance from gasCenterLeft until the gas density decreases to its 1/e-th part in meter"""

    gasSigmaRight = util.build_typesafe_property(float)
    """the distance from gasCenterRight until the gas density decreases to its 1/e-th part in meter"""

    gasFactor = util.build_typesafe_property(float)

    gasPower = util.build_typesafe_property(float)

    vacuumCellsY = util.build_typesafe_property(int)

    def __init__(self):
        # (nothing to do, overwrite from abstract parent)
        pass

    def check(self) -> None:
        "test"

    def _get_serialized(self) -> dict:
        self.check()

        return {
            "gasCenterLeft": self.gasCenterLeft,
            "gasCenterRight": self.gasCenterRight,
            "gasSigmaLeft": self.gasSigmaLeft,
            "gasSigmaRight": self.gasSigmaRight,
            "gasFactor": self.gasFactor,
            "gasPower" : self.gasPower,
            "vacuumCellsY" : self.vacuumCellsY
        }
