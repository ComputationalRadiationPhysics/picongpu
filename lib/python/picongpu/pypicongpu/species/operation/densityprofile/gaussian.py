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
    gaussian density profile

    density( y < gasCenterFront )                   =
        density_si * exp(gasFactor * (abs( (y - gasCenterFront) / gasSigmaFront))^gasPower)
    density( gasCenterFront >= y >= gasCenterRear ) = density_si
    density( gasCenterRear < y )                    =
        density_si * exp(gasFactor * (abs( (y - gasCenterRear) / gasSigmaRear))^gasPower)
    """

    gasCenterFront = util.build_typesafe_property(float)
    """position of the front edge of the constant of middle the density profile in meter"""

    gasCenterRear = util.build_typesafe_property(float)
    """position of the rear edge of the constant of middle the density profile in meter"""

    gasSigmaFront = util.build_typesafe_property(float)
    """distance in away from gasCenterFront until the gas density decreases to its 1/e-th part in meter"""

    gasSigmaRear = util.build_typesafe_property(float)
    """The distance from gasCenterRear until the gas density decreases to its 1/e-th part in meter"""

    gasFactor = util.build_typesafe_property(float)
    """exponential scaling factor, see formula above"""

    gasPower = util.build_typesafe_property(float)
    """power-exponent in exponent of density function"""

    vacuumCellsFront = util.build_typesafe_property(int)
    """number of vacuum cells in front of foil for laser init"""

    density_si = util.build_typesafe_property(float)
    """particle number density in m^-3"""

    def __init__(self):
        # (nothing to do, overwrite from abstract parent)
        pass

    def check(self) -> None:
        if self.density_si <= 0:
            raise ValueError("density must be > 0")
        if self.gasCenterFront < 0:
            raise ValueError("gasCenterFront must be >= 0")
        if self.gasCenterRear < 0:
            raise ValueError("gasCenterFront must be >= 0")
        if self.gasCenterRear < self.gasCenterFront:
            raise ValueError("gasCenterRear must be >= gasCenterFront")
        if self.gasCenterRear < self.gasCenterFront:
            raise ValueError("gasCenterRear must be >= gasCenterFront")
        if self.gasFactor <= 0:
            raise ValueError("gasFactor must be > 0")
        if self.gasPower <= 0:
            raise ValueError("gasPower must be > 0")
        if self.vacuumCellsFront < 0:
            raise ValueError("vacuumCellsFront must be > 0")
        if self.density_si <= 0:
            raise ValueError("density_si must be > 0")

    def _get_serialized(self) -> dict:
        self.check()

        return {
            "gasCenterFront": self.gasCenterFront,
            "gasCenterRear": self.gasCenterRear,
            "gasSigmaFront": self.gasSigmaFront,
            "gasSigmaRear": self.gasSigmaRear,
            "gasFactor": self.gasFactor,
            "gasPower": self.gasPower,
            "vacuumCellsFront": self.vacuumCellsFront,
            "density_si": self.density_si,
        }
