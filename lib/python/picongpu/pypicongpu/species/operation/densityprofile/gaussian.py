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

    gas_center_front = util.build_typesafe_property(float)
    """position of the front edge of the constant of middle the density profile in meter"""

    gas_center_rear = util.build_typesafe_property(float)
    """position of the rear edge of the constant of middle the density profile in meter"""

    gas_sigma_front = util.build_typesafe_property(float)
    """distance in away from gasCenterFront until the gas density decreases to its 1/e-th part in meter"""

    gas_sigma_rear = util.build_typesafe_property(float)
    """The distance from gasCenterRear until the gas density decreases to its 1/e-th part in meter"""

    gas_factor = util.build_typesafe_property(float)
    """exponential scaling factor, see formula above"""

    gas_power = util.build_typesafe_property(float)
    """power-exponent in exponent of density function"""

    vacuum_cells_front = util.build_typesafe_property(int)
    """number of vacuum cells in front of foil for laser init"""

    density = util.build_typesafe_property(float)
    """particle number density in m^-3"""

    def __init__(self):
        # (nothing to do, overwrite from abstract parent)
        pass

    def check(self) -> None:
        if self.density_si <= 0:
            raise ValueError("density must be > 0")
        if self.gas_center_front < 0:
            raise ValueError("gasCenterFront must be >= 0")
        if self.gas_center_rear < 0:
            raise ValueError("gasCenterFront must be >= 0")
        if self.gas_center_rear < self.gasCenterFront:
            raise ValueError("gasCenterRear must be >= gasCenterFront")
        if self.gas_factor <= 0:
            raise ValueError("gasFactor must be > 0")
        if self.gas_power <= 0:
            raise ValueError("gasPower must be > 0")
        if self.vacuum_cells_front < 0:
            raise ValueError("vacuumCellsFront must be > 0")
        if self.density <= 0:
            raise ValueError("density_si must be > 0")

    def _get_serialized(self) -> dict:
        self.check()

        return {
            "gas_center_front": self.gasCenterFront,
            "gas_center_rear": self.gasCenterRear,
            "gas_sigma_front": self.gasSigmaFront,
            "gas_sigma_rear": self.gasSigmaRear,
            "gas_factor": self.gasFactor,
            "gas_power": self.gasPower,
            "vacuum_cells_front": self.vacuumCellsFront,
            "density": self.density_si,
        }
