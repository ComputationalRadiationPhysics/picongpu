"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .densityprofile import DensityProfile
from .... import util
from typeguard import typechecked


@typechecked
class Uniform(DensityProfile):
    """
    globally constant density

    PIConGPU equivalent is the homogenous profile, but due to spelling
    ambiguities the PICMI name uniform is followed here.
    """

    density_si = util.build_typesafe_property(float)
    """density at every point in space (kg * m^-3)"""

    def __init__(self):
        # (nothing to do, overwrite from abstract parent)
        pass

    def check(self) -> None:
        if self.density_si <= 0:
            raise ValueError("density must be >0")

    def _get_serialized(self) -> dict:
        self.check()

        return {
            "density_si": self.density_si,
        }
