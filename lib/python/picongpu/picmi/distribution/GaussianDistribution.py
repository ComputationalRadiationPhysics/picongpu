"""
This file is part of the PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ...pypicongpu import species
from ...pypicongpu import util

from .Distribution import Distribution

import typeguard
import typing


@typeguard.typechecked
class GaussianDistribution(Distribution):
    """
    Describes a density distribution of particles with gaussian up- and down-ramps with a constant density region in
    between

    Will create the following profile:
    - for y < center_front:                density * exp(factor * abs(y - center_front/sigma_front)**power)
    - for center_front <= y <=center_rear: density
    - for y > center_rear:                 density * exp(factor * abs(y - center_rear/sigma_rear)**power)

    with y being the position in the simulation box
    """

    density: float
    """density particle number density, [m^-3]"""

    center_front: float
    """center_front center of gaussian ramp at the front, [m]"""
    center_rear: float
    """center_rear center of the gaussian ramp at the rear, [m]"""

    sigma_front: float
    """sigma of the gaussian ramp at the front, [m]"""
    sigma_rear: float
    """sigma of the gaussian ramp at the front, [m]"""

    power: float
    """power used in exponential function, 2 will yield a gaussian, 4+ a super-gaussian, unitless"""
    factor: float
    """sign and scaling factor, must be < 0, unitless"""

    vacuum_cells_front: int
    """number of cells to keep as vacuum in front of density for laser init and similar, unitless"""

    lower_bound: typing.Tuple[float, float, float] | typing.Tuple[None, None, None] = (None, None, None)
    upper_bound: typing.Tuple[float, float, float] | typing.Tuple[None, None, None] = (None, None, None)

    # @details pydantic provides an automatically generated __init__/constructor method which allows initialization off
    #   all attributes as keyword arguments

    # @note user may add additional attributes by hand, these will be available but not type verified

    def get_as_pypicongpu(self) -> species.operation.densityprofile.DensityProfile:
        util.unsupported("fill in not active", self.fill_in, True)

        # @todo support bounds, Brian Marre, 2024
        util.unsupported("lower bound", self.lower_bound, [None, None, None])
        util.unsupported("upper bound", self.upper_bound, [None, None, None])

        gaussian_profile = species.operation.densityprofile.Gaussian()

        if self.center_rear < self.center_front:
            raise ValueError("center_front must be <= center_rear")
        if self.density <= 0.0:
            raise ValueError("density must be > 0")

        # @todo change to constructor call once we switched PyPIConGPU to use pydantic, Brian Marre, 2024
        gaussian_profile.gas_center_front = self.center_front
        gaussian_profile.gas_center_rear = self.center_rear
        gaussian_profile.gas_sigma_front = self.sigma_front
        gaussian_profile.gas_sigma_rear = self.sigma_rear
        gaussian_profile.gas_factor = self.factor
        gaussian_profile.gas_power = self.power
        gaussian_profile.vacuum_cells_front = self.vacuum_cells_front
        gaussian_profile.density = self.density

        return gaussian_profile
