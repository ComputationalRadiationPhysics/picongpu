# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre, Pawel Ordyna
License: GPLv3+
"""

from ...pypicongpu import species
from ...pypicongpu import util
import numpy as np

from .Distribution import Distribution

import typeguard
import math


@typeguard.typechecked
class CylindricalDistribution(Distribution):
    """
    Describes a cylindrical density distribution of particles with gaussian up-ramp
    with a constant density region in between. It can have an arbitrary orientation
    and position in space.

    Will create the following profile:
      n = density if r < reduced_radius
      n is 0 or follows the exponential ramp if r > reduced_radius
      n is 0 if r > reduced_radius + prePlasmaCutoff
      the reduced_radius is equal = @f[\\sqrt{R^2 -L^2} -L @f]
      with R - cylinder_radius and L - prePlasmaLength (scale length of the ramp)
      the reduced radius ensures mass conservation
    """

    density: float
    """particle number density, [m^-3]"""

    center_position: tuple[float, float, float]
    """center of the cylinder [x, y, z], [m]"""

    radius: float
    """cylinder radius, [m]"""

    cylinder_axis: tuple[float, float, float]
    """cylinder axis [x, y, z], [unitless]"""

    exponential_pre_plasma_length: float | None
    """scale length of the exponential pre-plasma ramp, [m]"""
    exponential_pre_plasma_cutoff: float | None
    """cutoff of the exponential pre-plasma ramp, [m]"""

    cell_size: tuple[float, float, float] | None = None

    # @details pydantic provides an automatically generated __init__/constructor method which allows initialization off
    #   all attributes as keyword arguments

    # @note user may add additional attributes by hand, these will be available but not type verified

    def get_as_pypicongpu(self, grid):
        self.cell_size = grid.get_cell_size()
        util.unsupported("fill in not active", self.fill_in, True)

        if self.density <= 0.0:
            raise ValueError("density must be > 0")

        min_radius = (
            math.sqrt(2.0) * self.exponential_pre_plasma_length
            if self.exponential_pre_plasma_length is not None
            else 0.0
        )
        if self.radius < min_radius:
            raise ValueError(
                f"radius must be > sqrt(2)*pre_plasma_length = {min_radius}, so that the reduced radius stays non negative. In case of no preplasma radius must be >= 0.0., {self.exponential_pre_plasma_length}, {self.radius}"
            )

        # create prePlasma ramp if indicated by settings
        prePlasma: bool = (self.exponential_pre_plasma_cutoff is not None) and (
            self.exponential_pre_plasma_length is not None
        )
        explicitlyNoPrePlasma: bool = (self.exponential_pre_plasma_cutoff is None) and (
            self.exponential_pre_plasma_length is None
        )

        if prePlasma:
            pre_plasma_ramp = species.operation.densityprofile.plasmaramp.Exponential(
                PlasmaLength=self.exponential_pre_plasma_length,  # type: ignore
                PlasmaCutoff=self.exponential_pre_plasma_cutoff,  # type: ignore
            )
        elif explicitlyNoPrePlasma:
            pre_plasma_ramp = species.operation.densityprofile.plasmaramp.None_()
        else:
            raise ValueError(
                "either both exponential_pre_plasma_length and"
                " exponential_pre_plasma_cutoff must be set to"
                " none or neither!"
            )

        # @todo change to constructor call once we switched PyPIConGPU to use pydantic, Brian Marre, 2024
        return species.operation.densityprofile.Cylinder(
            density_si=self.density,
            center_position_si=self.center_position,
            radius_si=self.radius,
            cylinder_axis=self.cylinder_axis,
            pre_plasma_ramp=pre_plasma_ramp,
        )

    def __call__(
        self,
        x,
        y,
        z,
    ):
        if self.cell_size is None:
            message = (
                "Due to inconsistencies in the backend, evaluation of this function requires information about the cell_size."
                " You can either set it manually "
                " or you can perform anything that includes writing the input files on your simulation object."
                " This is a temporary workaround and will be fixed in the future."
            )
            raise NotImplementedError(message)

        # The definition of this density uses the origin of the cell
        # while the call operator uses the center.
        x += -0.5 * self.cell_size[0]
        y += -0.5 * self.cell_size[1]
        z += -0.5 * self.cell_size[2]

        # Just for convenience:
        pre_l = self.exponential_pre_plasma_length or 0.0
        pre_c = self.exponential_pre_plasma_cutoff or 0.0

        cylinder_axis = np.array(self.cylinder_axis) / np.linalg.norm(self.cylinder_axis)
        args = (x, y, z)
        positions = np.moveaxis(np.broadcast_arrays(x, y, z), 0, -1)
        r = np.linalg.norm(
            np.cross(
                positions
                - np.reshape(
                    self.center_position,
                    ((len(np.shape(positions)) - 1) * (1,)) + (3,),
                ),
                cylinder_axis,
            ),
            axis=-1,
        )
        radius = np.sqrt(self.radius**2 - pre_l**2) - pre_l
        result = np.zeros(np.broadcast_shapes(*map(np.shape, args)))

        result[r < radius] = 1.0
        mask = (r >= radius) * (r < radius + pre_c)
        result[mask] = np.exp((radius - r) / pre_l)[mask]
        return self.density * result
