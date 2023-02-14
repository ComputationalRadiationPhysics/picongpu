"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus
License: GPLv3+
"""

from ..pypicongpu import util, laser
from . import constants

import picmistandard

from typeguard import typechecked
import typing


@typechecked
class GaussianLaser(picmistandard.PICMI_GaussianLaser):
    """PICMI object for Gaussian Laser"""

    def scalarProduct(self, a: typing.List[float], b: typing.List[float] ) -> float:
        assert len(a) == len(b), "the scalar product is only defined for two vector of equal dimension"

        result = 0.;
        for i in range(len(a)):
            result += a[i] * b[i]

        return result

    def __init__(
            self, wavelength, waist, duration, propagation_direction,
            polarization_direction, focal_position, centroid_position,

            picongpu_polarization_type = laser.GaussianLaser.PolarizationType.LINEAR,
            picongpu_laguerre_modes: typing.Optional[
                typing.List[float]] = None,
            picongpu_laguerre_phases: typing.Optional[
                typing.List[float]] = None,
            picongpu_phase: float = 0.,
            # make sure to always place Huygens-surface inside PML-boundaries,
            #   default is valid for standard PMLs
            #   @todo create check for insufficient dimension
            #   @todo create check in simulation for conflict between PMLs and Huygens-surfaces
            picongpu_huygens_surface_positions: \
                typing.List[typing.List[int]] = [[16, -16], [16, -16], [16, -16]],
            **kw):

        if waist <= 0:
            raise ValueError("waist must be > 0")
        if wavelength <= 0:
            raise ValueError("wavelength must be > 0")
        if duration <= 0:
            raise ValueError("laser pulse duration must be > 0")

        assert (picongpu_laguerre_modes is None
                and picongpu_laguerre_phases is None) or \
               (picongpu_laguerre_modes is not None
                and picongpu_laguerre_phases is not None), \
               "laguerre_modes and laguerre_phases MUST BE both set or both unset"

        self.picongpu_polarization_type = picongpu_polarization_type
        self.picongpu_laguerre_modes = picongpu_laguerre_modes
        self.picongpu_laguerre_phases = picongpu_laguerre_phases
        self.picongpu_phase = picongpu_phase
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions

        super().__init__(wavelength, waist, duration, propagation_direction,
            polarization_direction, focal_position, centroid_position, **kw)

    def get_as_pypicongpu(self) -> laser.GaussianLaser:
        util.unsupported("laser name", self.name)
        util.unsupported("laser zeta", self.zeta)
        util.unsupported("laser beta", self.beta)
        util.unsupported("laser phi2", self.phi2)
        # unsupported: fill_in (do not warn, b/c we don't know if it has been
        # set explicitly, and always warning is bad)

        assert abs(self.scalarProduct(self.polarization_direction, self.polarization_direction) - 1) < 1e-9, \
            "the polarization direction vector must be normalized"

        # check for excessive phase values to avoid numerical precision problems
        assert abs(self.picongpu_phase) <= 2*3.1416, "abs(phase) must be < 2*pi"

        # check that initialising from y_min-plane only is sensible
        assert (self.scalarProduct(self.propagation_direction, [0., 1., 0.]) > 0.), \
            "laser propagation parallel to the y-plane or pointing outside from the inside \
            of the simulation box is not supported by this laser in picongpu"
        assert ((self.propagation_direction[0]**2
              + self.propagation_direction[1]**2
              + self.propagation_direction[2]**2) - 1. < 1e-9), "propagation vector must be normalized"

        # check centroid outside box
        assert (self.centroid_position[1] <= 0), "the laser maximum must be outside of the \
            simulation box, otherwise it is impossible to correctly initialize it \
            using a huygens surface  in the box, centroid_y < 0"
        # @todo implement check that laser field strength sufficiently small at simulation
        #   box boundary

        # check polarization vector normalization
        assert ((self.propagation_direction[0]**2
              + self.propagation_direction[1]**2
              + self.propagation_direction[2]**2) - 1. < 1e-9), "polarization vector must be normalized"

        pypicongpu_laser = laser.GaussianLaser()
        pypicongpu_laser.wavelength = self.wavelength
        pypicongpu_laser.waist = self.waist
        pypicongpu_laser.duration = self.duration
        pypicongpu_laser.focus_pos = self.focal_position
        pypicongpu_laser.phase = self.picongpu_phase
        pypicongpu_laser.E0 = self.E0

        pypicongpu_laser.pulse_init = max(
            -2*self.centroid_position[1]/
            (self.propagation_direction[1] * constants.c)/self.duration, 15)
            # unit: duration

        pypicongpu_laser.polarization_type = self.picongpu_polarization_type
        pypicongpu_laser.polarization_direction = self.polarization_direction

        pypicongpu_laser.propagation_direction = self.propagation_direction

        if self.picongpu_laguerre_modes is None:
            pypicongpu_laser.laguerre_modes = [1.0]
        else:
            pypicongpu_laser.laguerre_modes = self.picongpu_laguerre_modes

        if self.picongpu_laguerre_phases is None:
            pypicongpu_laser.laguerre_phases = [0.0]
        else:
            pypicongpu_laser.laguerre_phases = self.picongpu_laguerre_phases

        pypicongpu_laser.huygens_surface_positions = self.picongpu_huygens_surface_positions

        return pypicongpu_laser
