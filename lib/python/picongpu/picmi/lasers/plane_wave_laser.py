"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import math
import typing

import typeguard

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser
from .polarization_type import PolarizationType


@default_converts_to(laser.PlaneWaveLaser)
@typeguard.typechecked
class PlaneWaveLaser(BaseLaser):
    """
    Specifies a plane wave with a temporal shape

    Parameters
    ----------
    wavelength: float
        Laser wavelength [m], defined as :math:`\\lambda_0` in the above formula

    duration: float
        Duration of the Gaussian pulse [s], defined as :math:`\\tau` in the above formula

    propagation_direction: unit vector of length 3 of floats
        Direction of propagation [1]

    polarization_direction: unit vector of length 3 of floats
        Direction of polarization [1]

    centroid_position: vector of length 3 of floats
        Position of the laser centroid at time 0 [m]

    a0: float
        Normalized vector potential at focus
        Specify either a0 or E0 (E0 takes precedence).

    E0: float
        Maximum amplitude of the laser field [V/m]
        Specify either a0 or E0 (E0 takes precedence).

    phi0: float
        Carrier envelope phase (CEP) [rad]
    """

    def __init__(
        self,
        wavelength,
        duration,
        propagation_direction,
        polarization_direction,
        centroid_position,
        a0=None,
        E0=None,
        phi0: float = 0.0,
        picongpu_polarization_type=(PolarizationType.LINEAR),
        picongpu_plateau_duration=0.0,
        # make sure to always place Huygens-surface inside PML-boundaries,
        # default is valid for standard PMLs
        # @todo create check for insufficient dimension
        # @todo create check in simulation for conflict between PMLs and
        # Huygens-surfaces
        picongpu_huygens_surface_positions: typing.List[typing.List[int]] = [
            [16, -16],
            [16, -16],
            [16, -16],
        ],
    ):
        if wavelength <= 0:
            raise ValueError(f"wavelength must be > 0. You gave {wavelength=}.")
        if duration <= 0:
            raise ValueError(f"laser pulse duration must be > 0. You gave {duration=}.")

        self.wavelength = wavelength
        self.k0 = 2.0 * math.pi / wavelength
        self.duration = duration
        self.centroid_position = centroid_position
        self.propagation_direction = propagation_direction
        self.polarization_direction = polarization_direction
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, E0, a0)
        self.phi0 = phi0

        self.picongpu_plateau_duration = picongpu_plateau_duration
        self.picongpu_polarization_type = picongpu_polarization_type
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions
        self.focus_pos = [0.0, 0.0, 0.0]
        self._validate_common_properties()
        self.pulse_init = self._compute_pulse_init()

    """PICMI object for Plane Wave Laser"""

    def check(self):
        self._validate_common_properties()
