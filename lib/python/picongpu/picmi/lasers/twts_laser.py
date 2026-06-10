"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Alexander Debus
License: GPLv3+
"""

import math

import typeguard

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser
from .polarization_type import PolarizationType

from .. import constants


@default_converts_to(laser.TWTSLaser)
@typeguard.typechecked
class TWTSLaser(BaseLaser):
    """
    Specifies a Traveling-Wave Thomson Scattering (TWTS) laser

    Parameters
    ----------
    - wavelength: float
        Central wavelength of the laser [m].

    - waist : float
        Spot size (1/e^2 radius) of the laser at focus [m].

    - duration: float
        Duration of the TWTS pulse [s], defined as 1 sigma of std. Gauss for intensity.

    - laserIncidenceAngle: float
        Laser incidence angle [rad]
        denoting the mean laser phase propagation direction
        with respect to the y-axis.
        In general, the reference axis (here: y-axis) is defined by
        the focal axis and direction of focus movement.

    - polarizationAngle : float
        Linear laser polarization direction
        parameterized as a rotation angle [rad]
        of the x-axis vector around the mean
        laser phase propagation direction.
        In general, the reference vector (here: x-axis vector) is defined
        to be orthogonal to the laser propagation plane, spanned by
        the focal axis and central laser propagation direction.

    - focal_position : list[float]
        3D coordinates of the laser focus [m]
        in the simulation domain.

    - centroid_position : list[float]
        3D coordinates of the inital laser centroid [m].

    - focus_lateral_offset_si : float, optional [default = 0.0]
        Offset from the middle of the simulation domain
        to the laser focus in z-direction [m].
        In general, the offset direction (here: z-direction) is defined
        to be orthogonal to the focal axis within the central laser propagation plane.

    - a0 : float, optional
        Normalized vector potential at focus [dimensionless].
        Specify either a0 or E0, but not both.

    - E0 : float, optional
        Peak electric field amplitude [V/m].
        Specify either a0 or E0, but not both.

    - beta0 : float, optional [default = 1.0]
        Laser centroid speed in focal axis direction (here: y-direction)
        normalized to the vacuum speed of light [dimensionless].

    - windowStart : float, optional [default = 0.0]
        First time step number [#] at which the laser starts to be gradually switched on using a Blackman-Nuttall window.

    - windowEnd : float, optional [default = 0.0]
        Final time step number [#] after gradually switching off the laser using a Blackman-Nuttall window.
        The default values deactivates the switching functionality, such that the TWTS laser is always present.

    - windowLength : float, optional [default = 0.0]
        Denotes the respective switching duration by half a Blackman-Nuttall window in number of time steps unit [#].

    Description
    -----------
    This field describes an obliquely incident, cylindrically-focused, pulse-front tilted laser for some
    incidence angle as used for [1]. The TWTS implementation is based on the definition of eq. (7) in [1].
    Additionally, techniques from [2] and [3] are used to allow for strictly Maxwell-conform solutions
    for tight foci or small laser incident angles.

    Literature
    ----------
    [1] Steiniger et al., "Optical free-electron lasers with Traveling-Wave Thomson-Scattering",
        Journal of Physics B: Atomic, Molecular and Optical Physics, Volume 47, Number 23 (2014),
        https://doi.org/10.1088/0953-4075/47/23/234011
    [2] Mitri, F. G., "Cylindrical quasi-Gaussian beams", Opt. Lett., 38(22), pp. 4727-4730 (2013),
        https://doi.org/10.1364/OL.38.004727
    [3] Hua, J. F., "High-order corrected fields of ultrashort, tightly focused laser pulses",
        Appl. Phys. Lett. 85, 3705-3707 (2004),
        https://doi.org/10.1063/1.1811384

    Implementation
    --------------
    See source code  picongpu/include/picongpu/fields/background/templates/twtstight/TWTSTight.hpp
    """

    def __init__(
        self,
        wavelength,
        waist,
        duration,
        laserIncidenceAngle,
        polarizationAngle,
        focal_position,
        centroid_position,
        focus_lateral_offset_si,
        a0=None,
        E0=None,
        beta0=1.0,
        windowStart=0.0,
        windowEnd=0.0,
        windowLength=0.0,
        # make sure to always place Huygens-surface inside PML-boundaries,
        # default is valid for standard PMLs
        # @todo create check for insufficient dimension
        # @todo create check in simulation for conflict between PMLs and
        # Huygens-surfaces
        picongpu_huygens_surface_positions: list[list[int]] = [
            [16, -16],
            [16, -16],
            [16, -16],
        ],
    ):
        if wavelength <= 0:
            raise ValueError(f"wavelength must be > 0. You gave {wavelength=}.")
        if duration <= 0:
            raise ValueError(f"laser pulse duration must be > 0. You gave {duration=}.")
        if beta0 <= 0:
            raise ValueError(f"beta0 must be >0. You gave {beta0=}.")

        self.wavelength = wavelength
        self.k0 = 2.0 * math.pi / wavelength
        self.duration = duration
        self.propagation_direction = [0.0, math.cos(laserIncidenceAngle), math.sin(laserIncidenceAngle)]
        self.focal_position = focal_position
        self.centroid_position = centroid_position
        self.polarization_direction = [
            math.cos(polarizationAngle),
            -math.sin(polarizationAngle) * math.sin(laserIncidenceAngle),
            math.cos(polarizationAngle) * math.cos(laserIncidenceAngle),
        ]
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, E0, a0)
        self.laserIncidenceAngle = laserIncidenceAngle
        if laserIncidenceAngle > 0:
            self.laserIncidenceAnglePositive = True
        else:
            self.laserIncidenceAnglePositive = False
        self.beta0 = beta0
        # An additional laser phase phi0 is not yet supported by the TWTS laser.
        self.phi0 = 0.0
        self.waist = waist
        self.polarizationAngle = polarizationAngle
        self.time_offset_si = (focal_position[1] - centroid_position[1]) / (beta0 * constants.c)
        self.focus_lateral_offset_si = focus_lateral_offset_si
        self.windowStart = windowStart
        self.windowEnd = windowEnd
        self.windowLength = windowLength
        self.picongpu_polarization_type = PolarizationType.LINEAR
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions
        self.pulse_init = self._compute_pulse_init()

    """Init PICMI object for TWTS Laser"""
