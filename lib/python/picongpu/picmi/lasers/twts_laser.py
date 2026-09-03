"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Alexander Debus
License: GPLv3+
"""

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field, computed_field, model_validator

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser, PositiveFloat
from .polarization_type import PolarizationType

from .. import constants


@default_converts_to(laser.TWTSLaser)
class TWTSLaser(BaseModel, BaseLaser):
    """
    Specifies a Traveling-Wave Thomson Scattering (TWTS) laser

    Parameters
    ----------
    wavelength: float
        Central wavelength of the laser [m], must be > 0
    waist: float
        Spot size (1/e^2 radius) of the laser at focus [m], must be > 0
    duration: float
        Duration of the TWTS pulse [s], must be > 0
    laserIncidenceAngle: float
        Laser incidence angle [rad]
    polarizationAngle: float
        Linear laser polarization direction as rotation angle [rad]
    focal_position: list[float]
        3D coordinates of the laser focus [m]
    centroid_position: list[float]
        3D coordinates of the initial laser centroid [m]
    focus_lateral_offset_si: float
        Offset from the middle of the simulation domain to the laser focus [m]
    a0: float, optional
        Normalized vector potential at focus. Specify either a0 or E0.
    E0: float, optional
        Peak electric field amplitude [V/m]. Specify either a0 or E0.
    beta0: float, default 1.0
        Laser centroid speed normalized to speed of light, must be > 0
    windowStart: float, default 0.0
    windowEnd: float, default 0.0
    windowLength: float, default 0.0
    """

    wavelength: PositiveFloat
    waist: PositiveFloat
    duration: PositiveFloat
    laserIncidenceAngle: float
    polarizationAngle: float
    focal_position: Sequence[float]
    centroid_position: Sequence[float]
    focus_lateral_offset_si: float = 0.0
    a0: float | None = None
    E0: float | None = None
    beta0: PositiveFloat = 1.0
    windowStart: float = 0.0
    windowEnd: float = 0.0
    windowLength: float = 0.0

    picongpu_huygens_surface_positions: list[list[int]] = Field(
        default_factory=lambda: [[16, -16], [16, -16], [16, -16]]
    )
    picongpu_polarization_type: PolarizationType = PolarizationType.LINEAR

    @computed_field
    def pulse_init(self) -> float:
        return self._compute_pulse_init()

    @computed_field
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @computed_field
    def phi0(self) -> float:
        # TWTS has no carrier phase input; the phase is always zero.
        return 0.0

    @computed_field
    def propagation_direction(self) -> list[float]:
        return [0.0, math.cos(self.laserIncidenceAngle), math.sin(self.laserIncidenceAngle)]

    @computed_field
    def polarization_direction(self) -> list[float]:
        return [
            math.cos(self.polarizationAngle),
            -math.sin(self.polarizationAngle) * math.sin(self.laserIncidenceAngle),
            math.cos(self.polarizationAngle) * math.cos(self.laserIncidenceAngle),
        ]

    @computed_field
    def laserIncidenceAnglePositive(self) -> bool:
        return self.laserIncidenceAngle > 0

    @computed_field
    def time_offset_si(self) -> float:
        return (self.focal_position[1] - self.centroid_position[1]) / (self.beta0 * constants.c)

    @model_validator(mode="after")
    def _validate(self):
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, self.E0, self.a0)
        self._validate_common_properties()
        return self
