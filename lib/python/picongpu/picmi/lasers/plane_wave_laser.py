"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated, Sequence

import math

from pydantic import BaseModel, BeforeValidator, Field, computed_field, model_validator

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser
from .polarization_type import PolarizationType

PositiveFloat = Annotated[
    float,
    BeforeValidator(lambda v: float(v) if (float(v) > 0) else (_ for _ in ()).throw(ValueError("value must be > 0"))),
]


@default_converts_to(
    laser.PlaneWaveLaser,
    conversions={
        "focal_position": "focus_pos",
        "laser_nofocus_constant_si": lambda self: 0.0,
    },
)
class PlaneWaveLaser(BaseModel, BaseLaser):
    """
    Specifies a plane wave with a temporal shape

    Parameters
    ----------
    wavelength: float
        Laser wavelength [m], must be > 0
    duration: float
        Duration of the Gaussian pulse [s], must be > 0
    propagation_direction: unit vector of length 3 of floats
        Direction of propagation [1]
    polarization_direction: unit vector of length 3 of floats
        Direction of polarization [1]
    centroid_position: vector of length 3 of floats
        Position of the laser centroid at time 0 [m]
    a0: float
        Normalized vector potential at focus. Specify either a0 or E0.
    E0: float
        Maximum amplitude of the laser field [V/m]. Specify either a0 or E0.
    phi0: float
        Carrier envelope phase (CEP) [rad]
    """

    wavelength: PositiveFloat
    duration: PositiveFloat
    propagation_direction: Sequence[float]
    polarization_direction: Sequence[float]
    centroid_position: Sequence[float]
    a0: float | None = None
    E0: float | None = None
    phi0: float = 0.0

    picongpu_polarization_type: PolarizationType = PolarizationType.LINEAR
    polarization_type: PolarizationType = Field(default=PolarizationType.LINEAR, exclude=True)
    picongpu_plateau_duration: float = 0.0
    picongpu_huygens_surface_positions: list[list[int]] = [
        [16, -16],
        [16, -16],
        [16, -16],
    ]

    k0: float = 0.0
    focus_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @computed_field
    def pulse_init(self) -> float:
        return self._compute_pulse_init()

    @model_validator(mode="after")
    def _validate(self):
        self.k0 = 2.0 * math.pi / self.wavelength
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, self.E0, self.a0)
        self.focus_pos = [0.0, 0.0, 0.0]
        self.polarization_type = self.picongpu_polarization_type
        self._validate_common_properties()
        return self

    def check(self):
        self._validate_common_properties()
