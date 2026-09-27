"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from collections.abc import Sequence

import math

from picmistandard import PICMI_Laser, resolve_once

import numpy as np

from pydantic import Field, computed_field, model_validator

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser, PositiveFloat
from .polarization_type import PolarizationType


@default_converts_to(
    laser.PlaneWaveLaser,
    conversions={
        "focal_position": lambda *_, **__: [0, 0, 0],
        "laser_nofocus_constant_si": "picongpu_plateau_duration",
    },
)
class PlaneWaveLaser(PICMI_Laser, BaseLaser):
    """
    Specifies a plane wave with a temporal shape

    Parameters
    ----------
    wavelength: float
        Laser wavelength [m], must be > 0
    duration: float
        Duration of the temporal Gaussian pulse [s], must be > 0
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
    picongpu_plateau_duration: float = 0.0
    picongpu_huygens_surface_positions: list[list[int]] = Field(
        default_factory=lambda: [[16, -16], [16, -16], [16, -16]]
    )

    @computed_field
    def pulse_init(self) -> float:
        return self._compute_pulse_init()

    @computed_field
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @computed_field
    def focus_pos(self) -> list[float]:
        return [0.0, 0.0, 0.0]

    @model_validator(mode="after")
    @resolve_once
    def _validate(self):
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, self.E0, self.a0)
        self._validate_common_properties()
        return self

    def check(self):
        self._validate_common_properties()

    def complex_amplitude(self, x, y, z, t=0.0):
        from picongpu.picmi.constants import c

        prop = t / c - np.einsum("i,i...->...", self.propagation_direction, [x, y, z])
        prop_env = prop
        return np.exp(-(prop_env**2) / self.duration**2) * np.exp(-1.0j * self.k0 * prop)

    def E(self, x, y, z, t=0.0):
        return np.real(self.complex_amplitude(x, y, z, t))[..., np.newaxis] * self.polarization_vector_at(x, y, z, t)

    def Ex(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[..., 0]

    def Ey(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[..., 1]

    def Ez(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[..., 2]

    def polarization_vector_at(self, x, y, z, t=0.0):
        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        return np.ones(shape + (len(self.polarization_direction),)) * np.reshape(
            self.polarization_direction, len(shape) * (1,) + (-1,)
        )

