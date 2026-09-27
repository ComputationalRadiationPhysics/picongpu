"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch,
         Masoud Afshari
License: GPLv3+
"""

from typing import Annotated

import numpy as np
from picmistandard import PICMI_GaussianLaser
from pydantic import Field, computed_field, model_validator
from scipy.spatial.transform import Rotation

from ...pypicongpu import laser, util
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser
from .polarization_type import PolarizationType


@default_converts_to(
    laser.GaussianLaser,
    # PICMI's `duration` is the standard 1/e field width (tau), while PIConGPU's
    # `pulse_duration_si` (aliased as `duration`) is the 1 sigma of the intensity,
    # i.e. PULSE_DURATION = duration / 2 (#5739)
    conversions={"duration": lambda self, *args, **kwargs: self._pulse_duration_sigma_si()},
)
class GaussianLaser(PICMI_GaussianLaser, BaseLaser):
    """
    PICMI object for Gaussian Laser.

    Standard Gaussian laser pulse parameters are:

    - wavelength : float
        Central wavelength of the laser [m].

    - waist : float
        Spot size (1/e^2 radius of the intensity) of the laser at focus [m].

    - duration : float
        Duration of the Gaussian laser pulse [s], defined as ``tau`` in the
        electric-field envelope ``E ~ exp(-t^2 / tau^2)`` (i.e. the 1/e half
        width of the field amplitude), consistent with the PICMI standard.

    - propagation_direction : list[float]
        Normalized vector of propagation direction.

    - polarization_direction : list[float]
        Normalized vector of polarization direction.

    - focal_position : list[float]
        3D coordinates of the laser focus [m].

    - centroid_position : list[float]
        3D coordinates of the laser centroid [m].

    - a0 : float, optional
        Normalized vector potential (dimensionless).

    - E0 : float, optional
        Peak electric field amplitude [V/m].

    - picongpu_polarization_type: Polarization type in PIConGPU (LINEAR or CIRCULAR)

    - picongpu_laguerre_modes: Optional magnitudes of Laguerre modes (only relevant for structured beams)

    - picongpu_laguerre_phases: Optional phases of Laguerre modes (only relevant for structured beams)

    - picongpu_huygens_surface_positions : list[list[int]], default=[[16, -16],[16, -16],[16, -16]]
        Positions of the Huygens surface inside the PML. Each entry is a
        pair [min, max] indices along x, y, z.

    - phi0 : float, optional
    Initial phase offset [rad].

    Notes:
    - Exactly one of ``a0`` or ``E0`` must be provided, the other is
      calculated automatically.
    """

    picongpu_polarization_type: PolarizationType = PolarizationType.LINEAR
    picongpu_laguerre_modes: list[float] = Field(default_factory=lambda: [1.0])
    picongpu_laguerre_phases: list[float] = Field(default_factory=lambda: [0.0])
    # make sure to always place Huygens-surface inside PML-boundaries,
    # default is valid for standard PMLs
    # @todo create check for insufficient dimension
    # @todo create check in simulation for conflict between PMLs and
    # Huygens-surfaces
    picongpu_huygens_surface_positions: list[list[int]] = Field(
        default_factory=lambda: [[16, -16], [16, -16], [16, -16]]
    )
    phi0: float = 0.0

    # PICMI-standard laser options that PIConGPU does not implement are
    # rejected at construction time.
    name: Annotated[str | None, util.rejects_unsupported("laser name")] = None
    zeta: Annotated[float | None, util.rejects_unsupported("laser zeta")] = None
    beta: Annotated[float | None, util.rejects_unsupported("laser beta")] = None
    phi2: Annotated[float | None, util.rejects_unsupported("laser phi2")] = None

    @computed_field
    def pulse_init(self) -> float:
        return self._compute_pulse_init()

    def _Omega0(self):
        from picongpu.picmi.constants import c

        return 2 * np.pi * c / self.wavelength

    def _inverse_curvature_radius(self, z, w0):
        from picongpu.picmi.constants import c

        return z / (z**2 + (0.5 * self._Omega0() * w0**2 / c) ** 2)

    def _pulse_duration_sigma_si(self):
        """Convert the PICMI-standard laser ``duration`` to the PIConGPU
        ``PULSE_DURATION`` parameter.

        The PICMI standard defines the Gaussian temporal envelope as
        ``E ~ exp(-t^2 / duration^2)``, i.e. ``duration`` is the 1/e half width
        of the electric-field amplitude (see ``PICMI_GaussianLaser``).
        PIConGPU's Gaussian temporal envelope is
        ``E ~ exp(-t^2 / (4 * PULSE_DURATION^2))`` (see ``GaussianPulse.hpp``),
        i.e. ``PULSE_DURATION`` is the 1 sigma of the intensity (see
        ``BaseParam.def``; ``DispersivePulse.hpp`` documents
        ``tau_0 = 2 * PULSE_DURATION``). Matching the two envelopes gives
        ``duration = 2 * PULSE_DURATION``, hence ``PULSE_DURATION = duration / 2``.
        """
        return self.duration / 2.0

    @model_validator(mode="after")
    def _validate(self):
        if len(self.picongpu_laguerre_modes) != len(self.picongpu_laguerre_phases):
            raise ValueError(
                "Your setup specifies a different number of Laguerre modes and phases. "
                "Please be explicit about both and use the same length. "
                f"You gave: {self.picongpu_laguerre_modes=} and {self.picongpu_laguerre_phases=}."
            )
        self._validate_common_properties()

        assert self._propagation_connects_centroid_and_focus(), (
            "propagation_direction must connect centroid_position and focus_position"
        )
        return self

    def _complex_amplitude_standard_conditions(self, x, y, z, t):
        """
        This is
        - focus is in the origin at t=0
        - propagation in z direction
        - polarization in x direction
        """

        from picongpu.picmi.constants import c

        w0x = self.waist
        w0y = self.waist
        # Temporal envelope of the C++ GaussianPulse profile is
        # exp(-(t / (2 * PULSE_DURATION))^2) (see GaussianPulse.hpp) with
        # PULSE_DURATION = duration / 2 (the 1 sigma of the intensity), because
        # the PICMI `duration` is the 1/e field width (tau), cf. #5739.  Hence the
        # field decays as exp(-(t / duration)^2), i.e. the effective field duration
        # here is simply
        #     tau0 = duration .
        tau0 = self.duration
        zRx = 0.5 * self._Omega0() * w0x**2 / c
        zRy = 0.5 * self._Omega0() * w0y**2 / c
        Rx_inv = self._inverse_curvature_radius(z, w0x)
        Ry_inv = self._inverse_curvature_radius(z, w0y)
        wx = w0x * np.sqrt(1 + z**2 / zRx**2)
        wy = w0y * np.sqrt(1 + z**2 / zRy**2)
        gamma4 = (t - z / c - 0.5 / c * (x**2 * Rx_inv + y**2 * Ry_inv)) / tau0

        # The field of the DispersivePulse/GaussianPulse profiles is normalized such
        # that its on-axis, in-focus amplitude is AMPLITUDE (= E0).  The 1/(tau0
        # sqrt(pi)) factor appearing in models/lasers.rst belongs to the normalized
        # input spectrum and must *not* be multiplied here.
        return (
            self.E0
            * (1 + z**2 / zRx**2) ** (-1 / 4)
            * (1 + z**2 / zRy**2) ** (-1 / 4)
            * np.exp(1.0j * self._Omega0() * gamma4 * tau0)
            * np.exp(0.5j * (np.arctan(z / zRx) + np.arctan(z / zRy)))
            * np.exp(-(x**2 / wx**2 + y**2 / wy**2))
            * np.exp(-(gamma4**2))
        )

    def _polarization_vector_standard_conditions_at(self, x, y, z, t):
        """
        This is
        - focus is in the origin at t=0
        - propagation in z direction
        - polarization in x direction
        """
        if self.picongpu_polarization_type == PolarizationType.LINEAR:
            w0x = self.waist
            w0y = self.waist
            angle_x = np.arcsin(x * self._inverse_curvature_radius(z, w0x))
            angle_y = np.arcsin(y * self._inverse_curvature_radius(z, w0y))
            r = Rotation.from_euler("xy", np.moveaxis(np.asarray([angle_x, angle_y]), 0, -1))
            return np.moveaxis(r.apply(self.polarization_direction), -1, 0)
        elif self.picongpu_polarization_type == PolarizationType.CIRCULAR:
            raise NotImplementedError("Circular polarization is not yet implemented.")
        else:
            raise ValueError("Unknown {self.picongpu_polarization_type=}.")

    def polarization_vector_at(self, x, y, z, t=0.0):
        return self._polarization_vector_standard_conditions_at(*self._to_standard_coordinates(x, y, z, t))

    def _standard_rotation(self):
        return Rotation.align_vectors(
            [[1, 0, 0], [0, 0, 1]], [self.polarization_direction, self.propagation_direction]
        )[0].apply

    def _to_standard_coordinates(self, x, y, z, t):
        from picongpu.picmi.constants import c

        coords = np.asarray([x, y, z]) - as_broadcastable_factor_in_front_of(self.focal_position, of=(x, y, z))
        r = self._standard_rotation()
        x, y, z = np.moveaxis(r(np.moveaxis(coords, 0, -1)), -1, 0)
        t -= np.dot(np.asarray(self.focal_position) - self.centroid_position, self.propagation_direction) / c
        return x, y, z, t

    def complex_amplitude(self, x, y, z, t=0.0):
        return self._complex_amplitude_standard_conditions(*self._to_standard_coordinates(x, y, z, t))

    def E(self, x, y, z, t=0.0):
        return (
            self.polarization_vector_at(x, y, z, t)
            * np.real(self.complex_amplitude(x, y, z, t)).astype(float)[np.newaxis, ...]
        )

    def Ex(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[0]

    def Ey(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[1]

    def Ez(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[2]

    def envelope(self, x, y, z, t=0.0):
        return np.abs(self.complex_amplitude(x, y, z, t))


def as_broadcastable_factor_in_front_of(arr, of=tuple()):
    if of:
        return np.reshape(arr, (-1, *(1 for _ in np.broadcast_shapes(*map(np.shape, of)))))
    return np.asarray(arr)
