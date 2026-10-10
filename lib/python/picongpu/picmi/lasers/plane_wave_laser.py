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
from .. import constants
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser, PositiveFloat, difference, scalarProduct
from .polarization_type import PolarizationType


@default_converts_to(
    laser.PlaneWaveLaser,
    conversions={
        "focal_position": lambda *_, **__: [0, 0, 0],
        "laser_nofocus_constant_si": "picongpu_plateau_duration",
        # `pulse_init` is a core-only quantity; see _compute_core_pulse_init.
        "pulse_init": lambda self, cell_size=None, domain_cells=None, *args, **kwargs: self._compute_core_pulse_init(
            cell_size, domain_cells
        ),
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
    picongpu_plateau_duration: float
        Length of the central plateau [s]. The pulse maximum is at
        ``centroid_position`` at ``t = 0`` and the plateau spans
        ``[-picongpu_plateau_duration / 2, +picongpu_plateau_duration / 2]``
        around it.

    Notes
    -----
    The pulse is defined in the single, user-visible reference frame of the
    PICMI input and the openPMD output: the maximum is at ``centroid_position``
    at ``t = 0``. The Huygens-surface timing used internally by the PIConGPU
    core is an implementation detail of the translation layer.
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

    def check(self, *args, **kwargs):
        # Extra arguments (the grid info used to convert pulse_init) are accepted
        # and ignored: validation does not depend on the grid.
        self._validate_common_properties()

    def _Omega0(self):
        from picongpu.picmi.constants import c

        return 2.0 * math.pi * c / self.wavelength

    def _compute_core_pulse_init(self, cell_size=None, domain_cells=None):
        """Translate the centroid-based pulse into the core's reference frame.

        The plane wave's frontend model is anchored at the centroid: the sharp
        ramps sit symmetrically around the pulse maximum at ``centroid_position``
        at ``t = 0`` (the plateau, if any, is centred there too). In the core the
        ramps sit at ``pulse_init * sigma`` and ``pulse_init * sigma + plateau``,
        so this method shifts ``pulse_init`` by the beam-axis distance from the
        centroid to the displaced Huygens-surface origin, expressed in pulse
        durations. (The factor 2 in the Gaussian ``_compute_pulse_init`` is
        absent here because the plane-wave ``duration`` is already the intensity
        sigma, cf. ``_pulse_duration_sigma_si``.)
        """
        distance = scalarProduct(
            difference(self.centroid_position, self._huygens_surface_origin_si(cell_size, domain_cells)),
            self.propagation_direction,
        )
        return (-0.5 * self.picongpu_plateau_duration - distance / constants.c) / self._pulse_duration_sigma_si()

    def _standard_rotation(self):
        from scipy.spatial.transform import Rotation

        return Rotation.align_vectors(
            [[1, 0, 0], [0, 0, 1]], [self.polarization_direction, self.propagation_direction]
        )[0].apply

    def _to_standard_coordinates(self, x, y, z, t):
        # The pulse peak (envelope maximum) is at centroid_position at t=0.  The
        # ramps/plateau are anchored at that centroid: the plateau spans
        # [-plateau/2, +plateau/2] around the pulse maximum (see complex_amplitude).
        shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))
        coords = np.asarray([x, y, z]) - np.reshape(self.centroid_position, (-1,) + (1,) * len(shape))
        r = self._standard_rotation()
        x, y, z = np.moveaxis(r(np.moveaxis(coords, 0, -1)), -1, 0)
        # T = t - dot(r - centroid, direction)/c with the peak at the centroid at t=0.
        return x, y, z, t

    def complex_amplitude(self, x, y, z, t=0.0):
        from picongpu.picmi.constants import c

        # PICMI-standard centroid-anchored plane wave: the pulse maximum is at
        # centroid_position at t=0, with the plateau (if any) spanning
        # [-plateau/2, +plateau/2] around it.  In standard conditions (propagation
        # along +z) the longitudinal time argument is T = t - z/c, so the ramps sit
        # at T = -plateau/2 and T = +plateau/2 and the carrier starts at the
        # leading edge.  `duration` here is already the intensity 1-sigma width.
        _, _, z_std, t_std = self._to_standard_coordinates(x, y, z, t)
        T = t_std - z_std / c

        tau = self.duration * math.sqrt(2.0)
        half_plateau = 0.5 * self.picongpu_plateau_duration
        end_up = -half_plateau
        start_down = half_plateau

        envelope = np.full(np.shape(T), self.E0, dtype=np.float64)
        corr = np.zeros_like(envelope)
        up = T < end_up
        down = T > start_down
        envelope = np.where(up, self.E0 * np.exp(-0.5 * ((T - end_up) / tau) ** 2), envelope)
        envelope = np.where(down, self.E0 * np.exp(-0.5 * ((T - start_down) / tau) ** 2), envelope)
        corr = np.where(up, (T - end_up) / (self._Omega0() * tau * tau), corr)
        corr = np.where(down, (T - start_down) / (self._Omega0() * tau * tau), corr)

        t_oszi = T - end_up
        phase = self._Omega0() * t_oszi + self.phi0
        value = (np.sin(phase) + np.cos(phase) * corr) * envelope
        return value.astype(np.complex128)

    def E(self, x, y, z, t=0.0):
        return self.polarization_vector_at(x, y, z, t) * np.real(self.complex_amplitude(x, y, z, t))[np.newaxis, ...]

    def Ex(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[0]

    def Ey(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[1]

    def Ez(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[2]

    def polarization_vector_at(self, x, y, z, t=0.0):
        # constant (wavefront-uncurved) polarization, component-first like GaussianLaser
        shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))
        return np.reshape(self.polarization_direction, (-1,) + (1,) * len(shape)) * np.ones((3,) + tuple(shape))
