"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch,
         Masoud Afshari
License: GPLv3+
"""

import math

import picmistandard
import typeguard

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
@typeguard.typechecked
class GaussianLaser(picmistandard.PICMI_GaussianLaser, BaseLaser):
    """
    PICMI object for Gaussian Laser.

    Standard Gaussian laser pulse parameters are:

    - wavelength : float
        Central wavelength of the laser [m].

    - waist : float
        Spot size (1/e^2 radius) of the laser at focus [m].

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

    def __init__(
        self,
        wavelength,
        waist,
        duration,
        propagation_direction,
        polarization_direction,
        focal_position,
        centroid_position,
        a0=None,
        E0=None,
        picongpu_polarization_type=(PolarizationType.LINEAR),
        picongpu_laguerre_modes: list[float] | None = None,
        picongpu_laguerre_phases: list[float] | None = None,
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
        **kw,
    ):
        if waist <= 0:
            raise ValueError(f"waist must be > 0. You gave {waist=}.")
        if wavelength <= 0:
            raise ValueError(f"wavelength must be > 0. You gave {wavelength=}.")
        if duration <= 0:
            raise ValueError(f"laser pulse duration must be > 0. You gave {duration=}.")

        assert (picongpu_laguerre_modes is None and picongpu_laguerre_phases is None) or (
            picongpu_laguerre_modes is not None and picongpu_laguerre_phases is not None
        ), (
            "laguerre_modes and laguerre_phases MUST BE both set or both \
            unset"
        )

        self.picongpu_polarization_type = picongpu_polarization_type
        self.picongpu_laguerre_modes = picongpu_laguerre_modes or [1.0]
        self.picongpu_laguerre_phases = picongpu_laguerre_phases or [0.0]
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions

        # Calculate a0 and E0 using our base laser, as the PICMI standard does not provide consistency checks.
        self.k0 = 2.0 * math.pi / wavelength
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, E0, a0)
        kw["E0"] = self.E0
        kw["a0"] = self.a0

        super().__init__(
            wavelength,
            waist,
            duration,
            propagation_direction,
            polarization_direction,
            focal_position,
            centroid_position,
            **kw,
        )

        self.phi0 = self.phi0 or 0.0
        self._validate_common_properties()
        self.pulse_init = self._compute_pulse_init()

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

    def check(self):
        util.unsupported("laser name", self.name)
        util.unsupported("laser zeta", self.zeta)
        util.unsupported("laser beta", self.beta)
        util.unsupported("laser phi2", self.phi2)
        # unsupported: fill_in (do not warn, b/c we don't know if it has been
        # set explicitly, and always warning is bad)

        self._validate_common_properties()
        assert self._propagation_connects_centroid_and_focus(), (
            "propagation_direction must connect centroid_position and focus_position"
        )
