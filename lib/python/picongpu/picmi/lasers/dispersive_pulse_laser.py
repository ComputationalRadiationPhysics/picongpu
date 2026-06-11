"""
# SPDX-FileCopyrightText: Julian Lenz, Masoud Afshari
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import typeguard

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .gaussian_laser import GaussianLaser  # inherit standard Gaussian laser fields


@default_converts_to(laser.DispersivePulseLaser)
@typeguard.typechecked
class DispersivePulseLaser(GaussianLaser):
    """
    PICMI Dispersive Pulse Laser.

    Extends `GaussianLaser` class with additional dispersion-specific parameters used in PIConGPU.

    All standard Gaussian laser fields are inherited. please refer to class:`GaussianLaser`

    Additional dispersive parameters (PIConGPU-specific):

    - picongpu_spectral_support : float, default=6.0
        Width of spectral support (dimensionless).

    - picongpu_sd_si : float, default=0.0
        Spatial dispersion coefficient [m*s].

    - picongpu_ad_si : float, default=0.0
        Angular dispersion coefficient [rad*s].

    - picongpu_gdd_si : float, default=0.0
        Group delay dispersion (GDD) [s^2].

    - picongpu_tod_si : float, default=0.0
        Third-order dispersion (TOD) [s^3].
    """

    def __init__(
        self,
        picongpu_spectral_support: float = 6.0,
        picongpu_sd_si: float = 0.0,
        picongpu_ad_si: float = 0.0,
        picongpu_gdd_si: float = 0.0,
        picongpu_tod_si: float = 0.0,
        **kw,  # all standard GaussianLaser arguments
    ):
        # Forbid Laguerre modes and phases
        if "picongpu_laguerre_modes" in kw and kw["picongpu_laguerre_modes"] is not None:
            raise ValueError("DispersivePulseLaser does not support Laguerre modes.")
        if "picongpu_laguerre_phases" in kw and kw["picongpu_laguerre_phases"] is not None:
            raise ValueError("DispersivePulseLaser does not support Laguerre phases.")

        # Initialize standard Gaussian laser fields
        super().__init__(**kw)

        # Store dispersive extensions
        self.picongpu_spectral_support = picongpu_spectral_support
        self.picongpu_sd_si = picongpu_sd_si
        self.picongpu_ad_si = picongpu_ad_si
        self.picongpu_gdd_si = picongpu_gdd_si
        self.picongpu_tod_si = picongpu_tod_si
