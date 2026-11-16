"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from pydantic import Field, model_validator

from ...pypicongpu import laser
from .._copy_attributes import default_converts_to
from .gaussian_laser import GaussianLaser


@default_converts_to(
    laser.DispersivePulseLaser,
    # PICMI's `duration` is the standard 1/e field width (tau), while PIConGPU's
    # `pulse_duration_si` (aliased as `duration`) is the 1 sigma of the intensity,
    # i.e. PULSE_DURATION = duration / 2 (#5739)
    conversions={"duration": lambda self, *args, **kwargs: self._pulse_duration_sigma_si()},
)
class DispersivePulseLaser(GaussianLaser):
    """
    PICMI Dispersive Pulse Laser.

    Extends `GaussianLaser` with additional dispersion-specific parameters.
    """

    picongpu_spectral_support: float = Field(default=6.0, description="Width of spectral support (dimensionless).")
    picongpu_sd_si: float = Field(default=0.0, description="Spatial dispersion coefficient [m*s].")
    picongpu_ad_si: float = Field(default=0.0, description="Angular dispersion coefficient [rad*s].")
    picongpu_gdd_si: float = Field(default=0.0, description="Group delay dispersion (GDD) [s^2].")
    picongpu_tod_si: float = Field(default=0.0, description="Third-order dispersion (TOD) [s^3].")

    @model_validator(mode="wrap")
    @classmethod
    def _forbid_laguerre(cls, data, handler):
        if isinstance(data, dict):
            if data.get("picongpu_laguerre_modes", None) is not None:
                raise ValueError("DispersivePulseLaser does not support Laguerre modes.")
            if data.get("picongpu_laguerre_phases", None) is not None:
                raise ValueError("DispersivePulseLaser does not support Laguerre phases.")
        return handler(data)
