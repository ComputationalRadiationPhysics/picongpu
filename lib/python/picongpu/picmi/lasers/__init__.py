"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from .dispersive_pulse_laser import DispersivePulseLaser
from .from_openpmd_pulse_laser import FromOpenPMDPulseLaser
from .from_lasy_laser import FromLasyLaser
from .gaussian_laser import GaussianLaser
from .plane_wave_laser import PlaneWaveLaser
from .polarization_type import PolarizationType
from .twts_laser import TWTSLaser

AnyLaser = DispersivePulseLaser | FromOpenPMDPulseLaser | FromLasyLaser | GaussianLaser | PlaneWaveLaser | TWTSLaser

__all__ = [
    "AnyLaser",
    "DispersivePulseLaser",
    "FromOpenPMDPulseLaser",
    "FromLasyLaser",
    "GaussianLaser",
    "PlaneWaveLaser",
    "PolarizationType",
    "TWTSLaser",
]
