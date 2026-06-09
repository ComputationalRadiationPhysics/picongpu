# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from .dispersive_pulse_laser import DispersivePulseLaser
from .from_openpmd_pulse_laser import FromOpenPMDPulseLaser
from .gaussian_laser import GaussianLaser
from .plane_wave_laser import PlaneWaveLaser
from .polarization_type import PolarizationType
from .twts_laser import TWTSLaser

AnyLaser = DispersivePulseLaser | FromOpenPMDPulseLaser | GaussianLaser | PlaneWaveLaser | TWTSLaser

__all__ = [
    "AnyLaser",
    "DispersivePulseLaser",
    "FromOpenPMDPulseLaser",
    "GaussianLaser",
    "PlaneWaveLaser",
    "PolarizationType",
    "TWTSLaser",
]
