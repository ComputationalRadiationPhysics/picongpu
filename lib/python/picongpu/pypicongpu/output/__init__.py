"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .binning import Binning
from .checkpoint import Checkpoint
from .energy_histogram import EnergyHistogram
from .macro_particle_count import MacroParticleCount
from .openpmd_plugin import OpenPMDPlugin
from .phase_space import PhaseSpace
from .radiation import RadiationConfiguration, RadiationPlugin, RadiationObserverConfiguration
from .timestepspec import TimeStepSpec

AnyPlugin = Binning | Checkpoint | EnergyHistogram | MacroParticleCount | OpenPMDPlugin | PhaseSpace | RadiationPlugin

__all__ = [
    "OpenPMDPlugin",
    "AnyPlugin",
    "PhaseSpace",
    "EnergyHistogram",
    "MacroParticleCount",
    "TimeStepSpec",
    "Checkpoint",
    "RadiationPlugin",
    "RadiationObserverConfiguration",
    "RadiationConfiguration",
]
