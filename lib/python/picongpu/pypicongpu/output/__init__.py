"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from .binning import Binning
from .checkpoint import Checkpoint
from .energy_histogram import EnergyHistogram
from .field_energy_monitor import FieldEnergyMonitor
from .macro_particle_count import MacroParticleCount
from .openpmd_plugin import OpenPMDPlugin
from .particle_energy import ParticleEnergy
from .phase_space import PhaseSpace
from .radiation import RadiationConfiguration, RadiationPlugin, RadiationObserverConfiguration
from .timestepspec import TimeStepSpec

AnyPlugin = (
    Binning
    | Checkpoint
    | EnergyHistogram
    | FieldEnergyMonitor
    | MacroParticleCount
    | OpenPMDPlugin
    | ParticleEnergy
    | PhaseSpace
    | RadiationPlugin
)

__all__ = [
    "OpenPMDPlugin",
    "AnyPlugin",
    "PhaseSpace",
    "EnergyHistogram",
    "FieldEnergyMonitor",
    "MacroParticleCount",
    "ParticleEnergy",
    "TimeStepSpec",
    "Checkpoint",
    "RadiationPlugin",
    "RadiationObserverConfiguration",
    "RadiationConfiguration",
]
