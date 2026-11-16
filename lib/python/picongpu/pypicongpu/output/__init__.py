"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from .binning import Binning as Binning
from .checkpoint import Checkpoint as Checkpoint
from .energy_histogram import EnergyHistogram as EnergyHistogram
from .field_energy_monitor import FieldEnergyMonitor as FieldEnergyMonitor
from .macro_particle_count import MacroParticleCount as MacroParticleCount
from .openpmd_plugin import OpenPMDPlugin as OpenPMDPlugin
from .particle_energy import ParticleEnergy as ParticleEnergy
from .phase_space import PhaseSpace as PhaseSpace
from .radiation import RadiationConfiguration as RadiationConfiguration
from .radiation import RadiationObserverConfiguration as RadiationObserverConfiguration
from .radiation import RadiationPlugin as RadiationPlugin
from .timestepspec import TimeStepSpec as TimeStepSpec

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
