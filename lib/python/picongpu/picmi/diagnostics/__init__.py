"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from .backend_config import BackendConfig, OpenPMDConfig
from .binning import Binning, BinningAxis, BinSpec
from .checkpoint import Checkpoint
from .energy_histogram import EnergyHistogram
from .field_dump import DerivedFieldDump, NativeFieldDump
from .field_energy_monitor import FieldEnergyMonitor
from .macro_particle_count import MacroParticleCount
from .particle_dump import ParticleDump
from .particle_energy import ParticleEnergy
from .phase_space import PhaseSpace
from .radiation import Radiation
from .timestepspec import TS, TimeStepSpec

AnyDiagnostic = (
    Binning
    | Checkpoint
    | EnergyHistogram
    | DerivedFieldDump
    | NativeFieldDump
    | FieldEnergyMonitor
    | MacroParticleCount
    | ParticleDump
    | ParticleEnergy
    | PhaseSpace
    | Radiation
)
__all__ = [
    "AnyDiagnostic",
    "BackendConfig",
    "OpenPMDConfig",
    "Binning",
    "BinningAxis",
    "BinSpec",
    "PhaseSpace",
    "EnergyHistogram",
    "FieldEnergyMonitor",
    "MacroParticleCount",
    "ParticleDump",
    "ParticleEnergy",
    "NativeFieldDump",
    "DerivedFieldDump",
    "TimeStepSpec",
    "TS",
    "Checkpoint",
    "Radiation",
]
