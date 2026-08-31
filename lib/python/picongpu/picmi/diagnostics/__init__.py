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
from .field_dump import AverageDerivedFieldDump, DerivedFieldDump, NativeDerivedFieldDump, NativeFieldDump
from .macro_particle_count import MacroParticleCount
from .particle_dump import ParticleDump
from .phase_space import PhaseSpace
from .radiation import Radiation
from .timestepspec import TimeStepSpec

AnyDiagnostic = (
    Binning
    | Checkpoint
    | EnergyHistogram
    | DerivedFieldDump
    | NativeDerivedFieldDump
    | AverageDerivedFieldDump
    | NativeFieldDump
    | MacroParticleCount
    | ParticleDump
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
    "MacroParticleCount",
    "ParticleDump",
    "NativeFieldDump",
    "DerivedFieldDump",
    "NativeDerivedFieldDump",
    "AverageDerivedFieldDump",
    "TimeStepSpec",
    "Checkpoint",
    "Radiation",
]
