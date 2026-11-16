"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Julian Lenz, Masoud Afshari
License: GPLv3+
"""

from .backend_config import BackendConfig as BackendConfig
from .backend_config import OpenPMDConfig as OpenPMDConfig
from .backend_config import OpenPMDBackendConfig as OpenPMDBackendConfig
from .backend_config import RangeSpec as RangeSpec
from .backend_config import RangeSpecEntry as RangeSpecEntry
from .backend_config import RangeSpecUnit as RangeSpecUnit
from .binning import Binning as Binning
from .binning import BinningAxis as BinningAxis
from .binning import BinSpec as BinSpec
from .checkpoint import Checkpoint as Checkpoint
from .energy_histogram import EnergyHistogram as EnergyHistogram
from .field_dump import DerivedFieldDump as DerivedFieldDump
from .field_dump import NativeFieldDump as NativeFieldDump
from .field_energy_monitor import FieldEnergyMonitor as FieldEnergyMonitor
from .macro_particle_count import MacroParticleCount as MacroParticleCount
from .particle_dump import ParticleDump as ParticleDump
from .particle_energy import ParticleEnergy as ParticleEnergy
from .phase_space import PhaseSpace as PhaseSpace
from .radiation import Radiation as Radiation
from .radiation import RadiationObserverConfiguration as RadiationObserverConfiguration
from .timestepspec import TS as TS
from .timestepspec import TimeStepSpec as TimeStepSpec

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
