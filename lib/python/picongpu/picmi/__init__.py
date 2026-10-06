"""
PICMI for PIConGPU
"""

import sys

import picmistandard

from . import constants, diagnostics
from .applied_field import AnalyticAppliedField, ConstantAppliedField
from .constants import B, GB, GiB, KiB, MB, MiB, kB
from .distribution import (
    AnalyticDistribution,
    CylindricalDistribution,
    FoilDistribution,
    GaussianBunchDistribution,
    GaussianDistribution,
    UniformDistribution,
)
from .grid import Cartesian2DGrid, Cartesian3DGrid
from .interaction import (
    Collision,
    ConstLogCollision,
    DynamicLogCollision,
    Interaction,
    Synchrotron,
)
from .interaction.ionization.electroniccollisionalequilibrium import ThomasFermi
from .interaction.ionization.fieldionization import (
    ADK,
    BSI,
    ADKVariant,
    BSIExtension,
    FieldIonization,
    Keldysh,
)
from .lasers import (
    DispersivePulseLaser,
    FromOpenPMDPulseLaser,
    GaussianLaser,
    PlaneWaveLaser,
    TWTSLaser,
)
from .layout import GriddedLayout, OnePositionLayout, PseudoRandomLayout
from .memory_config import MemoryConfig
from .multi_species import MultiSpecies
from .particle_functor import FilteredSpecies, ParticleFilter, ParticleFunctor
from .precision_config import PrecisionConfig
from .simulation import Simulation
from .solver import BinomialSmoother, ElectromagneticSolver
from .species import Species

assert sys.version_info.major > 3 or sys.version_info.minor >= 11, "Python 3.11 is required for PIConGPU PICMI"

__all__ = [
    "Simulation",
    "ParticleFunctor",
    "Cartesian3DGrid",
    "Cartesian2DGrid",
    "ElectromagneticSolver",
    "BinomialSmoother",
    "DispersivePulseLaser",
    "FromOpenPMDPulseLaser",
    "GaussianLaser",
    "TWTSLaser",
    "PlaneWaveLaser",
    "ConstantAppliedField",
    "AnalyticAppliedField",
    "Species",
    "MemoryConfig",
    "PrecisionConfig",
    "MultiSpecies",
    "FilteredSpecies",
    "ParticleFilter",
    "PseudoRandomLayout",
    "GriddedLayout",
    "OnePositionLayout",
    "constants",
    "B",
    "kB",
    "MB",
    "GB",
    "KiB",
    "MiB",
    "GiB",
    "FoilDistribution",
    "UniformDistribution",
    "GaussianDistribution",
    "GaussianBunchDistribution",
    "AnalyticDistribution",
    "ADK",
    "ADKVariant",
    "BSI",
    "BSIExtension",
    "Keldysh",
    "FieldIonization",
    "ThomasFermi",
    "Synchrotron",
    "Interaction",
    "diagnostics",
    "CylindricalDistribution",
    "Collision",
    "ConstLogCollision",
    "DynamicLogCollision",
]


codename = "picongpu"
"""
name of this PICMI implementation
required by PICMI interface
"""

picmistandard.register_codename(codename)
picmistandard.register_constants(constants)
