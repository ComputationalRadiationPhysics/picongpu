"""
PICMI for PIConGPU
"""

import sys

import picmistandard

from . import constants, diagnostics
from .constants import B, GB, GiB, KiB, MB, MiB, kB
from .distribution import (
    AnalyticDistribution,
    CylindricalDistribution,
    FoilDistribution,
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
    "Species",
    "MemoryConfig",
    "PrecisionConfig",
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
    "AnalyticDistribution",
    "ADK",
    "ADKVariant",
    "BSI",
    "BSIExtension",
    "Keldysh",
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
