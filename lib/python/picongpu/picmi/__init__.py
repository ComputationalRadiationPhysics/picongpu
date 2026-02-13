"""
PICMI for PIConGPU
"""

import sys

import picmistandard

from . import constants, diagnostics
from .distribution import (
    AnalyticDistribution,
    CylindricalDistribution,
    FoilDistribution,
    GaussianDistribution,
    UniformDistribution,
)
from .grid import Cartesian3DGrid
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
from .particle_functor import FilteredSpecies, ParticleFilter, ParticleFunctor
from .simulation import Simulation
from .solver import ElectromagneticSolver
from .species import Species

assert sys.version_info.major > 3 or sys.version_info.minor >= 11, "Python 3.11 is required for PIConGPU PICMI"

__all__ = [
    "Simulation",
    "ParticleFunctor",
    "Cartesian3DGrid",
    "ElectromagneticSolver",
    "DispersivePulseLaser",
    "FromOpenPMDPulseLaser",
    "GaussianLaser",
    "TWTSLaser",
    "PlaneWaveLaser",
    "Species",
    "FilteredSpecies",
    "ParticleFilter",
    "PseudoRandomLayout",
    "GriddedLayout",
    "OnePositionLayout",
    "constants",
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
