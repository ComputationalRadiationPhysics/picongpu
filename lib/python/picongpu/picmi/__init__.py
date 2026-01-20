"""
PICMI for PIConGPU
"""

from .simulation import Simulation
from .grid import Cartesian3DGrid
from .solver import ElectromagneticSolver
from .lasers import DispersivePulseLaser, GaussianLaser, PlaneWaveLaser, FromOpenPMDPulseLaser, TWTSLaser
from .species import Species
from .layout import PseudoRandomLayout, GriddedLayout
from . import constants

from . import diagnostics

from .distribution import (
    FoilDistribution,
    UniformDistribution,
    GaussianDistribution,
    CylindricalDistribution,
    AnalyticDistribution,
)

from .interaction import Interaction, Synchrotron, Collision, ConstLogCollision, DynamicLogCollision
from .interaction.ionization.fieldionization import (
    ADK,
    ADKVariant,
    BSI,
    BSIExtension,
    Keldysh,
)
from .interaction.ionization.electroniccollisionalequilibrium import ThomasFermi

import picmistandard

import sys

assert sys.version_info.major > 3 or sys.version_info.minor >= 11, "Python 3.11 is required for PIConGPU PICMI"

__all__ = [
    "Simulation",
    "Cartesian3DGrid",
    "ElectromagneticSolver",
    "DispersivePulseLaser",
    "FromOpenPMDPulseLaser",
    "GaussianLaser",
    "TWTSLaser",
    "PlaneWaveLaser",
    "Species",
    "PseudoRandomLayout",
    "GriddedLayout",
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
