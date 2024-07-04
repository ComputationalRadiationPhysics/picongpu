"""
PICMI for PIConGPU
"""

from .simulation import Simulation
from .grid import Cartesian3DGrid
from .solver import ElectromagneticSolver
from .gaussian_laser import GaussianLaser
from .species import Species
from .layout import PseudoRandomLayout
from . import constants

from .distribution import FoilDistribution
from .distribution import UniformDistribution
from .distribution import GaussianDistribution

import picmistandard

import sys

assert sys.version_info.major > 3 or sys.version_info.minor >= 10, "Python 3.10 is required for PIConGPU PICMI"

__all__ = [
    "Simulation",
    "Cartesian3DGrid",
    "ElectromagneticSolver",
    "GaussianLaser",
    "Species",
    "PseudoRandomLayout",
    "FoilDistribution",
    "UniformDistribution",
    "GaussianDistribution",
    "constants",
]

codename = "picongpu"
"""
name of this PICMI implementation
required by PICMI interface
"""

picmistandard.register_constants(constants)
