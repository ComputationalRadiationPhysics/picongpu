"""
internal representation of params to generate PIConGPU input files
"""

from .simulation import Simulation
from .runner import Runner

from . import laser
from . import grid
from . import rendering
from . import solver
from . import species
from . import util
from . import output

__all__ = [
    "Simulation",
    "Runner",
    "laser",
    "output",
    "rendering",
    "solver",
    "species",
    "util",
    "grid",
]

# note: put down here b/c linter complains if imports are not at top
import sys

assert sys.version_info.major > 3 or sys.version_info.minor >= 9, "Python 3.9 is required for PIConGPU"
