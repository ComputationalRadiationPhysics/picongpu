"""
internal representation of params to generate PIConGPU input files
"""

from .simulation import Simulation
from .runner import Runner
from .output.phase_space import PhaseSpace
from .output.energy_histogram import EnergyHistogram
from .output.macro_particle_count import MacroParticleCount
from .output.png import Png
from .output.checkpoint import Checkpoint
from .field_solver.DefaultSolver import Solver
from .field_solver.Yee import YeeSolver
from .field_solver.Lehe import LeheSolver

from . import laser
from . import grid
from . import rendering
from . import species
from . import util
from . import output
from . import customuserinput

__all__ = [
    "Simulation",
    "Runner",
    "laser",
    "output",
    "rendering",
    "Solver",
    "YeeSolver",
    "LeheSolver",
    "species",
    "util",
    "grid",
    "customuserinput",
    "PhaseSpace",
    "EnergyHistogram",
    "MacroParticleCount",
    "Png",
    "Checkpoint",
]

# note: put down here b/c linter complains if imports are not at top
import sys

assert sys.version_info.major > 3 or sys.version_info.minor >= 9, "Python 3.9 is required for PIConGPU"
