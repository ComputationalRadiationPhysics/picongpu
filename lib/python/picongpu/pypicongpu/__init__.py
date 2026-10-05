"""
internal representation of params to generate PIConGPU input files
"""

from . import backgroundfield, customuserinput, grid, laser, output, rendering, species, util
from .field_solver.ArbitraryOrderFDTD import ArbitraryOrderFDTDSolver
from .field_solver.CKC import CKCSolver
from .field_solver.Lehe import LeheSolver
from .field_solver.NoneSolver import NoneSolver
from .field_solver.Yee import YeeSolver
from .output.checkpoint import Checkpoint
from .output.energy_histogram import EnergyHistogram
from .output.field_energy_monitor import FieldEnergyMonitor
from .output.macro_particle_count import MacroParticleCount
from .output.phase_space import PhaseSpace
from .runner import Runner
from .simulation import Simulation

__all__ = [
    "Simulation",
    "Runner",
    "laser",
    "output",
    "rendering",
    "YeeSolver",
    "LeheSolver",
    "CKCSolver",
    "ArbitraryOrderFDTDSolver",
    "NoneSolver",
    "species",
    "util",
    "grid",
    "customuserinput",
    "backgroundfield",
    "PhaseSpace",
    "EnergyHistogram",
    "MacroParticleCount",
    "Checkpoint",
    "FieldEnergyMonitor",
]

# note: put down here b/c linter complains if imports are not at top
import sys

assert sys.version_info.major > 3 or sys.version_info.minor >= 9, "Python 3.9 is required for PIConGPU"
