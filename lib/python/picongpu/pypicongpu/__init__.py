# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
internal representation of params to generate PIConGPU input files
"""

from . import customuserinput, grid, laser, output, rendering, species, util
from .field_solver.Lehe import LeheSolver
from .field_solver.Yee import YeeSolver
from .output.checkpoint import Checkpoint
from .output.energy_histogram import EnergyHistogram
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
    "species",
    "util",
    "grid",
    "customuserinput",
    "PhaseSpace",
    "EnergyHistogram",
    "MacroParticleCount",
    "Checkpoint",
]

# note: put down here b/c linter complains if imports are not at top
import sys

assert sys.version_info.major > 3 or sys.version_info.minor >= 9, "Python 3.9 is required for PIConGPU"
