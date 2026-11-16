"""
internal representation of params to generate PIConGPU input files

The package root is intentionally thin: it exposes the two entry points
:class:`~picongpu.pypicongpu.runner.Runner` and
:class:`~picongpu.pypicongpu.simulation.Simulation` together with the
explicitly imported submodules. Classes otherwise stay in their submodule
(``pypicongpu.field_solver.YeeSolver``, ``pypicongpu.output.Checkpoint``,
...).
"""

import sys as _sys

from . import backgroundfield as backgroundfield
from . import collision as collision
from . import customuserinput as customuserinput
from . import field_solver as field_solver
from . import grid as grid
from . import laser as laser
from . import memory as memory
from . import movingwindow as movingwindow
from . import output as output
from . import particle_functor as particle_functor
from . import precision_config as precision_config
from . import rendering as rendering
from . import runner as runner
from . import simulation as simulation
from . import species as species
from . import util as util
from . import walltime as walltime
from .runner import Runner as Runner
from .simulation import Simulation as Simulation

assert _sys.version_info.major > 3 or _sys.version_info.minor >= 9, "Python 3.9 is required for PIConGPU"
