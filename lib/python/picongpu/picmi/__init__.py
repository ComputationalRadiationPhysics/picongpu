"""
PICMI for PIConGPU
------------------

The top level exposes exactly the objects a user passes to
:class:`~picongpu.picmi.simulation.Simulation` or
:class:`~picongpu.picmi.species.Species` (or picmistandard names), plus the
deliberately bound concern submodules. Everything else lives at its concern
level:

* diagnostics and their inputs (``RangeSpec``, ``TimeStepSpec``, ...) live in
  :mod:`picongpu.picmi.diagnostics`,
* functor-authoring helpers (``ParticleFunctor``, ``ParticleFilter``,
  ``FilteredSpecies``, ...) live in :mod:`picongpu.picmi.particle_functor`,
* unions (``AnyInteraction``, ``AnyAppliedField``, ...) live with their concern
  submodule and are never promoted to the top level,
* physical constants live only in :mod:`picongpu.picmi.constants`.

Backend (``pypicongpu``) objects are not re-exported here.
"""

import sys as _sys

import picmistandard as _picmistandard

# Deliberately bound concern submodules. Their contents are not flattened into
# the top level beyond the user-input classes imported below.
from . import applied_field as applied_field
from . import constants as constants
from . import diagnostics as diagnostics
from . import distribution as distribution
from . import grid as grid
from . import interaction as interaction
from . import lasers as lasers
from . import layout as layout
from . import memory_config as memory_config
from . import multi_species as multi_species
from . import particle_functor as particle_functor
from . import precision_config as precision_config
from . import simulation as simulation
from . import solver as solver
from . import species as species
from . import unit_dimension as unit_dimension
from .applied_field import AnalyticAppliedField as AnalyticAppliedField
from .applied_field import ConstantAppliedField as ConstantAppliedField
from .distribution import AnalyticDistribution as AnalyticDistribution
from .distribution import CylindricalDistribution as CylindricalDistribution
from .distribution import FoilDistribution as FoilDistribution
from .distribution import GaussianBunchDistribution as GaussianBunchDistribution
from .distribution import GaussianDistribution as GaussianDistribution
from .distribution import UniformDistribution as UniformDistribution
from .grid import Cartesian2DGrid as Cartesian2DGrid
from .grid import Cartesian3DGrid as Cartesian3DGrid
from .interaction import Collision as Collision
from .interaction import CollisionalPhysicsSetup as CollisionalPhysicsSetup
from .interaction import ConstLogCollision as ConstLogCollision
from .interaction import DynamicLogCollision as DynamicLogCollision
from .interaction import Synchrotron as Synchrotron
from .interaction.ionization.electroniccollisionalequilibrium import (
    ThomasFermi as ThomasFermi,
)
from .interaction.ionization.fieldionization import ADK as ADK
from .interaction.ionization.fieldionization import BSI as BSI
from .interaction.ionization.fieldionization import ADKVariant as ADKVariant
from .interaction.ionization.fieldionization import BSIExtension as BSIExtension
from .interaction.ionization.fieldionization import FieldIonization as FieldIonization
from .interaction.ionization.fieldionization import Keldysh as Keldysh
from .lasers import DispersivePulseLaser as DispersivePulseLaser
from .lasers import FromOpenPMDPulseLaser as FromOpenPMDPulseLaser
from .lasers import GaussianLaser as GaussianLaser
from .lasers import PlaneWaveLaser as PlaneWaveLaser
from .lasers import PolarizationType as PolarizationType
from .lasers import TWTSLaser as TWTSLaser
from .layout import GriddedLayout as GriddedLayout
from .layout import OnePositionLayout as OnePositionLayout
from .layout import PseudoRandomLayout as PseudoRandomLayout
from .memory_config import MemoryConfig as MemoryConfig
from .multi_species import MultiSpecies as MultiSpecies
from .precision_config import PrecisionConfig as PrecisionConfig
from .simulation import Simulation as Simulation
from .solver import BinomialSmoother as BinomialSmoother
from .solver import ElectromagneticSolver as ElectromagneticSolver
from .solver import ElectrostaticSolver as ElectrostaticSolver
from .species import Species as Species

assert _sys.version_info.major > 3 or _sys.version_info.minor >= 11, "Python 3.11 is required for PIConGPU PICMI"


codename = "picongpu"
"""
name of this PICMI implementation
required by PICMI interface
"""

_picmistandard.register_codename(codename)
_picmistandard.register_constants(constants)
