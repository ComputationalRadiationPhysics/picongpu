"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from . import plasmaramp as plasmaramp
from .cylinder import Cylinder as Cylinder
from .foil import Foil as Foil
from .free_formula import FreeFormula as FreeFormula
from .gaussian import Gaussian as Gaussian
from .uniform import Uniform as Uniform

AnyDensityProfile = Uniform | Foil | Gaussian | FreeFormula | Cylinder
