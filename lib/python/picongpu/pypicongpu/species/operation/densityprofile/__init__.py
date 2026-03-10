"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from .uniform import Uniform
from .foil import Foil
from .gaussian import Gaussian
from .cylinder import Cylinder
from .free_formula import FreeFormula

from . import plasmaramp

AnyDensityProfile = Uniform | Foil | Gaussian | FreeFormula | Cylinder

__all__ = [
    "AnyDensityProfile",
    "Uniform",
    "Foil",
    "plasmaramp",
    "Gaussian",
    "FreeFormula",
    "Cylinder",
]
