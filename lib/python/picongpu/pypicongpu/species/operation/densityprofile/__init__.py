"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
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
