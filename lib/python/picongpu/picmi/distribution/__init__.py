# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PICMI for PIConGPU
"""

from .UniformDistribution import UniformDistribution
from .FoilDistribution import FoilDistribution
from .Distribution import Distribution
from .GaussianDistribution import GaussianDistribution
from .CylindricalDistribution import CylindricalDistribution
from .AnalyticDistribution import AnalyticDistribution

AnyDistribution = (
    UniformDistribution | FoilDistribution | GaussianDistribution | CylindricalDistribution | AnalyticDistribution
)

__all__ = [
    "UniformDistribution",
    "FoilDistribution",
    "Distribution",
    "GaussianDistribution",
    "AnalyticDistribution",
    "CylindricalDistribution",
]
