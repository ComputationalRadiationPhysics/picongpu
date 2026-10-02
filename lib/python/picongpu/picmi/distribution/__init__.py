"""
PICMI for PIConGPU
"""

from .UniformDistribution import UniformDistribution
from .FoilDistribution import FoilDistribution
from .Distribution import Distribution
from .GaussianDistribution import GaussianDistribution
from .GaussianBunchDistribution import GaussianBunchDistribution
from .CylindricalDistribution import CylindricalDistribution
from .AnalyticDistribution import AnalyticDistribution

AnyDistribution = (
    UniformDistribution
    | FoilDistribution
    | GaussianDistribution
    | GaussianBunchDistribution
    | CylindricalDistribution
    | AnalyticDistribution
)

__all__ = [
    "UniformDistribution",
    "FoilDistribution",
    "Distribution",
    "GaussianDistribution",
    "GaussianBunchDistribution",
    "AnalyticDistribution",
    "CylindricalDistribution",
]
