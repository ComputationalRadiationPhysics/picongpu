"""
PICMI for PIConGPU
"""

from .AnalyticDistribution import AnalyticDistribution as AnalyticDistribution
from .CylindricalDistribution import CylindricalDistribution as CylindricalDistribution
from .Distribution import Distribution as Distribution
from .FoilDistribution import FoilDistribution as FoilDistribution
from .GaussianBunchDistribution import GaussianBunchDistribution as GaussianBunchDistribution
from .GaussianDistribution import GaussianDistribution as GaussianDistribution
from .UniformDistribution import UniformDistribution as UniformDistribution

AnyDistribution = (
    UniformDistribution
    | FoilDistribution
    | GaussianDistribution
    | GaussianBunchDistribution
    | CylindricalDistribution
    | AnalyticDistribution
)
