"""
This file is part of PIConGPU.
Copyright 2023-2025 PIConGPU contributors
Authors: Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from .....rendering import SelfRegisteringRenderedObject
import typeguard


@typeguard.typechecked
class PlasmaRamp(SelfRegisteringRenderedObject):
    """
    abstract parent class for all plasma ramps

    A plasma ramp describes ramp up of an edge of an initial density
    distribution
    """

    pass
