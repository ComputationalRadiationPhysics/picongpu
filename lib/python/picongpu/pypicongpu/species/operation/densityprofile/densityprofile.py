"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from ....rendering import SelfRegisteringRenderedObject

import typeguard


@typeguard.typechecked
class DensityProfile(SelfRegisteringRenderedObject):
    """
    (abstract) parent class of all density profiles

    A density profile describes the density in space.
    """

    pass
