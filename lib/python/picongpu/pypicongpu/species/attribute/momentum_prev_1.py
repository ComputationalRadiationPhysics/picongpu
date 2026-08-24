"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .attribute import Attribute


class MomentumPrev1(Attribute):
    """
    Momentum of previous time step of a macroparticle
    """

    picongpu_name: str = "momentumPrev1"
