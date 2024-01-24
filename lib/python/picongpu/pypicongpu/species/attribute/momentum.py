"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .attribute import Attribute


class Momentum(Attribute):
    """
    Position of a macroparticle
    """

    PICONGPU_NAME = "momentum"

    def __init__(self):
        pass
