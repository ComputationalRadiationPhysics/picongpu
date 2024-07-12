"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .attribute import Attribute


class Position(Attribute):
    """
    Position of a macroparticle
    """

    PICONGPU_NAME = "position<position_pic>"

    def __init__(self):
        pass
