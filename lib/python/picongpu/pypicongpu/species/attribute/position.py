"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
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
