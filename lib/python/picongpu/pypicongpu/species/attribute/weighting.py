"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from .attribute import Attribute


class Weighting(Attribute):
    """
    Position of a macroparticle
    """

    picongpu_name: str = "weighting"
