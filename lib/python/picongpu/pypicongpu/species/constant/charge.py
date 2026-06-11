"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import typeguard

from .constant import Constant


@typeguard.typechecked
class Charge(Constant):
    """
    charge of a physical particle
    """

    charge_si: float
    """charge in C of an individual particle"""
