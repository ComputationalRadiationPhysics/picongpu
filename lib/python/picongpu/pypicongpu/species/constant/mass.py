"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from pydantic import Field
from .constant import Constant


class Mass(Constant):
    """
    mass of a physical particle
    """

    mass_si: float = Field(ge=0.0)
    """mass in kg of an individual particle"""
