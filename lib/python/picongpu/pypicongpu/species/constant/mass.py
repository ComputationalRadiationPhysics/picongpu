# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from pydantic import Field
from .constant import Constant


class Mass(Constant):
    """
    mass of a physical particle
    """

    mass_si: float = Field(ge=0.0)
    """mass in kg of an individual particle"""
