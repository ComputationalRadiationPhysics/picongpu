# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2023-2025 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field

from .plasmaramp import AllPlasmaRamps, None_


class Foil(BaseModel):
    """
    Directional density profile with thickness and pre- and
    post-plasma lengths and cutoffs
    """

    type_foil: Literal[True] = True

    density_si: float = Field(gt=0.0)
    """particle number density at at the foil plateau (m^-3)"""

    y_value_front_foil_si: float = Field(ge=0.0)
    """position of the front of the foil plateau (m)"""

    thickness_foil_si: float = Field(ge=0.0)
    """thickness of the foil plateau (m)"""

    pre_foil_plasmaRamp: AllPlasmaRamps = None_()
    """pre(lower y) foil-plateau ramp of density"""

    post_foil_plasmaRamp: AllPlasmaRamps = None_()
    """post(higher y) foil-plateau ramp of density"""
