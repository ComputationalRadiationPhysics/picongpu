"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Literal

from pydantic import BaseModel, Field


class Uniform(BaseModel):
    """
    globally constant density

    PIConGPU equivalent is the homogenous profile, but due to spelling
    ambiguities the PICMI name uniform is followed here.
    """

    type_uniform: Literal[True] = True

    density_si: float = Field(gt=0.0)
    """density at every point in space (kg * m^-3)"""
