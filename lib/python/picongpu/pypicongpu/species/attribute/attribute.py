"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from pydantic import BaseModel


class Attribute(BaseModel):
    """
    attribute of a species

    Property of individual macroparticles (i.e. can be different from
    macroparticle to macroparticle).
    Can change over time (not relevant for initialization here).

    Owned by exactly one species.

    Set by exactly one operation (an operation may define multiple attributes
    even across multiple species though).

    Identified by its PIConGPU name.

    PIConGPU term: "particle attributes"
    """

    picongpu_name: str
    """C++ Code implementing this attribute"""
