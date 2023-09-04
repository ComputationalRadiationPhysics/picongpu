"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""


class Attribute:
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

    PICONGPU_NAME: str = None
    """C++ Code implementing this attribute"""

    def __init__(self):
        raise NotImplementedError()
