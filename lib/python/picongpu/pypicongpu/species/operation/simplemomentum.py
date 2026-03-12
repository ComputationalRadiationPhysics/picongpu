"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from typing import Literal
from .momentum import Temperature, Drift
from ..species import Species
from pydantic import BaseModel


class SimpleMomentum(BaseModel):
    """
    provides momentum to a species

    specified by:

    - temperature
    - drift

    Both are optional. If both are missing, momentum **is still provided**, but
    left at 0 (default).
    """

    species: Species
    """species for which momentum will be set"""

    temperature: Temperature | None
    """temperature of particles (if any)"""

    drift: Drift | None
    """drift of particles (if any)"""

    type_simplemomentum: Literal[True] = True
