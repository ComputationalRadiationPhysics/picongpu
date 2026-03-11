"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator

from ..species import Species
from .densityprofile import AnyDensityProfile
from .layout import AnyLayout


class SimpleDensity(BaseModel):
    """
    Place a set of species together, using the same density profile

    These species will have **the same** macroparticle placement.

    For this operation, only the random layout is supported.

    parameters:

    - ppc: particles placed per cell
    - profile: density profile to use
    - species: species to be placed with the given profile
      note that their density ratios will be respected
    """

    profile: AnyDensityProfile
    """density profile to use, describes the actual density"""

    species: list[Species] = Field(exclude=True)
    """species to be placed"""

    layout: AnyLayout

    type_simpledensity: Literal[True] = True

    @field_validator("species", mode="before")
    @classmethod
    def validate_species(cls, species):
        return sorted(
            set(species),
            key=lambda species: (
                None if species.constants.density_ratio is None else species.constants.density_ratio.ratio
            ),
        )

    @computed_field
    def placed_species_initial(self) -> Species:
        return self.species[0]

    @computed_field
    def placed_species_copied(self) -> list[Species]:
        return self.species[1:]

    def __init__(self, *args, **kwargs):
        return BaseModel.__init__(self, *args, **kwargs)
