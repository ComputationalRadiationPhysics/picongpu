"""
# SPDX-FileCopyrightText: Masoud Afshari, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Literal

from pydantic import BaseModel, Field

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.species import Species


class EnergyHistogram(BaseModel):
    species: Species | FilteredSpecies
    period: TimeStepSpec
    bin_count: int = Field(gt=0)
    min_energy: float
    max_energy: float

    type_energyhistogram: Literal[True] = True
