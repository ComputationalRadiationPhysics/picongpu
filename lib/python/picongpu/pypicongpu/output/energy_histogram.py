"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, PrivateAttr

from picongpu.pypicongpu.output.plugin import Plugin
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.species import Species


class EnergyHistogram(Plugin, BaseModel):
    species: Species | FilteredSpecies
    period: TimeStepSpec
    bin_count: int
    min_energy: float
    max_energy: float

    _name: str = PrivateAttr("energyhistogram")
