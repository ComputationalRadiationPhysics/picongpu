"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, PrivateAttr, model_validator

from picongpu.pypicongpu.output.plugin import Plugin
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.species import Species


class PhaseSpace(Plugin, BaseModel):
    species: Species | FilteredSpecies
    period: TimeStepSpec
    spatial_coordinate: Literal["x", "y", "z"]
    momentum_coordinate: Literal["px", "py", "pz"]
    min_momentum: float
    max_momentum: float

    _name: str = PrivateAttr("phasespace")

    @model_validator(mode="after")
    def check(self):
        if self.min_momentum >= self.max_momentum:
            raise ValueError(
                "PhaseSpace's min_momentum should be smaller than max_momentum. "
                f"You gave: {self.min_momentum=} and {self.max_momentum=}."
            )
        return self
