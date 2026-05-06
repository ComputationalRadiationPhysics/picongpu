"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, computed_field

from picongpu.picmi.particle_functor.particle_functor import Particle, ParticleFunctor
from picongpu.picmi.species import Species
from picongpu.pypicongpu.particle_functor import FilteredSpecies as PyPIConGPUFilteredSpecies


class ParticleFilter(ParticleFunctor):
    def __init__(self, functor: Callable[[Particle], Any], name: str | None = None):
        return super().__init__(name=name, functor=functor, return_type=bool, unit_dimension=None)

    def get_as_pypicongpu(self, mode="Filter"):
        return super().get_as_pypicongpu(mode=mode)


class FilteredSpecies(BaseModel):
    species: Species
    functor: ParticleFilter

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    def name_with_filter(self) -> str:
        return f"{self.species.name}_{self.functor.name}"

    @computed_field
    def species_name(self) -> str:
        return self.species.name

    @computed_field
    def name(self) -> str:
        return self.name_with_filter

    def get_as_pypicongpu(self, mode="Filter"):
        return PyPIConGPUFilteredSpecies(
            species=self.species.get_as_pypicongpu(), functor=self.functor.get_as_pypicongpu(mode=mode)
        )
