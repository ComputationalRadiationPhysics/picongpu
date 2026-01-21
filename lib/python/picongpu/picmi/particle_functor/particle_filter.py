"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any, Callable

from pydantic import BaseModel

from picongpu.picmi.copy_attributes import default_converts_to
from picongpu.picmi.particle_functor.particle_functor import Particle, ParticleFunctor
from picongpu.picmi.species import Species
from picongpu.pypicongpu.particle_functor import FilteredSpecies as PyPIConGPUFiltereSpecies


class ParticleFilter(ParticleFunctor):
    def __init__(self, name: str, functor: Callable[[Particle], Any]):
        return super().__init__(name=name, functor=functor, return_type=bool, unit_dimension=None)

    def get_as_pypicongpu(self, mode="Filter"):
        return super().get_as_pypicongpu(mode=mode)


@default_converts_to(PyPIConGPUFiltereSpecies)
class FilteredSpecies(BaseModel):
    species: Species
    functor: ParticleFilter

    class Config:
        arbitrary_types_allowed = True
