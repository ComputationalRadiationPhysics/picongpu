"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any, Callable

from pydantic import BaseModel

from picongpu.picmi.particle_functor.particle_functor import Particle, ParticleFunctor
from picongpu.picmi.species import Species


class ParticleFilter(ParticleFunctor):
    def __init__(self, name: str, functor: Callable[[Particle], Any]):
        return super().__init__(name=name, functor=functor, return_type=bool, unit_dimension=None)

    def get_as_pypicongpu(self, mode="Filter"):
        return super().get_as_pypicongpu(mode=mode)


class FilteredSpecies(BaseModel):
    species: Species
    functor: ParticleFilter

    def get_as_pypicongpu(self):
        tmp = self.species.get_as_pypicongpu()
        tmp.name = f"{tmp.name}_{self.functor.name}"
        return tmp

    class Config:
        arbitrary_types_allowed = True
