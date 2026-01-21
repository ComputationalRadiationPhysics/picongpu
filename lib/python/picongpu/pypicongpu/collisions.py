"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, computed_field, field_serializer

from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.rendering.renderedobject import SelfRegisteringRenderedObject
from picongpu.pypicongpu.species.species import Species
from picongpu.pypicongpu.util import unique


class _CollisionFunctor(SelfRegisteringRenderedObject, BaseModel):
    pass


class ConstLogCollision(_CollisionFunctor):
    _name: str = PrivateAttr("constlog")
    coulomb_log: float


class DynamicLogCollision(_CollisionFunctor):
    _name: str = PrivateAttr("dynamiclog")

    # Our current context validation doesn't like empty leafs.
    # This puts some unused content into the context.
    # Ain't pretty but was the fastest solution.
    # General overhaul of rendering is on the agenda anyways.
    @computed_field
    def unused(self) -> str:
        return ""


CollisionFunctor = ConstLogCollision | DynamicLogCollision


class Collision(BaseModel):
    species_pairs: list[tuple[Species, Species]]
    functor: CollisionFunctor

    @computed_field
    def species(self) -> list[Species]:
        return unique(sum(self.species_pairs, tuple()))

    @field_serializer("species_pairs", mode="plain")
    def _species_pairs_serializer(self, value):
        return [
            {"species_lhs": pair[0].model_dump(mode="json"), "species_rhs": pair[1].model_dump(mode="json")}
            for pair in value
        ]

    @field_serializer("functor")
    def _serialize_functor(self, value):
        return value.get_rendering_context()


class CollisionNumericsConfig(BaseModel):
    precision: Literal[32, 64, "X"] = 64
    cell_list_chunk_size: int | None = None
    debug_screening_length: bool = False


class CollisionalPhysicsSetup(BaseModel):
    collisions: list[Collision] = Field(default_factory=list)
    screening_species: list[Species | FilteredSpecies] = Field(default_factory=list)
    numerics_config: CollisionNumericsConfig = CollisionNumericsConfig()

    @computed_field
    def num_tmp_field_slots(self) -> int:
        if len(self.screening_species) == 0:
            return 1
        if len(self.screening_species) == 1:
            return 2
        return 3
