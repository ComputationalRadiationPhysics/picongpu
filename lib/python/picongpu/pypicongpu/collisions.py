"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from itertools import chain
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)

from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.species.species import Species
from picongpu.pypicongpu.util import alt, unique


class ConstLogCollision(BaseModel):
    type_constlog: Literal[True] = True
    coulomb_log: float


class DynamicLogCollision(BaseModel):
    type_dynamiclog: Literal[True] = True


CollisionFunctor = ConstLogCollision | DynamicLogCollision


def species(s: Species | FilteredSpecies):
    return alt(lambda: s.species, lambda: s)


def functor(s: Species | FilteredSpecies):
    return alt(lambda: s.functor, None)


class Collision(BaseModel):
    species_pairs: list[tuple[Species | FilteredSpecies, Species | FilteredSpecies]]
    functor: CollisionFunctor

    @field_validator("species_pairs", mode="after")
    @classmethod
    def _validate_species_pairs(cls, pairs):
        invalid_pairs = [
            pair for pair in pairs if species(pair[0]) == species(pair[1]) and functor(pair[0]) != functor(pair[1])
        ]
        if invalid_pairs:
            raise ValueError(
                f"Intra-species collisions with differently filtered species are not supported by PIConGPU. You gave: {invalid_pairs=}."
            )
        return pairs

    @computed_field
    def species(self) -> list[Species]:
        return unique(sum(self.species_pairs, tuple()))

    @computed_field
    def has_filters(self) -> bool:
        return any(isinstance(s, FilteredSpecies) for p in self.species_pairs for s in p)

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


def split_into_single(collision):
    return (Collision(species_pairs=[pair], functor=collision.functor) for pair in collision.species_pairs)


class CollisionalPhysicsSetup(BaseModel):
    collisions: list[Collision] = Field(default_factory=list)
    screening_species: list[Species | FilteredSpecies] = Field(default_factory=list)
    numerics_config: CollisionNumericsConfig = CollisionNumericsConfig()

    @field_validator("collisions", mode="after")
    @classmethod
    def _validate_collisions(cls, collisions):
        # Applying filters inside of the collision pipeline has a weird syntax
        # which makes it pretty hard to apply arbitrary filters to individual pairs.
        # What we do instead is that we split each collision to only hold a single pair.
        return list(chain(*map(split_into_single, collisions)))

    @computed_field
    def num_tmp_field_slots(self) -> int:
        if len(self.screening_species) == 0:
            return 1
        if len(self.screening_species) == 1:
            return 2
        return 3
