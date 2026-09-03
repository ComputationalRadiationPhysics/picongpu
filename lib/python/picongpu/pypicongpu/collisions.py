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
    field_validator,
    model_validator,
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


class SpeciesPair(BaseModel):
    """
    Two species that collide with each other.

    A pair may also be given as a bare two-element ``(species_lhs, species_rhs)``
    sequence, which is accepted for convenience.
    """

    species_lhs: Species | FilteredSpecies
    species_rhs: Species | FilteredSpecies

    @model_validator(mode="before")
    @classmethod
    def _from_pair(cls, data):
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return {"species_lhs": data[0], "species_rhs": data[1]}
        return data

    @model_validator(mode="after")
    def _validate_intra_species_filters(self):
        if species(self.species_lhs) == species(self.species_rhs) and functor(self.species_lhs) != functor(
            self.species_rhs
        ):
            raise ValueError(
                "Intra-species collisions with differently filtered species are not"
                " supported by PIConGPU. You gave: "
                f"{self=}."
            )
        return self


class Collision(BaseModel):
    species_pairs: list[SpeciesPair]
    functor: CollisionFunctor

    @computed_field
    def species(self) -> list[Species]:
        return unique([s for p in self.species_pairs for s in (p.species_lhs, p.species_rhs)])

    @computed_field
    def has_filters(self) -> bool:
        return any(isinstance(s, FilteredSpecies) for p in self.species_pairs for s in (p.species_lhs, p.species_rhs))


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
