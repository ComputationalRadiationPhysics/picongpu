"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from itertools import combinations, combinations_with_replacement

from pydantic import BaseModel, Field, model_validator, field_validator

from picongpu.picmi.species import Species
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.pypicongpu.collisions import Collision as PyPIConGPUCollision
from picongpu.pypicongpu.collisions import CollisionalPhysicsSetup as PyPIConGPUCollisionalPhysicsSetup
from picongpu.pypicongpu.collisions import CollisionFunctor
from picongpu.pypicongpu.collisions import CollisionNumericsConfig as CollisionNumericsConfig
from picongpu.pypicongpu.collisions import ConstLogCollision as ConstLogCollision
from picongpu.pypicongpu.collisions import DynamicLogCollision as DynamicLogCollision


class Collision(BaseModel):
    species_pairs: list[tuple[Species | FilteredSpecies, Species | FilteredSpecies]]
    functor: CollisionFunctor

    @classmethod
    def construct_from_pairs(cls, species_pairs, **kwargs):
        """Construct from Collision from pairs. Same as normal constructor."""
        return cls(species_pairs=species_pairs, **kwargs)

    @classmethod
    def construct_one_to_all(cls, one, to_all_of, **kwargs):
        """Construct collision of one species with all of the `to_all_of` species."""
        return cls(species_pairs=[(one, a) for a in to_all_of], **kwargs)

    @classmethod
    def construct_all_to_all(cls, species, include_self_collisions=True, **kwargs):
        """Construct collisions among all the given species."""
        combine = combinations_with_replacement if include_self_collisions else combinations
        return cls(species_pairs=list(combine(species, 2)), **kwargs)

    def get_as_pypicongpu(self):
        return PyPIConGPUCollision(
            species_pairs=[(lhs.get_as_pypicongpu(), rhs.get_as_pypicongpu()) for lhs, rhs in self.species_pairs],
            functor=self.functor,
        )


class CollisionalPhysicsSetup(BaseModel):
    collisions: list[Collision] = Field(default_factory=list)
    screening_species: list[Species | FilteredSpecies] = Field(default_factory=list)
    numerics_config: CollisionNumericsConfig = CollisionNumericsConfig()

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if "collisions" not in kwargs:
                kwargs["collisions"] = args[0]
                args = tuple()
            else:
                raise ValueError(f"Duplicated collisions argument given: You gave {args=} and {kwargs=}.")
        return super().__init__(*args, **kwargs)

    @field_validator("collisions", mode="before")
    @classmethod
    def _validate_collisions(cls, value):
        if isinstance(value, Collision):
            return [value]
        return value

    @model_validator(mode="after")
    def _validate(self):
        if (
            any(isinstance(collision.functor, DynamicLogCollision) for collision in self.collisions)
            and len(self.screening_species) == 0
        ):
            message = (
                "Your collisional physics setup is inconsistent: "
                "You requested a dynamic log for some of your collisions "
                "but didn't give any screening species to compute this from. "
                f"You gave:\n{self.collisions=}\nand\n{self.screening_species=}."
            )
            raise ValueError(message)
        return self

    def get_as_pypicongpu(self):
        return PyPIConGPUCollisionalPhysicsSetup(
            collisions=[c.get_as_pypicongpu() for c in self.collisions],
            screening_species=[s.get_as_pypicongpu() for s in self.screening_species],
            numerics_config=self.numerics_config,
        )
