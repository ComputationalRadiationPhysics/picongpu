"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ... import util
from ...species import Species
from .source_base import SourceBase
import typeguard
import typing


@typeguard.typechecked
class Density(SourceBase):
    species = util.build_typesafe_property(Species)
    filter = util.build_typesafe_property(str)

    def __init__(self, species: Species, filter: str = "all"):
        self.species = species
        self.filter = filter
        self.check()

    def check(self) -> None:
        # Validate parameters
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if not isinstance(self.species, Species):
            raise ValueError(f"Species must be a Species, got {type(self.species)}")

    def _get_serialized(self) -> typing.Dict:
        # Return serialized representation
        self.check()
        return {
            "species": self.species.get_rendering_context(),
            "filter": self.filter,
        }
