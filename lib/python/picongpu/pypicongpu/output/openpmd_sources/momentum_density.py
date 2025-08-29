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
from typing import Literal


@typeguard.typechecked
class MomentumDensity(SourceBase):
    species = util.build_typesafe_property(Species)
    filter = util.build_typesafe_property(str)
    direction = util.build_typesafe_property(Literal["x", "y", "z"])

    def __init__(self, species: Species, filter: str = "species_all", direction: Literal["x", "y", "z"] = "x"):
        self.species = species
        self.filter = filter
        self.direction = direction
        self.check()

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if self.filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self.filter}")
        if not isinstance(self.species, Species):
            raise ValueError(f"Species must be a Species, got {type(self.species)}")
        if self.direction not in ["x", "y", "z"]:
            raise ValueError(f"Direction must be 'x', 'y', or 'z', got {self.direction}")

    def _get_serialized(self) -> typing.Dict:
        self.check()
        result = super()._get_serialized()
        result.update(
            {"species": self.species.get_rendering_context(), "filter": self.filter, "direction": self.direction}
        )
        return result
