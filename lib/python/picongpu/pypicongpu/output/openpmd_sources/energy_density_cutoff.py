"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ... import util
from ...species import Species
from .source_base import SourceBase
import typeguard
import typing


@typeguard.typechecked
class EnergyDensityCutoff(SourceBase):
    species = util.build_typesafe_property(Species)
    filter = util.build_typesafe_property(str)
    cutoff_max_energy = util.build_typesafe_property(typing.Optional[float])

    def __init__(self, species: Species, filter: str = "species_all", cutoff_max_energy: typing.Optional[float] = None):
        self.species = species
        self.filter = filter
        self.cutoff_max_energy = cutoff_max_energy
        self.check()

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if self.filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self.filter}")
        if not isinstance(self.species, Species):
            raise ValueError(f"Species must be a Species, got {type(self.species)}")
        if self.cutoff_max_energy is not None and not isinstance(self.cutoff_max_energy, (int, float)):
            raise ValueError(f"cutoff_max_energy must be a number or None, got {type(self.cutoff_max_energy)}")
        if self.cutoff_max_energy is not None and self.cutoff_max_energy <= 0:
            raise ValueError(f"cutoff_max_energy must be positive, got {self.cutoff_max_energy}")

    def _get_serialized(self) -> typing.Dict:
        self.check()
        return {
            "type": "energydensitycutoff",
            "species": self.species.get_rendering_context(),
            "filter": self.filter,
            "cutoff_max_energy": self.cutoff_max_energy,
        }
