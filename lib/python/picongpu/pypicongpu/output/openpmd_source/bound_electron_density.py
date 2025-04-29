"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...species import Species as PyPIConGPUSpecies
import typeguard
import typing


@typeguard.typechecked
class BoundElectronDensity:
    filter = property(lambda self: self._filter)
    species = property(lambda self: self._species)

    def __init__(self, filter: str, species: typing.List[PyPIConGPUSpecies]):
        self._filter = filter
        self._species = species
        self.check()

    def check(self) -> None:
        if not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string, got {type(self._filter)}")
        if not self._species or not all(isinstance(s, PyPIConGPUSpecies) for s in self._species):
            raise ValueError("Species must be a non-empty list of PyPIConGPUSpecies")

    def _get_serialized(self) -> typing.Dict:
        return {
            "dataset": "boundElectronDensity",
            "filter": self._filter,
            "species": [s.get_rendering_context() for s in self._species],
        }
