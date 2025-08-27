"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from picongpu.pypicongpu.output.openpmd_sources import BoundElectronDensity as PyPIConGPUBoundElectronDensity
from ...species import Species as PICMISpecies
import typing
import typeguard


@typeguard.typechecked
class BoundElectronDensity(SourceBase):
    """
    Bound electron density diagnostic for PIConGPU.
    """

    def __init__(self, species: PICMISpecies, filter: typing.Optional[str] = "all"):
        self.species = species
        self._filter = filter
        self.check()

    @property
    def filter(self) -> typing.Optional[str]:
        return self._filter

    def check(self) -> None:
        if not isinstance(self.species, PICMISpecies):
            raise TypeError(f"Species must be a PICMISpecies, got {type(self.species)}")
        if self._filter is not None and not isinstance(self._filter, str):
            raise TypeError(f"Filter must be a string or None, got {type(self._filter)}")

    def get_as_pypicongpu(
        self, dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any]
    ) -> PyPIConGPUBoundElectronDensity:
        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species.name} is not known to Simulation")
        return PyPIConGPUBoundElectronDensity(
            filter=self._filter, species=dict_species_picmi_to_pypicongpu[self.species]
        )
