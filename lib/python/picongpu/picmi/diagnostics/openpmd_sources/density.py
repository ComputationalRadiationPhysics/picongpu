"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from picongpu.pypicongpu.output.openpmd_sources import Density as PyPIConGPUDensity
from ...species import Species as PICMISpecies
import typeguard
import typing


@typeguard.typechecked
class Density(SourceBase):
    """
    Particle density data source for openPMD output in PIConGPU.

    Derives a scalar field representing the number density (in m^-3) of a specified
    particle species, optionally filtered by a selection criterion.
    The density is calculated based on the species' weighting and position attributes
    and mapped to cells according to the PIC code's spatial shape assignment.
    """

    def __init__(self, species: PICMISpecies, filter: str = "species_all"):
        self.species = species
        self._filter = filter
        self.check()

    @property
    def filter(self) -> str:
        return self._filter

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self.species, PICMISpecies):
            raise TypeError(f"Species must be a PICMISpecies, got {type(self.species)}")
        if not isinstance(self._filter, str):
            raise TypeError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any],
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box=None,
    ) -> PyPIConGPUDensity:
        self.check()
        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species.name} is not known to Simulation")
        return PyPIConGPUDensity(filter=self._filter, species=dict_species_picmi_to_pypicongpu[self.species])
