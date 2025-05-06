"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ...pypicongpu.output.openpmd_source import Counter as PyPIConGPUCounter
from ...species import Species as PICMISpecies
import typeguard
import typing


@typeguard.typechecked
class Counter(SourceBase):
    """
    Particle counter data source for openPMD output

    This source derives a scalar field representing the number of real particles per cell
    for a specified species, optionally filtered by a selection criterion, in particle-in-cell
    simulations. The particle count is based on the species' weighting attribute and assigned
    directly to the cell containing each particle. Intended primarily for debugging due to
    its non-physical deposition shape, which differs from standard charge or momentum-conserving
    assignments.

    @param species Particle species to count (e.g., electrons, ions). Must have a weighting attribute.
    @param filter Name of a filter to select particles contributing to the source.
        Default: "all" (includes all particles of the specified species).
    """

    def __init__(self, species: PICMISpecies, filter: str = "all"):
        self.species = species
        self.filter = filter
        self.check()

    def check(self) -> None:
        """
        Validate the parameters.

        @throw ValueError If filter is not a string or species is not a PICMISpecies.
        """
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if not isinstance(self.species, PICMISpecies):
            raise ValueError(f"Species must be a PICMISpecies, got {type(self.species)}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, typing.Any],
    ) -> PyPIConGPUCounter:
        """
        Convert this Counter source to a PyPIConGPU Counter source.

        @param dict_species_picmi_to_pypicongpu Mapping of PICMI species to PyPIConGPU species.
        @return A PyPIConGPU Counter instance with the same filter and species.
        @throw ValueError If the species is not known to the simulation or not mapped to a PyPIConGPUSpecies.
        """
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        return PyPIConGPUCounter(filter=self.filter, species=pypicongpu_species)
