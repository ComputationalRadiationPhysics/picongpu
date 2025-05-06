"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ...pypicongpu.output.openpmd_source import MacroCounter as PyPIConGPUMacroCounter
from ...species import Species as PICMISpecies
import typeguard
import typing


@typeguard.typechecked
class MacroCounter(SourceBase):
    """
    Macro-particle counter data source for openPMD output

    Derives a scalar field counting macro-particles per cell for a specified particle species,
    optionally filtered, in particle-in-cell simulations. Assigns each macro-particle directly
    to its cell via floor operation. Intended for debugging (e.g., validating particle memory).

    @param species Particle species to count (e.g., electrons, ions).
    @param filter Name of a filter to select particles. Default: "all".
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
    ) -> PyPIConGPUMacroCounter:
        """
        Convert to a PyPIConGPU MacroCounter source.

        @param dict_species_picmi_to_pypicongpu Mapping of PICMI to PyPIConGPU species.
        @return A PyPIConGPU MacroCounter instance with the same filter and species.
        @throw ValueError If species is unknown or unmapped to a PyPIConGPUSpecies.
        """
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        return PyPIConGPUMacroCounter(filter=self.filter, species=pypicongpu_species)
