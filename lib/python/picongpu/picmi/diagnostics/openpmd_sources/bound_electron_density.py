"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ....pypicongpu.output.openpmd_sources import BoundElectronDensity as PyPIConGPUBoundElectronDensity
from ...species import Species as PICMISpecies
import typeguard
import typing


@typeguard.typechecked
class BoundElectronDensity(SourceBase):
    """
    Bound electron density data source for openPMD output

    This source calculates the density of bound electrons from a specified particle species,
    optionally filtered by a selection criterion, for particle-in-cell simulations.

    @param species Particle species contributing to the bound electron density (e.g., ions).
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
    ) -> PyPIConGPUBoundElectronDensity:
        """
        Convert this BoundElectronDensity source to a PyPIConGPU BoundElectronDensity source.

        @param dict_species_picmi_to_pypicongpu Mapping of PICMI species to PyPIConGPU species.
        @return A PyPIConGPU BoundElectronDensity instance with the same filter and species.
        @throw ValueError If the species is not known to the simulation or not mapped to a PyPIConGPUSpecies.
        """
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        return PyPIConGPUBoundElectronDensity(filter=self.filter, species=pypicongpu_species)
