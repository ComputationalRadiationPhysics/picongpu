"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ....pypicongpu.output.openpmd_sources import EnergyDensityCutoff as PyPIConGPUEnergyDensityCutoff
from ...species import Species as PICMISpecies
import typeguard
import typing


@typeguard.typechecked
class EnergyDensityCutoff(SourceBase):
    """
    Kinetic energy density data source with cutoff for openPMD output

    Derives a scalar field of kinetic energy density (in J/m^3) for a specified particle species,
    optionally filtered, including only particles with kinetic energy below a user-defined cutoff,
    in particle-in-cell simulations. Uses weighting, momentum, and mass attributes, mapped to cells
    by the PIC code's spatial shape.

    @param species Particle species to calculate energy density for (e.g., electrons, ions).
        Must have weighting, momentum, and mass attributes.
    @param filter Name of a filter to select particles. Default: "all".
    @param cutoff_max_energy Maximum kinetic energy cutoff (in Joules).
    """

    def __init__(self, species: PICMISpecies, filter: str = "all", cutoff_max_energy: float = None):
        self.species = species
        self.filter = filter
        self.cutoff_max_energy = cutoff_max_energy
        self.check()

    def check(self) -> None:
        """
        Validate the parameters.

        @throw ValueError If filter is not a string, species is not a PICMISpecies, or cutoff_max_energy is invalid.
        """
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if not isinstance(self.species, PICMISpecies):
            raise ValueError(f"Species must be a PICMISpecies, got {type(self.species)}")
        if self.cutoff_max_energy is not None and not isinstance(self.cutoff_max_energy, (int, float)):
            raise ValueError(f"cutoff_max_energy must be a number or None, got {type(self.cutoff_max_energy)}")
        if self.cutoff_max_energy is not None and self.cutoff_max_energy <= 0:
            raise ValueError(f"cutoff_max_energy must be positive, got {self.cutoff_max_energy}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, typing.Any],
    ) -> PyPIConGPUEnergyDensityCutoff:
        """
        Convert to a PyPIConGPU EnergyDensityCutoff source.

        @param dict_species_picmi_to_pypicongpu Mapping of PICMI to PyPIConGPU species.
        @return A PyPIConGPU EnergyDensityCutoff instance with the same filter, species, and cutoff.
        @throw ValueError If species is unknown or unmapped to a PyPIConGPUSpecies.
        """
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        return PyPIConGPUEnergyDensityCutoff(
            filter=self.filter,
            species=pypicongpu_species,
            cutoff_max_energy=self.cutoff_max_energy,
        )
