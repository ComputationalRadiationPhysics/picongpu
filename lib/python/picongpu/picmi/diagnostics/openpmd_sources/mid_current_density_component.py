"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ....pypicongpu.output.openpmd_sources import MidCurrentDensityComponent as PyPIConGPUMidCurrentDensityComponent
from ...species import Species as PICMISpecies
import typeguard
import typing


@typeguard.typechecked
class MidCurrentDensityComponent(SourceBase):
    """
    Current density component data source for openPMD output

    Derives a scalar field of current density (in A/m^2) in a specified direction (x=0, y=1, z=2)
    for a specified particle species, optionally filtered, in particle-in-cell simulations. Uses
    weighting, position, momentum, mass, and charge attributes, mapped to cells by the PIC code's
    spatial shape. Intended for debugging (e.g., validating current solvers).

    @param species Particle species to calculate current density for (e.g., electrons, ions).
        Must have weighting, position, momentum, mass, and charge attributes.
    @param filter Name of a filter to select particles. Default: "all".
    @param direction Direction of current density (0=x, 1=y, 2=z).
    """

    def __init__(self, species: PICMISpecies, filter: str = "all", direction: int = 0):
        self.species = species
        self.filter = filter
        self.direction = direction
        self.check()

    def check(self) -> None:
        """
        Validate the parameters.

        @throw ValueError If filter is not a string, species is not a PICMISpecies, or direction is invalid.
        """
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if not isinstance(self.species, PICMISpecies):
            raise ValueError(f"Species must be a PICMISpecies, got {type(self.species)}")
        if not isinstance(self.direction, int):
            raise ValueError(f"Direction must be an integer, got {type(self.direction)}")
        if self.direction not in [0, 1, 2]:
            raise ValueError(f"Direction must be 0 (x), 1 (y), or 2 (z), got {self.direction}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, typing.Any],
    ) -> PyPIConGPUMidCurrentDensityComponent:
        """
        Convert to a PyPIConGPU MidCurrentDensityComponent source.

        @param dict_species_picmi_to_pypicongpu Mapping of PICMI to PyPIConGPU species.
        @return A PyPIConGPU MidCurrentDensityComponent instance with the same filter, species, and direction.
        @throw ValueError If species is unknown or unmapped to a PyPIConGPUSpecies.
        """
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        return PyPIConGPUMidCurrentDensityComponent(
            filter=self.filter,
            species=pypicongpu_species,
            direction=self.direction,
        )
