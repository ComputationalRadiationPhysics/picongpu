"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from ...pypicongpu.output.openpmd_source import ChargeDensity as PyPIConGPUChargeDensity
from ...species import Species as PICMISpecies
import typeguard
import typing


@typeguard.typechecked
class ChargeDensity(SourceBase):
    """
    Represents a charge density data source for openPMD output in particle-in-cell simulations.

    This source calculates the charge density from a specified particle species, optionally
    filtered by a selection criterion.

    Parameters
    ----------
    species: PICMISpecies
        Particle species contributing to the charge density (e.g., electrons, protons).

    filter: str, optional
        Name of a filter to select particles contributing to the source.
        Default: "all" (includes all particles of the specified species).
    """

    def __init__(self, species: PICMISpecies, filter: str = "all"):
        self.species = species
        self.filter = filter
        self.check()

    def check(self) -> None:
        """
        Validate the parameters.

        Raises
        ------
        ValueError
            If filter is not a string or species is not a PICMISpecies.
        """
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if not isinstance(self.species, PICMISpecies):
            raise ValueError(f"Species must be a PICMISpecies, got {type(self.species)}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, typing.Any],
        time_step_size: float,
        num_steps: int,
    ) -> PyPIConGPUChargeDensity:
        """
        Convert this ChargeDensity source to a PyPIConGPU ChargeDensity source.

        Parameters
        ----------
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, Any]
            Mapping of PICMI species to PyPIConGPU species.
        time_step_size: float
            Size of a simulation time step (unused).
        num_steps: int
            Total number of simulation steps (unused).

        Returns
        -------
        PyPIConGPUChargeDensity
            A PyPIConGPU ChargeDensity instance with the same filter and species.

        Raises
        ------
        ValueError
            If the species is not known to the simulation or not mapped to a PyPIConGPUSpecies.
        """
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        return PyPIConGPUChargeDensity(filter=self.filter, species=pypicongpu_species)
