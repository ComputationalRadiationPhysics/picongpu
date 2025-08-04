"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .timestepspec import TimeStepSpec
from ...pypicongpu.output.energy_histogram import (
    EnergyHistogram as PyPIConGPUEnergyHistogram,
)
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies


from ..species import Species as PICMISpecies

import typeguard


@typeguard.typechecked
class EnergyHistogram:
    """
    Specifies the parameters for the output of Energy Histogram of species such as electrons.

    This plugin extracts energy histogram data from the simulation, allowing
    for detailed analysis of energy distributions of particles.

    Parameters
    ----------
    species: string
        Name of the particle species to track (e.g., "electron", "proton").

    period: int
        Number of simulation steps between consecutive outputs.
        If set to a non-zero value, the energy histogram of all electrons is computed.
        By default, the value is 0 and no histogram for the electrons is computed.
        Unit: steps (simulation time steps).

    bin_count: int
        Number of bins for the energy histogram.

    min_energy: float
        Minimum value for the energy histogram range.
        Unit: keV
        Default is 0, meaning 0 keV.

    max_energy: float
        Maximum value for the energy histogram range.
        Unit: keV
        There is no default value.

    name: string, optional
        Optional name for the energy histogram plugin.
    """

    def check(self):
        if self.min_energy >= self.max_energy:
            raise ValueError("min_energy must be less than max_energy")
        if self.bin_count <= 0:
            raise ValueError("bin_count must be > 0")

    def __init__(
        self,
        species: PICMISpecies,
        period: TimeStepSpec,
        bin_count: int,
        min_energy: float,
        max_energy: float,
    ):
        self.species = species
        self.period = period
        self.bin_count = bin_count
        self.min_energy = min_energy
        self.max_energy = max_energy

    def get_as_pypicongpu(
        # to get the corresponding PyPIConGPUSpecies instance for the given PICMISpecies.
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size,
        num_steps,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUEnergyHistogram:
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        # checks if PICMISpecies instance exists in the dictionary. If yes, it returns the corresponding PyPIConGPUSpecies instance.
        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        pypicongpu_energy_histogram = PyPIConGPUEnergyHistogram()
        pypicongpu_energy_histogram.species = pypicongpu_species
        pypicongpu_energy_histogram.period = self.period.get_as_pypicongpu(time_step_size, num_steps)
        pypicongpu_energy_histogram.bin_count = self.bin_count
        pypicongpu_energy_histogram.min_energy = self.min_energy
        pypicongpu_energy_histogram.max_energy = self.max_energy

        return pypicongpu_energy_histogram
