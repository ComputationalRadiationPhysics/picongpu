"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
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
from typing import Union


@typeguard.typechecked
class EnergyHistogram:
    """
    Specifies the parameters for the output of Energy Histogram of species such as electrons.

    This plugin extracts energy histogram data from the simulation, allowing
    for detailed analysis of energy distributions of particles.

    Parameters
    ----------
    species: PICMISpecies
        Particle species to count (e.g., an instance with name="electron" or "proton").
    period: int or TimeStepSpec
        Number of simulation steps between consecutive outputs (e.g., 10 for every 10 steps).
        Use 0 to disable output. Alternatively, a TimeStepSpec can be provided for
        PyPIConGPU-specific step selection (e.g., TimeStepSpec([slice(0, None, 10)])).
    bin_count: int
        Number of bins for the energy histogram. Must be positive.
    min_energy: float
        Minimum value for the energy histogram range.
        Unit: keV
    max_energy: float
        Maximum value for the energy histogram range. Must be greater than min_energy.
        Unit: keV
    """

    def __init__(
        self,
        species: PICMISpecies,
        period: Union[int, TimeStepSpec],
        bin_count: int,
        min_energy: float,
        max_energy: float,
    ):
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = TimeStepSpec([slice(None, None, period)]) if period > 0 else TimeStepSpec()
        else:
            self.period = period
        self.species = species
        self.bin_count = bin_count
        self.min_energy = min_energy
        self.max_energy = max_energy

    def check(self):
        if not isinstance(self.species, PICMISpecies):
            raise TypeError("species must be a PICMISpecies")
        if not isinstance(self.species.name, str) or not self.species.name:
            raise TypeError("species must have a non-empty name")
        if not isinstance(self.period, TimeStepSpec):
            raise TypeError("period must be a TimeStepSpec")
        if self.bin_count <= 0:
            raise ValueError("bin_count must be > 0")
        if self.min_energy >= self.max_energy:
            raise ValueError("min_energy must be less than max_energy")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size: float,
        num_steps: int,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUEnergyHistogram:
        self.check()
        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species} is not known to Simulation")
        pypicongpu_species = dict_species_picmi_to_pypicongpu[self.species]
        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")
        pypicongpu_energy_histogram = PyPIConGPUEnergyHistogram(
            species=pypicongpu_species,
            period=self.period.get_as_pypicongpu(time_step_size, num_steps),
            bin_count=self.bin_count,
            min_energy=self.min_energy,
            max_energy=self.max_energy,
        )
        return pypicongpu_energy_histogram
