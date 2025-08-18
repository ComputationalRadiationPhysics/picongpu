"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.energy_histogram import (
    EnergyHistogram as PyPIConGPUEnergyHistogram,
)
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies
from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec

import typeguard
import warnings
from typing import Optional, Dict, Union


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
    period: int or TimeStepSpec, optional
        Number of simulation steps between consecutive counts (e.g., 10 for every 10 steps).
        Use 0 to disable counting.
        Alternatively, a TimeStepSpec can be provided for PIConGPU-specific step selection
        (e.g., TimeStepSpec([5, 10]), TimeStepSpec([slice(-10, None, 1)])).
        Unit: steps or seconds (via TimeStepSpec unit).
    bin_count: int
        Number of bins for the energy histogram. Must be positive.
    min_energy: float, optional
        Minimum value for the energy histogram range. Default: 0.0.
        Unit: keV
    max_energy: float
        Maximum value for the energy histogram range. Must be greater than min_energy.
        Unit: keV
    """

    def check(self):
        if not isinstance(self.species, PICMISpecies):
            raise TypeError("species must be a Species")
        if self.bin_count <= 0:
            raise ValueError("bin_count must be > 0")
        if self.min_energy is not None and self.max_energy is not None and self.min_energy >= self.max_energy:
            raise ValueError("min_energy must be less than max_energy")
        if (
            self.period is not None
            and isinstance(self.period, TimeStepSpec)
            and not self.period.get_as_pypicongpu(1.0, 200).get_rendering_context().get("specs", [])
        ):
            warnings.warn("EnergyHistogram is disabled because period is set to 0 or an empty TimeStepSpec")

    def __init__(
        self,
        species: PICMISpecies,
        period: Optional[Union[int, TimeStepSpec]] = None,
        bin_count: int = 100,
        min_energy: Optional[float] = 0.0,
        max_energy: float = 1000.0,
    ):
        if period is not None and not isinstance(period, (int, TimeStepSpec)):
            raise TypeError("period must be an integer or TimeStepSpec")
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = TimeStepSpec([slice(None, None, period)]) if period > 0 else TimeStepSpec()
        else:
            self.period = period if period is not None else TimeStepSpec()
        self.species = species
        self.bin_count = bin_count
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.check()

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: Dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size: float,
        num_steps: int,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUEnergyHistogram:
        self.check()
        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species.name} is not known to Simulation")

        pypicongpu_energy_histogram = PyPIConGPUEnergyHistogram()
        pypicongpu_energy_histogram.species = dict_species_picmi_to_pypicongpu[self.species]
        pypicongpu_energy_histogram.period = self.period.get_as_pypicongpu(time_step_size, num_steps)
        pypicongpu_energy_histogram.bin_count = self.bin_count
        pypicongpu_energy_histogram.min_energy = self.min_energy
        pypicongpu_energy_histogram.max_energy = self.max_energy
        pypicongpu_energy_histogram._name = "energyhistogram"

        return pypicongpu_energy_histogram
