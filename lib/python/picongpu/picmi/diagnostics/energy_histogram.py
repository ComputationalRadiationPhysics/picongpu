"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .auto import Auto
from ...pypicongpu.output.energy_histogram import EnergyHistogram as PyPIConGPUEnergyHistogram
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies
from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec
import typeguard
import warnings
from typing import Optional, Dict, Union
import logging

# Set up logging for debugging
logger = logging.getLogger(__name__)


@typeguard.typechecked
class EnergyHistogram(Auto):
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

    def __init__(
        self,
        species: PICMISpecies,
        period: Optional[Union[int, TimeStepSpec]] = None,
        bin_count: int = 100,
        min_energy: Optional[float] = 0.0,
        max_energy: float = 1000.0,
        **kw,
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
        super().__init__(period=self.period, **kw)
        self.check()

    def check(self):
        super().check()
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

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: Dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size: float,
        num_steps: int,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUEnergyHistogram:
        """
        Convert to PyPIConGPU EnergyHistogram.

        :param dict_species_picmi_to_pypicongpu: Dict mapping PICMI species to PyPIConGPU species.
        :param time_step_size: Size of one time step in seconds (must be positive).
        :param num_steps: Total number of simulation steps (must be positive).
        :param simulation_box: Not used, included for compatibility.
        :return: PyPIConGPUEnergyHistogram with converted attributes.
        """
        logger.debug(f"Converting EnergyHistogram for species {self.species.name} with period {self.period}")
        self.check()
        if time_step_size <= 0:
            raise ValueError("time_step_size must be positive")
        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species.name} is not known to Simulation")

        # Ensure period is converted and validated
        try:
            period = (
                self.period.get_as_pypicongpu(time_step_size, num_steps)
                if isinstance(self.period, TimeStepSpec)
                else TimeStepSpec().get_as_pypicongpu(time_step_size, num_steps)
            )
            logger.debug(f"Converted period: {period.get_rendering_context()}")
        except ValueError as e:
            logger.error(f"Period conversion failed: {str(e)}")
            raise

        pypicongpu_energy_histogram = PyPIConGPUEnergyHistogram(
            species=dict_species_picmi_to_pypicongpu[self.species],
            period=period,
            bin_count=self.bin_count,
            min_energy=self.min_energy,
            max_energy=self.max_energy,
        )
        logger.debug(f"Created PyPIConGPUEnergyHistogram: {pypicongpu_energy_histogram.get_rendering_context()}")
        return pypicongpu_energy_histogram
