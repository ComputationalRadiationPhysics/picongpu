"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, ConfigDict

from picongpu.picmi.copy_attributes import default_converts_to
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.energy_histogram import EnergyHistogram as PyPIConGPUEnergyHistogram


@default_converts_to(PyPIConGPUEnergyHistogram)
class EnergyHistogram(BaseModel):
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

    def check(self, *args, **kwargs):
        if self.min_energy >= self.max_energy:
            raise ValueError("min_energy must be less than max_energy")
        if self.bin_count <= 0:
            raise ValueError("bin_count must be > 0")

    species: Species | FilteredSpecies
    period: TimeStepSpec
    bin_count: int
    min_energy: float
    max_energy: float

    model_config = ConfigDict(arbitrary_types_allowed=True)
