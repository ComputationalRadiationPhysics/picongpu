"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.macro_particle_count import (
    MacroParticleCount as PyPIConGPUMacroParticleCount,
)
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies

from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec

import typeguard


@typeguard.typechecked
class MacroParticleCount:
    """
    Specifies the parameters for counting the total number of macro particles of a given species.

    This plugin counts the total number of macro particles in the simulation,
    useful for tracking particle statistics and population dynamics.

    Parameters
    ----------
    species: string
        Name of the particle species to count (e.g., "electron", "proton").

    period: int
        Number of simulation steps between consecutive counts.
        Unit: steps (simulation time steps).

    name: string, optional
        Optional name for the macro particle count plugin.
    """

    def check(self):
        pass

    def __init__(self, species: PICMISpecies, period: TimeStepSpec):
        self.species = species
        self.period = period

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size,
        num_steps,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUMacroParticleCount:
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        pypicongpu_macro_count = PyPIConGPUMacroParticleCount()
        pypicongpu_macro_count.species = pypicongpu_species
        pypicongpu_macro_count.period = self.period.get_as_pypicongpu(time_step_size, num_steps)

        return pypicongpu_macro_count
