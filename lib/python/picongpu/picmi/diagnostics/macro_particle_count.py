"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
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
import warnings


@typeguard.typechecked
class MacroParticleCount:
    """
    Specifies the parameters for counting the total number of macro particles of a given species.

    This plugin counts the total number of macro particles in the simulation,
    useful for tracking particle statistics and population dynamics.

    Parameters
    ----------
    species: PICMISpecies
        Particle species to count (e.g., an instance with name="electron" or "proton").

    period: int or TimeStepSpec
        Number of simulation steps between consecutive counts (e.g., 10 for every 10 steps).
        Use 0 to disable counting.
        Alternatively, a TimeStepSpec can be provided for PIConGPU-specific step selection
        (e.g., TimeStepSpec[5, 10], TimeStepSpec[-10:]).
        Unit: steps (simulation time steps).
    """

    def check(self):
        if not self.period.get_as_pypicongpu(1.0, 100).get_rendering_context().get("specs", []):
            warnings.warn("MacroParticleCount is disabled because period is set to 0 or an empty TimeStepSpec")

    def __init__(self, species: PICMISpecies, period: int | TimeStepSpec):
        self.species = species
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = TimeStepSpec[::period] if period > 0 else TimeStepSpec()
        else:
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
