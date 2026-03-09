"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, ConfigDict

from picongpu.picmi.copy_attributes import default_converts_to

from ...pypicongpu.output.macro_particle_count import (
    MacroParticleCount as PyPIConGPUMacroParticleCount,
)
from ..species import Species as Species
from .timestepspec import TimeStepSpec


@default_converts_to(PyPIConGPUMacroParticleCount)
class MacroParticleCount(BaseModel):
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

    species: Species
    period: TimeStepSpec

    model_config = ConfigDict(arbitrary_types_allowed=True)
