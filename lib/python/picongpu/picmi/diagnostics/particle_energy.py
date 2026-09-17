"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, ConfigDict

from picongpu.picmi.copy_attributes import default_converts_to
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.particle_energy import ParticleEnergy as PyPIConGPUParticleEnergy


@default_converts_to(PyPIConGPUParticleEnergy)
class ParticleEnergy(BaseModel):
    """
    Specifies the parameters for the per-species particle-energy diagnostic.

    This plugin computes the kinetic and total energy of the particles of a given
    species at the given time steps and writes it to
    ``<species>_energy_<filter>.dat``.

    Parameters
    ----------
    species: string
        Name of the particle species to track (e.g., "electron", "proton").

    period: int
        Number of simulation steps between consecutive outputs.
        Unit: steps (simulation time steps).

    name: string, optional
        Optional name for the particle energy plugin.
    """

    species: Species | FilteredSpecies
    period: TimeStepSpec

    model_config = ConfigDict(arbitrary_types_allowed=True)
