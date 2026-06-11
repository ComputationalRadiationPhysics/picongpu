"""
# SPDX-FileCopyrightText: Masoud Afshari, Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from picongpu.picmi.copy_attributes import default_converts_to
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.phase_space import PhaseSpace as PyPIConGPUPhaseSpace


@default_converts_to(PyPIConGPUPhaseSpace)
class PhaseSpace(BaseModel):
    """
    Specifies the parameters for the output of Phase Space of species such as electrons.

    This plugin extracts phase-space data from the simulation, allowing
    for detailed analysis of particle distributions in position-momentum space.

    Parameters
    ----------
    species: string
        Name of the particle species to track (e.g., "electron", "proton").

    period: TimeStepSpec
        Specify on which time steps the plugin should run.
        Unit: steps (simulation time steps).

    spatial_coordinate: string
        Spatial coordinate used in phase space (e.g., 'x', 'y', 'z').

    momentum: string
        Momentum coordinate used in phase space (e.g., 'px', 'py', 'pz').

    min_momentum: float
        Minimum value for the phase-space coordinate range.
        Unit: kg*m/s (momentum in SI units).

    max_momentum: float
        Maximum value for the phase-space coordinate range.
        Unit: kg*m/s (momentum in SI units).

    name: string, optional
        Optional name for the phase-space plugin.
    """

    species: Species | FilteredSpecies
    period: TimeStepSpec
    spatial_coordinate: Literal["x", "y", "z"]
    momentum_coordinate: Literal["px", "py", "pz"]
    min_momentum: float
    max_momentum: float

    def check(self, *args, **kwargs):
        if self.min_momentum >= self.max_momentum:
            raise ValueError("min_momentum must be less than max_momentum")

    model_config = ConfigDict(arbitrary_types_allowed=True)
