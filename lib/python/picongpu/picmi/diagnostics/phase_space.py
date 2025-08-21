"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.phase_space import PhaseSpace as PyPIConGPUPhaseSpace
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies

from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec

import typeguard
from typing import Literal


@typeguard.typechecked
class PhaseSpace:
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

    def check(self):
        if self.min_momentum >= self.max_momentum:
            raise ValueError("min_momentum must be less than max_momentum")

    def __init__(
        self,
        species: PICMISpecies,
        period: TimeStepSpec,
        spatial_coordinate: Literal["x", "y", "z"],
        momentum_coordinate: Literal["px", "py", "pz"],
        min_momentum: float,
        max_momentum: float,
    ):
        self.species = species
        self.period = period
        self.spatial_coordinate = spatial_coordinate
        self.momentum_coordinate = momentum_coordinate
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum

    def get_as_pypicongpu(
        # to get the corresponding PyPIConGPUSpecies instance for the given PICMISpecies.
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size,
        num_steps,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUPhaseSpace:
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        # checks if PICMISpecies instance exists in the dictionary. If yes, it returns the corresponding PyPIConGPUSpecies instance.
        # self.species refers to the species attribute of the class  PhaseSpace(picmistandard.PICMI_PhaseSpace).
        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        pypicongpu_phase_space = PyPIConGPUPhaseSpace()
        pypicongpu_phase_space.species = pypicongpu_species
        pypicongpu_phase_space.period = self.period.get_as_pypicongpu(time_step_size, num_steps)
        pypicongpu_phase_space.spatial_coordinate = self.spatial_coordinate
        pypicongpu_phase_space.momentum_coordinate = self.momentum_coordinate
        pypicongpu_phase_space.min_momentum = self.min_momentum
        pypicongpu_phase_space.max_momentum = self.max_momentum

        return pypicongpu_phase_space
