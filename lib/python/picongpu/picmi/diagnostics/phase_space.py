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
import warnings
from typing import Literal, Optional, Union


@typeguard.typechecked
class PhaseSpace:
    """
    Specifies the parameters for the output of Phase Space of species such as electrons.

    This plugin extracts phase-space data from the simulation, allowing
    for detailed analysis of particle distributions in position-momentum space.

    Parameters
    ----------
    species: PICMISpecies
        Particle species to track (e.g., an instance with name="electron" or "proton").
    period: int or TimeStepSpec, optional
        Number of simulation steps between consecutive outputs (e.g., 10 for every 10 steps).
        Use 0 to disable output. Alternatively, a TimeStepSpec can be provided.
        Unit: steps (simulation time steps).
    spatial_coordinate: string
        Spatial coordinate used in phase space (e.g., 'x', 'y', 'z').
    momentum_coordinate: string
        Momentum coordinate used in phase space (e.g., 'px', 'py', 'pz').
    min_momentum: float
        Minimum value for the phase-space momentum range.
        Unit: kg*m/s (momentum in SI units).
    max_momentum: float
        Maximum value for the phase-space momentum range.
        Unit: kg*m/s (momentum in SI units).
    """

    def check(self):
        if self.min_momentum >= self.max_momentum:
            raise ValueError("min_momentum must be less than max_momentum")
        if self.period is not None and isinstance(self.period, TimeStepSpec):
            if not self.period.get_as_pypicongpu(1.0, 200).get_rendering_context().get("specs", []):
                warnings.warn("PhaseSpace is disabled because period is empty")

    def __init__(
        self,
        species: PICMISpecies,
        period: Optional[Union[int, TimeStepSpec]] = None,
        spatial_coordinate: Literal["x", "y", "z"] = "x",
        momentum_coordinate: Literal["px", "py", "pz"] = "px",
        min_momentum: float = 0.0,
        max_momentum: float = 1.0,
    ):
        if period is not None and not isinstance(period, (int, TimeStepSpec)):
            raise TypeError("period must be an integer or TimeStepSpec")
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = (
                TimeStepSpec([slice(None, None, period)])("steps") if period > 0 else TimeStepSpec([])("steps")
            )
        else:
            self.period = period if period is not None else TimeStepSpec([slice(0, None, 1)])("steps")
        self.species = species
        self.spatial_coordinate = spatial_coordinate
        self.momentum_coordinate = momentum_coordinate
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size: float,
        num_steps: int,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUPhaseSpace:
        self.check()
        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species} is not known to Simulation")
        pypicongpu_species = dict_species_picmi_to_pypicongpu[self.species]
        pypicongpu_phase_space = PyPIConGPUPhaseSpace()
        pypicongpu_phase_space.species = pypicongpu_species
        pypicongpu_phase_space.period = self.period.get_as_pypicongpu(time_step_size, num_steps)
        pypicongpu_phase_space.spatial_coordinate = self.spatial_coordinate
        pypicongpu_phase_space.momentum_coordinate = self.momentum_coordinate
        pypicongpu_phase_space.min_momentum = self.min_momentum
        pypicongpu_phase_space.max_momentum = self.max_momentum
        return pypicongpu_phase_space
