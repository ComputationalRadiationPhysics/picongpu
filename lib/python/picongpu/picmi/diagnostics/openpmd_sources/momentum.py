"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from picongpu.pypicongpu.output.openpmd_sources import Momentum as PyPIConGPUMomentum
from picongpu.picmi.species import Species as PICMISpecies
import typing
import typeguard


@typeguard.typechecked
class Momentum(SourceBase):
    """
    Momentum component data source for openPMD output in PIConGPU.

    Derives a scalar field of momentum (in kg·m/s) in a specified direction (x, y, z)
    for a specified particle species, optionally filtered, in particle-in-cell simulations.
    Uses weighting and momentum attributes, mapped to cells by the PIC code's spatial shape.
    Intended for debugging or analyzing particle dynamics.
    """

    def __init__(self, species: PICMISpecies, filter: str = "species_all", direction: str = "x"):
        self.species = species
        self._filter = filter
        self.direction = direction
        self.check()

    @property
    def filter(self) -> str:
        return self._filter

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        valid_directions = ["x", "y", "z"]
        if not isinstance(self.species, PICMISpecies):
            raise TypeError(f"Species must be a PICMISpecies, got {type(self.species)}")
        if not isinstance(self._filter, str):
            raise TypeError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")
        if not isinstance(self.direction, str):
            raise TypeError(f"Direction must be a string, got {type(self.direction)}")
        if self.direction not in valid_directions:
            raise ValueError(f"Direction must be 'x', 'y', or 'z', got {self.direction}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any],
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box=None,
    ) -> PyPIConGPUMomentum:
        self.check()
        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species.name} is not known to Simulation")
        return PyPIConGPUMomentum(
            filter=self._filter,
            species=dict_species_picmi_to_pypicongpu[self.species],
            direction=self.direction,
        )
