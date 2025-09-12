"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import typing
import typeguard

from .source_base import SourceBase
from ..species import Species as PICMISpecies
import pypicongpu.output.openpmd_sources as pypicongpu_sources


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@typeguard.typechecked
class SourceBaseSpeciesFilter(SourceBase):
    """Common base for sources that use (species, filter)."""

    def __init__(self, species: PICMISpecies, filter: str = "species_all"):
        self.species = species
        self._filter = filter
        self.check()

    @property
    def filter(self) -> str:
        return self._filter

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")
        if not isinstance(self.species, PICMISpecies):
            raise ValueError(f"Species must be a PICMISpecies, got {type(self.species)}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any],
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box=None,
    ) -> typing.Any:
        self.check()
        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species.name} is not known to Simulation")
        return getattr(pypicongpu_sources, self.__class__.__name__)(
            species=dict_species_picmi_to_pypicongpu[self.species],
            filter=self._filter,
        )


@typeguard.typechecked
class SourceBaseFilterOnly(SourceBase):
    """Common base for sources that only use filter."""

    def __init__(self, filter: str = "species_all"):
        self._filter = filter
        self.check()

    @property
    def filter(self) -> str:
        return self._filter

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Optional[typing.Dict] = None,
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box=None,
    ) -> typing.Any:
        self.check()
        return getattr(pypicongpu_sources, self.__class__.__name__)(filter=self._filter)


# ---------------------------------------------------------------------------
# Sources with (species + filter, no extras)
# ---------------------------------------------------------------------------


class BoundElectronDensity(SourceBaseSpeciesFilter):
    ...


class ChargeDensity(SourceBaseSpeciesFilter):
    ...


class Counter(SourceBaseSpeciesFilter):
    ...


class Density(SourceBaseSpeciesFilter):
    ...


class Energy(SourceBaseSpeciesFilter):
    ...


class EnergyDensity(SourceBaseSpeciesFilter):
    ...


class LarmorPower(SourceBaseSpeciesFilter):
    ...


class MacroCounter(SourceBaseSpeciesFilter):
    ...


# ---------------------------------------------------------------------------
# Sources with (filter only)
# ---------------------------------------------------------------------------


class Auto(SourceBaseFilterOnly):
    ...


class DerivedAttributes(SourceBaseFilterOnly):
    ...


# ---------------------------------------------------------------------------
# Sources with extra parameters
# ---------------------------------------------------------------------------


@typeguard.typechecked
class EnergyDensityCutoff(SourceBaseSpeciesFilter):
    """Energy density source with cutoff parameter."""

    def __init__(
        self,
        species: PICMISpecies,
        filter: str = "species_all",
        cutoff_max_energy: typing.Optional[float] = None,
    ):
        if cutoff_max_energy is None:
            raise ValueError("cutoff_max_energy is required and must be a positive number")
        self.cutoff_max_energy = cutoff_max_energy
        super().__init__(species, filter)

    def check(self) -> None:
        super().check()
        if not isinstance(self.cutoff_max_energy, (int, float)):
            raise TypeError(f"cutoff_max_energy must be a number, got {type(self.cutoff_max_energy)}")
        if self.cutoff_max_energy <= 0:
            raise ValueError(f"cutoff_max_energy must be positive, got {self.cutoff_max_energy}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any],
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box=None,
    ) -> typing.Any:
        return pypicongpu_sources.EnergyDensityCutoff(
            species=dict_species_picmi_to_pypicongpu[self.species],
            filter=self._filter,
            cutoff_max_energy=self.cutoff_max_energy,
        )


@typeguard.typechecked
class Momentum(SourceBaseSpeciesFilter):
    """Momentum-like sources with a direction (x, y, z)."""

    def __init__(self, species: PICMISpecies, filter: str = "species_all", direction: str = "x"):
        self.direction = direction
        super().__init__(species, filter)

    def check(self) -> None:
        super().check()
        if self.direction not in ["x", "y", "z"]:
            raise ValueError(f"Direction must be 'x', 'y', or 'z', got {self.direction}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any],
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box=None,
    ) -> typing.Any:
        return getattr(pypicongpu_sources, self.__class__.__name__)(  # dynamic dispatch
            species=dict_species_picmi_to_pypicongpu[self.species],
            filter=self._filter,
            direction=self.direction,
        )


class MidCurrentDensityComponent(Momentum):
    ...


class MomentumDensity(Momentum):
    ...


class WeightedVelocity(Momentum):
    ...
