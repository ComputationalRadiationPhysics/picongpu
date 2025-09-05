"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ... import util
from ...species import Species
from .source_base import SourceBase
import typeguard
import typing
from typing import Optional


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@typeguard.typechecked
class SourceBaseSpeciesFilter(SourceBase):
    """Common base for sources that use (species, filter)."""

    species = util.build_typesafe_property(Species)
    filter = util.build_typesafe_property(str)

    def __init__(self, species: Species, filter: str = "species_all"):  # default filter ="species_all"
        self.species = species
        self.filter = filter
        self.check()

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if self.filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self.filter}")
        if not isinstance(self.species, Species):
            raise ValueError(f"Species must be a Species, got {type(self.species)}")

    def _get_serialized(self) -> typing.Dict:
        self.check()
        return {
            "species": self.species.get_rendering_context(),
            "filter": self.filter,
            "type": self.__class__.__name__.lower(),
        }


@typeguard.typechecked
class SourceBaseFilterOnly(SourceBase):
    """Common base for sources that only use filter."""

    filter = util.build_typesafe_property(str)

    def __init__(self, filter: str = "species_all"):
        self.filter = filter
        self.check()

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if self.filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self.filter}")

    def _get_serialized(self) -> typing.Dict:
        self.check()
        return {"filter": self.filter, "type": self.__class__.__name__.lower()}


# ---------------------------------------------------------------------------
# sources with (species + filter, no extras)
# ---------------------------------------------------------------------------


class BoundElectronDensity(SourceBaseSpeciesFilter):
    pass


class ChargeDensity(SourceBaseSpeciesFilter):
    pass


class Counter(SourceBaseSpeciesFilter):
    pass


class Density(SourceBaseSpeciesFilter):
    pass


class Energy(SourceBaseSpeciesFilter):
    pass


class EnergyDensity(SourceBaseSpeciesFilter):
    pass


class LarmorPower(SourceBaseSpeciesFilter):
    pass


class MacroCounter(SourceBaseSpeciesFilter):
    pass


# ---------------------------------------------------------------------------
# sources with (filter only)
# ---------------------------------------------------------------------------


class Auto(SourceBaseFilterOnly):
    pass


class DerivedAttributes(SourceBaseFilterOnly):
    pass


# ---------------------------------------------------------------------------
# Sources with extra parameters
# ---------------------------------------------------------------------------


@typeguard.typechecked
class EnergyDensityCutoff(SourceBaseSpeciesFilter):
    cutoff_max_energy = util.build_typesafe_property(Optional[float])

    def __init__(self, species: Species, filter: str = "species_all", cutoff_max_energy: Optional[float] = None):
        self.cutoff_max_energy = cutoff_max_energy
        super().__init__(species, filter)

    def check(self) -> None:
        super().check()
        if self.cutoff_max_energy is not None and not isinstance(self.cutoff_max_energy, (int, float)):
            raise ValueError(f"cutoff_max_energy must be a number or None, got {type(self.cutoff_max_energy)}")
        if self.cutoff_max_energy is not None and self.cutoff_max_energy <= 0:
            raise ValueError(f"cutoff_max_energy must be positive, got {self.cutoff_max_energy}")

    def _get_serialized(self) -> typing.Dict:
        base = super()._get_serialized()
        base.update({"cutoff_max_energy": self.cutoff_max_energy})
        return base


@typeguard.typechecked
class Momentum(SourceBaseSpeciesFilter):
    direction = util.build_typesafe_property(str)

    def __init__(self, species: Species, filter: str = "species_all", direction: str = "x"):
        self.direction = direction
        super().__init__(species, filter)

    def check(self) -> None:
        super().check()
        if self.direction not in ["x", "y", "z"]:
            raise ValueError(f"Direction must be 'x', 'y', or 'z', got {self.direction}")

    def _get_serialized(self) -> typing.Dict:
        base = super()._get_serialized()
        base.update({"direction": self.direction})
        return base


@typeguard.typechecked
class MidCurrentDensityComponent(Momentum):
    """Same as Momentum (species + filter + direction)."""

    pass


@typeguard.typechecked
class MomentumDensity(Momentum):
    """Same as Momentum (species + filter + direction)."""

    pass


@typeguard.typechecked
class WeightedVelocity(Momentum):
    """Same as Momentum (species + filter + direction)."""

    pass
