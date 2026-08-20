"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from os import PathLike
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.openpmd_plugin import NATIVE_FIELDS
from .backend_config import BackendConfig, OpenPMDConfig
from .timestepspec import TimeStepSpec
from picongpu.picmi.particle_functor import ParticleFunctor


class _FieldDump(BaseModel):
    period: TimeStepSpec = TimeStepSpec[:]("steps")
    options: BackendConfig = OpenPMDConfig(file="simData")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def result_path(self, prefix_path: PathLike):
        return self.options.result_path(prefix_path=Path(prefix_path) / "simOutput" / "openPMD")


class NativeFieldDump(_FieldDump):
    fieldname: Literal[*NATIVE_FIELDS]
    filtername: None = None


class DerivedFieldDump(_FieldDump):
    species: Species | FilteredSpecies
    functor: ParticleFunctor

    @computed_field
    def filtername(self) -> None | str:
        return None if isinstance(self.species, Species) else self.species.functor.name

    @computed_field
    def fieldname(self) -> str:
        species_name = self.species.name if isinstance(self.species, Species) else self.species.species.name
        return f"{species_name}_{self.filtername or 'all'}_{self.functor.name}"


ScalarDerivedField = Literal[
    "Density",
    "BoundElectronDensity",
    "ChargeDensity",
    "Counter",
    "Energy",
    "EnergyDensity",
    "LarmorPower",
    "MacroCounter",
]
DirectionalDerivedField = Literal["MidCurrentDensityComponent", "Momentum", "MomentumDensity", "WeightedVelocity"]
CombinedDerivedField = Literal["RelativisticDensity", "ScreeningInvSquared"]
NativeDerivedField = ScalarDerivedField | DirectionalDerivedField | CombinedDerivedField
AverageableDerivedField = Literal[
    "Density",
    "BoundElectronDensity",
    "ChargeDensity",
    "Counter",
    "Energy",
    "EnergyDensity",
    "LarmorPower",
    "MidCurrentDensityComponent",
    "Momentum",
    "MomentumDensity",
    "WeightedVelocity",
]
Direction = Literal["x", "y", "z"]

_DIRECTION_INDEX = {"x": 0, "y": 1, "z": 2}
_DIRECTIONAL_FIELDS = {"MidCurrentDensityComponent", "Momentum", "MomentumDensity", "WeightedVelocity"}
_FIELD_NAMES = {
    "Density": "density",
    "BoundElectronDensity": "boundElectronDensity",
    "ChargeDensity": "chargeDensity",
    "Counter": "particleCounter",
    "Energy": "particleEnergy",
    "EnergyDensity": "energyDensity",
    "LarmorPower": "larmorPower",
    "MacroCounter": "macroParticleCounter",
    "MidCurrentDensityComponent": "midCurrentDensity",
    "Momentum": "particleMomentum",
    "MomentumDensity": "momentumDensity",
    "WeightedVelocity": "weightedVelocity",
    "RelativisticDensity": "relativisticDensity",
    "ScreeningInvSquared": "invSquaredScreenLength",
}


class _BuiltinDerivedFieldDump(_FieldDump):
    species: Species | FilteredSpecies
    direction: Direction | None = None
    field: str

    _average: ClassVar[bool] = False

    @model_validator(mode="after")
    def _validate_direction(self):
        is_directional = self.field in _DIRECTIONAL_FIELDS
        if is_directional and self.direction is None:
            raise ValueError(f"direction is required for directional field {self.field}")
        if not is_directional and self.direction is not None:
            raise ValueError(f"direction is only valid for directional fields, not {self.field}")
        return self

    @computed_field
    def filtername(self) -> None | str:
        return None if isinstance(self.species, Species) else self.species.functor.name

    @property
    def _native_name(self) -> str:
        name = _FIELD_NAMES[self.field]
        return f"{name}/{self.direction}" if self.direction is not None else name

    @computed_field
    def fieldname(self) -> str:
        species_name = self.species.name if isinstance(self.species, Species) else self.species.species.name
        native_name = f"Average_{self._native_name}" if self._average else self._native_name
        return f"{species_name}_{self.filtername or 'all'}_{native_name}"

    def get_builtin_solver(self) -> tuple[str, str]:
        direction = f"<{_DIRECTION_INDEX[self.direction]}>" if self.direction is not None else ""
        base_type = f"deriveField::derivedAttributes::{self.field}{direction}"
        solver_type = f"deriveField::combinedAttributes::AverageAttribute<{base_type}>" if self._average else base_type
        if self.field in ("RelativisticDensity", "ScreeningInvSquared"):
            solver_type = f"deriveField::combinedAttributes::{self.field}"
        identifier_parts = ["Average" if self._average else "Native", self.field]
        if self.direction is not None:
            identifier_parts.append(str(_DIRECTION_INDEX[self.direction]))
        if self.filtername is not None:
            identifier_parts.append(self.filtername)
        return solver_type, "_".join(identifier_parts)


class NativeDerivedFieldDump(_BuiltinDerivedFieldDump):
    """Dump a field using a built-in PIConGPU particle-to-grid implementation."""

    field: NativeDerivedField


class AverageDerivedFieldDump(_BuiltinDerivedFieldDump):
    """Dump the cell-wise average of a built-in PIConGPU derived field."""

    field: AverageableDerivedField
    _average: ClassVar[bool] = True
