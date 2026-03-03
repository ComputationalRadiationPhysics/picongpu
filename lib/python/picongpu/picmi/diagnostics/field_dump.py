"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from os import PathLike
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field

from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.openpmd_plugin import NATIVE_FIELDS
from .backend_config import BackendConfig, OpenPMDConfig
from .timestepspec import TimeStepSpec
from picongpu.picmi.particle_functor import ParticleFunctor


class _FieldDump(BaseModel):
    period: TimeStepSpec = TimeStepSpec[:]("steps")
    options: BackendConfig = OpenPMDConfig(file="simData")

    class Config:
        arbitrary_types_allowed = True

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
