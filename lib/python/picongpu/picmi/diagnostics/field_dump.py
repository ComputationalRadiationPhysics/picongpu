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

from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.openpmd_plugin import NATIVE_FIELDS
from .backend_config import BackendConfig, OpenPMDConfig
from .timestepspec import TimeStepSpec
from picongpu.picmi.diagnostics.particle_functor import ParticleFunctor


class _FieldDump(BaseModel):
    period: TimeStepSpec = TimeStepSpec[:]("steps")
    options: BackendConfig = OpenPMDConfig(file="simData")

    class Config:
        arbitrary_types_allowed = True

    def result_path(self, prefix_path: PathLike):
        return self.options.result_path(prefix_path=Path(prefix_path) / "simOutput" / "openPMD")


class NativeFieldDump(_FieldDump):
    fieldname: Literal[*NATIVE_FIELDS]


class DerivedFieldDump(_FieldDump):
    species: Species
    functor: ParticleFunctor

    @computed_field
    def fieldname(self) -> str:
        return f"{self.species.name}_all_{self.functor.name}"
