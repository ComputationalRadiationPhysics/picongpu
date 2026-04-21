"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, model_validator

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.particle_functor.particle_functor import ParticleFunctor
from picongpu.pypicongpu.particle_functor.translate_to_cpp_type import translate_from_cpp_type
from picongpu.pypicongpu.rendering.renderedobject import RenderedObject
from picongpu.pypicongpu.species import Species


class BinSpec(RenderedObject, BaseModel):
    kind: str
    start: float | int
    stop: float | int
    nsteps: int


class BinningAxis(RenderedObject, BaseModel):
    axis_name: str = Field(alias="name")
    bin_spec: BinSpec
    axis_functor: ParticleFunctor = Field(alias="functor")
    use_overflow_bins: bool

    @model_validator(mode="after")
    def _match_bin_spec_type(self):
        self.bin_spec = BinSpec(
            kind=self.bin_spec.kind,
            nsteps=self.bin_spec.nsteps,
            start=translate_from_cpp_type(self.axis_functor.return_type)(self.bin_spec.start),
            stop=translate_from_cpp_type(self.axis_functor.return_type)(self.bin_spec.stop),
        )
        return self


class Binning(BaseModel):
    binner_name: str = Field(alias="name")
    deposition_functor: ParticleFunctor
    axes: list[BinningAxis]
    species: list[Species | FilteredSpecies]
    period: TimeStepSpec
    openPMD: dict[str, Any] | None
    openPMDExtension: str | None = Field(alias="openPMDExt")
    openPMDInfix: str | None
    dumpPeriod: int

    type_binning: Literal[True] = True

    @field_serializer("openPMD")
    def _serialize_openPMD(self, value) -> str | None:
        return None if value is None else json.dumps(value)
