"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, field_serializer

from picongpu.pypicongpu.output.plugin import Plugin
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.particle_functor.particle_functor import ParticleFunctor
from picongpu.pypicongpu.rendering.renderedobject import RenderedObject
from picongpu.pypicongpu.species import Species


class BinSpec(RenderedObject, BaseModel):
    kind: str
    start: float | int
    stop: float | int
    nsteps: float | int


class BinningAxis(RenderedObject, BaseModel):
    axis_name: str = Field(alias="name")
    bin_spec: BinSpec
    axis_functor: ParticleFunctor = Field(alias="functor")
    use_overflow_bins: bool


class Binning(Plugin, BaseModel):
    binner_name: str = Field(alias="name")
    deposition_functor: ParticleFunctor
    axes: list[BinningAxis]
    species: list[Species | FilteredSpecies]
    period: TimeStepSpec
    openPMD: dict[str, Any] | None
    openPMDExtension: str | None = Field(alias="openPMDExt")
    openPMDInfix: str | None
    dumpPeriod: int

    _name: str = PrivateAttr("binning")

    @field_serializer("openPMD")
    def _serialize_openPMD(self, value) -> str | None:
        return None if value is None else json.dumps(value)
