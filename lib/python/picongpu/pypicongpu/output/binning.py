"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.particle_functor.particle_functor import ParticleFunctor
from picongpu.pypicongpu.particle_functor.translate_to_cpp_type import translate_from_cpp_type
from picongpu.pypicongpu.rendering.renderedobject import RenderedObject
from picongpu.pypicongpu.species import Species

PARTICLE_REGIONS: tuple["ParticleRegion", ...] = ("Bounded", "Leaving")
ParticleRegion = Literal[*PARTICLE_REGIONS]


class BinSpec(RenderedObject, BaseModel):
    kind: str
    start: int | float
    stop: int | float
    nsteps: int


class BinningAxis(RenderedObject, BaseModel):
    axis_name: str = Field(alias="name")
    bin_spec_raw: BinSpec = Field(exclude=True)
    axis_functor: ParticleFunctor = Field(alias="functor")
    use_overflow_bins: bool

    @computed_field
    def bin_spec(self) -> BinSpec:
        return BinSpec(
            kind=self.bin_spec_raw.kind,
            nsteps=self.bin_spec_raw.nsteps,
            start=translate_from_cpp_type(self.axis_functor.return_type)(self.bin_spec_raw.start),
            stop=translate_from_cpp_type(self.axis_functor.return_type)(self.bin_spec_raw.stop),
        )


class Binning(BaseModel):
    binner_name: str = Field(alias="name")
    deposition_functor: ParticleFunctor
    axes: list[BinningAxis]
    species: list[Species | FilteredSpecies]
    period: TimeStepSpec
    openPMDBackendConfig: dict[str, Any] | None
    openPMDExtension: str | None = Field(alias="openPMDExt")
    openPMDInfix: str | None
    dumpPeriod: int
    # `region_directives` (below) is the serialised/computed form of `particle_region`; the raw
    # field is excluded because rendering requires list leaves to contain only dicts
    particle_region: list[ParticleRegion] = Field(default=["Bounded"], exclude=True)

    type_binning: Literal[True] = True

    @field_validator("particle_region")
    @classmethod
    def _validate_particle_region(cls, value):
        if not value:
            raise ValueError("particle_region must contain at least one region")
        if len(set(value)) != len(value):
            raise ValueError("particle_region must not contain duplicate regions")
        return value

    @computed_field
    def region_directives(self) -> list[dict[str, str]]:
        return [
            {
                "action": "enableRegion" if region in self.particle_region else "disableRegion",
                "region": f"ParticleRegion::{region}",
            }
            for region in PARTICLE_REGIONS
        ]

    @field_serializer("openPMDBackendConfig")
    def _serialize_openPMDBackendConfig(self, value) -> str | None:
        return None if value is None else json.dumps(value)
