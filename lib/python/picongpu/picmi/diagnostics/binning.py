"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import warnings
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from picongpu.picmi.diagnostics.backend_config import OpenPMDConfig
from picongpu.picmi.particle_functor import ParticleFunctor as BinningFunctor
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.binning import Binning as PyPIConGPUBinning
from picongpu.pypicongpu.output.binning import BinningAxis as PyPIConGPUBinningAxis
from picongpu.pypicongpu.output.binning import BinSpec as PyPIConGPUBinSpec
from picongpu.pypicongpu.output.binning import ParticleRegion

from ..copy_attributes import default_converts_to
from .timestepspec import TimeStepSpec


def _period_starts_at_zero(period: TimeStepSpec) -> bool:
    for spec in period.specs + period.specs_in_seconds:
        if spec.start is None or spec.start == 0:
            return True
    return False


@default_converts_to(PyPIConGPUBinSpec, conversions={"kind": lambda self, *_, **__: self.kind.lower().capitalize()})
class BinSpec(BaseModel):
    kind: str
    start: int | float
    stop: int | float
    nsteps: int


class BinningAxis(BaseModel):
    functor: BinningFunctor
    bin_spec: BinSpec
    name: str | None = None
    use_overflow_bins: bool = True

    @model_validator(mode="after")
    def _set_default_name(self):
        self.name = self.name or self.functor.name
        return self

    def get_as_pypicongpu(self) -> PyPIConGPUBinningAxis:
        return PyPIConGPUBinningAxis(
            name=self.name,
            functor=self.functor.get_as_pypicongpu(mode="Binning"),
            bin_spec_raw=self.bin_spec.get_as_pypicongpu(),
            use_overflow_bins=self.use_overflow_bins,
        )


class Binning(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    deposition_functor: BinningFunctor
    axes: list[BinningAxis]
    species: Species | FilteredSpecies | list[Species | FilteredSpecies]
    period: TimeStepSpec | None = None
    openPMDBackendConfig: dict | None = None
    openPMDExt: str | None = None
    openPMDInfix: str | None = None
    dumpPeriod: int = 1
    particle_region: list[ParticleRegion] | set[ParticleRegion] | tuple[ParticleRegion, ...] = ["Bounded"]

    @field_validator("species", mode="before")
    @classmethod
    def _normalise_species_to_list(cls, species):
        if isinstance(species, Species) or isinstance(species, FilteredSpecies):
            return [species]
        return species

    @field_validator("particle_region", mode="before")
    @classmethod
    def _normalise_particle_region(cls, particle_region):
        if isinstance(particle_region, set):
            return sorted(particle_region)
        if isinstance(particle_region, tuple):
            return list(particle_region)
        return particle_region

    @field_validator("particle_region")
    @classmethod
    def _validate_particle_region(cls, particle_region):
        if not particle_region:
            raise ValueError("at least one particle region must be selected")
        if len(set(particle_region)) != len(particle_region):
            raise ValueError("particle_region must not contain duplicate regions")
        return particle_region

    @model_validator(mode="after")
    def _set_default_period(self):
        self.period = self.period or TimeStepSpec[:]
        return self

    @model_validator(mode="after")
    def _warn_leaving_notify_period(self):
        if "Leaving" in self.particle_region and _period_starts_at_zero(self.period):
            warnings.warn(
                "Binning region 'Leaving' with a notify period starting at timestep 0: the binner is "
                "notified at timestep 0 but onParticleLeave is not called then, so with time averaging "
                "this adds a spurious accumulate count. Consider starting the notify period at timestep 1, "
                "e.g. TimeStepSpec[1:] (see binningPlugin.rst).",
                UserWarning,
                stacklevel=2,
            )
        return self

    def result_path(self, prefix_path):
        return OpenPMDConfig(
            file=self.name, ext=self.openPMDExt or ".bp5", infix=self.openPMDInfix or "_%06T"
        ).result_path(prefix_path=Path(prefix_path) / "simOutput" / "binningOpenPMD")

    def get_as_pypicongpu(
        self,
        time_step_size,
        num_steps,
    ) -> PyPIConGPUBinning:
        return PyPIConGPUBinning(
            name=self.name,
            deposition_functor=self.deposition_functor.get_as_pypicongpu(mode="Binning"),
            axes=list(map(BinningAxis.get_as_pypicongpu, self.axes)),
            species=[s.get_as_pypicongpu(mode="Binning") for s in self.species],
            period=self.period.get_as_pypicongpu(time_step_size, num_steps),
            openPMDBackendConfig=self.openPMDBackendConfig,
            openPMDExt=self.openPMDExt,
            openPMDInfix=self.openPMDInfix,
            dumpPeriod=self.dumpPeriod,
            particle_region=self.particle_region,
        )
