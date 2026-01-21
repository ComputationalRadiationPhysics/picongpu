"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path

from picongpu.picmi.diagnostics.backend_config import OpenPMDConfig
from picongpu.picmi.particle_functor import ParticleFunctor as BinningFunctor
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.binning import Binning as PyPIConGPUBinning
from picongpu.pypicongpu.output.binning import BinningAxis as PyPIConGPUBinningAxis
from picongpu.pypicongpu.output.binning import BinSpec as PyPIConGPUBinSpec

from ..copy_attributes import default_converts_to
from .timestepspec import TimeStepSpec


@default_converts_to(PyPIConGPUBinSpec, conversions={"kind": lambda self, *_, **__: self.kind.lower().capitalize()})
class BinSpec:
    def __init__(self, kind, start, stop, nsteps):
        self.kind = kind
        self.start = start
        self.stop = stop
        self.nsteps = nsteps


class BinningAxis:
    def __init__(
        self,
        functor: BinningFunctor,
        bin_spec: BinSpec,
        name: str | None = None,
        use_overflow_bins: bool = True,
    ):
        self.functor = functor
        self.bin_spec = bin_spec
        self.name = name or functor.name
        self.use_overflow_bins = use_overflow_bins

    def get_as_pypicongpu(self) -> PyPIConGPUBinningAxis:
        return PyPIConGPUBinningAxis(
            name=self.name,
            functor=self.functor.get_as_pypicongpu(mode="Binning"),
            bin_spec=self.bin_spec.get_as_pypicongpu(),
            use_overflow_bins=self.use_overflow_bins,
        )


class Binning:
    def __init__(
        self,
        name: str,
        deposition_functor: BinningFunctor,
        axes: list[BinningAxis],
        species: Species | list[Species],
        period: TimeStepSpec | None = None,
        openPMD: dict | None = None,
        openPMDExt: str | None = None,
        openPMDInfix: str | None = None,
        dumpPeriod: int = 1,
    ):
        self.name = name
        self.deposition_functor = deposition_functor
        self.axes = axes
        if isinstance(species, Species) or isinstance(species, FilteredSpecies):
            species = [species]
        self.species = species
        self.period = period or TimeStepSpec[:]
        self.openPMD = openPMD
        self.openPMDExt = openPMDExt
        self.openPMDInfix = openPMDInfix
        self.dumpPeriod = dumpPeriod

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
            openPMD=self.openPMD,
            openPMDExt=self.openPMDExt,
            openPMDInfix=self.openPMDInfix,
            dumpPeriod=self.dumpPeriod,
        )
