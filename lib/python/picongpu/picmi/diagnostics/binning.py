"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any, Callable, Optional

import sympy
import typeguard

from ...pypicongpu.output.binning import (
    Binning as PyPIConGPUBinning,
    BinningAxis as PyPIConGPUBinningAxis,
    BinningFunctor as PyPIConGPUBinningFunctor,
    BinSpec as PyPIConGPUBinSpec,
)
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies
from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec
from .unit import Unit

_COORDINATE_SYSTEM = {
    (
        origin.lower(),
        precision.lower(),
        unit.lower(),
    ): tuple(sympy.Symbol(f"{c}_{precision.lower()}_{unit.lower()}") for c in coords)
    for (origin, coords) in (
        ("TOTAL", ("xt", "yt", "zt")),
        ("GLOBAL", ("xg", "yg", "zg")),
        ("LOCAL", ("xl", "yl", "zl")),
        ("MOVING_WINDOW", ("xmw", "ymw", "zmw")),
        ("LOCAL_WITH_GUARDS", ("xlg", "ylg", "zlg")),
    )
    for precision in ("CELL", "SUB_CELL")
    for unit in ("CELL", "PIC", "SI")
}


class BinningParticle:
    def __init__(self):
        self.used_attributes = {}

    def get_attribute_map(self):
        return self.used_attributes

    def get(self, attribute, **kwargs):
        if attribute == "position":
            origin = kwargs.get("origin", "total")
            precision = kwargs.get("precision", "cell")
            unit = kwargs.get("unit", "cell")
            symbols = _COORDINATE_SYSTEM[(origin, precision, unit)]
            self.used_attributes |= {symbols: ("position", origin, precision, unit)}

        elif attribute == "momentum":
            symbols = sympy.symbols("px,py,pz")
            self.used_attributes |= {symbols: "momentum"}

        elif attribute in ["gamma", "kinetic energy", "velocity"]:
            # This relies on python dictionaries having a stable ordering.
            # We first add mass and momentum
            # and later use their symbols inside of the same preamble.
            self.get("mass")
            self.get("momentum")
            if attribute == "gamma":
                symbols = sympy.Symbol("gamma")
            if attribute == "kinetic energy":
                symbols = sympy.Symbol("Ekin")
            if attribute == "velocity":
                symbols = sympy.symbols("vx,vy,vz")
            self.used_attributes |= {symbols: attribute}

        else:
            symbols = sympy.Symbol(attribute)
            self.used_attributes |= {symbols: attribute}

        return symbols


@typeguard.typechecked
class BinningFunctor:
    def check(self):
        pass

    def __init__(
        self,
        name: str,
        functor: Callable[[BinningParticle], Any],
        return_type: type | str,
        unit_dimension: type(Unit()) | None = None,
    ):
        self.name = name
        self.functor = functor
        self.return_type = return_type
        self.unit_dimension = unit_dimension

    def get_as_pypicongpu(self) -> PyPIConGPUBinningFunctor:
        self.check()
        particle = BinningParticle()
        functor_expression = self.functor(particle)
        return PyPIConGPUBinningFunctor(
            name=self.name,
            functor_expression=functor_expression,
            attribute_mapping=particle.get_attribute_map(),
            return_type=self.return_type,
            unit_dimension=self.unit_dimension,
        )


@typeguard.typechecked
class BinSpec:
    def __init__(self, kind, start, stop, nsteps):
        self.kind = kind
        self.start = start
        self.stop = stop
        self.nsteps = nsteps

    def get_as_pypicongpu(self):
        return PyPIConGPUBinSpec(self.kind.lower().capitalize(), self.start, self.stop, self.nsteps)


@typeguard.typechecked
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
            functor=self.functor.get_as_pypicongpu(),
            bin_spec=self.bin_spec.get_as_pypicongpu(),
            use_overflow_bins=self.use_overflow_bins,
        )


@typeguard.typechecked
class Binning:
    def __init__(
        self,
        name: str,
        deposition_functor: BinningFunctor,
        axes: list[BinningAxis],
        species: PICMISpecies | list[PICMISpecies],
        period: Optional[TimeStepSpec] = None,
        openPMD: Optional[dict] = None,
        openPMDExt: Optional[str] = None,
        openPMDInfix: Optional[str] = None,
        dumpPeriod: int = 1,
    ):
        self.name = name
        self.deposition_functor = deposition_functor
        self.axes = axes
        if isinstance(species, PICMISpecies):
            species = [species]
        self.species = species
        self.period = period or TimeStepSpec[:]
        self.openPMD = openPMD
        self.openPMDExt = openPMDExt
        self.openPMDInfix = openPMDInfix
        self.dumpPeriod = dumpPeriod

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size,
        num_steps,
    ) -> PyPIConGPUBinning:
        if len(not_found := [s for s in self.species if s not in dict_species_picmi_to_pypicongpu.keys()]) > 0:
            raise ValueError(f"Species {not_found} are not known to Simulation")
        pypic_species = list(map(dict_species_picmi_to_pypicongpu.get, self.species))

        return PyPIConGPUBinning(
            name=self.name,
            deposition_functor=self.deposition_functor.get_as_pypicongpu(),
            axes=list(map(BinningAxis.get_as_pypicongpu, self.axes)),
            species=pypic_species,
            period=self.period.get_as_pypicongpu(time_step_size, num_steps),
            openPMD=self.openPMD,
            openPMDExt=self.openPMDExt,
            openPMDInfix=self.openPMDInfix,
            dumpPeriod=self.dumpPeriod,
        )
