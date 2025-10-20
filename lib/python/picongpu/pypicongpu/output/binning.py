"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
import numbers

from typing import Optional
from .timestepspec import TimeStepSpec
from ..rendering.renderedobject import RenderedObject
from ..rendering.pmaccprinter import PMAccPrinter
from ..species import Species
from .. import util
from .plugin import Plugin

import typeguard


def by_bracket(attribute):
    return f"particle[{attribute}_]"


ACCESSORS = {
    (
        "position",
        origin.lower(),
        precision.lower(),
        unit.lower(),
    ): f"getParticlePosition<DomainOrigin::{origin}, PositionPrecision::{precision}, PositionUnits::{unit}>(domainInfo, particle)"
    for origin in ("TOTAL", "GLOBAL", "LOCAL", "MOVING_WINDOW", "LOCAL_WITH_GUARDS")
    for precision in ("CELL", "SUB_CELL")
    for unit in ("CELL", "PIC", "SI")
} | {
    "mass": "picongpu::traits::attribute::getMass(particle[weighting_], particle)",
    # CAUTION: The names in the gamma formula are currently hardcoded.
    # We'll certainly trip over this, should we ever dare to change the internal names.
    "gamma": "picongpu::Gamma()(momentum::type{px, py, pz}, mass)",
    "kinetic energy": "picongpu::KinEnergy()(momentum::type{px, py, pz}, mass)",
    "velocity": "picongpu::Velocity()(momentum::type{px, py, pz}, mass)",
    "charge": "picongpu::traits::attribute::getCharge(particle[weighting_], particle)",
    "charge_state": "picongpu::traits::attribute::getChargeState(particle)",
    "damped_weighting": "picongpu::traits::attribute::getDampedWeighting(particle)",
    "timestep": "domainInfo.currentStep",
}


def symbol_to_string(symbol):
    return str(symbol) if not isinstance(symbol, tuple) else "[" + ",".join(map(str, symbol)) + "]"


def generate_preamble(attribute_mapping):
    return [
        {"statement": f"auto const {symbol_to_string(symbol)} = {ACCESSORS.get(attribute, by_bracket(attribute))};"}
        for symbol, attribute in attribute_mapping.items()
    ]


def translate_to_cpp_type(return_type):
    try:
        if issubclass(return_type, numbers.Integral):
            return "int"
        if issubclass(return_type, numbers.Real):
            return "float_X"
        if issubclass(return_type, bool):
            return "bool"
    except TypeError:
        pass
    if isinstance(return_type, str):
        return return_type
    raise ValueError(f"Cannot translate {return_type=} to a C++ type.")


def translate_to_cpp_unitDimVec(unit_dimension):
    if unit_dimension == None:
        # catch picmi default input
        unit_dimension = [0., 0., 0., 0., 0., 0., 0.]

    if (isinstance(unit_dimension, list) and
        len(unit_dimension) == 7 and
        all(isinstance(item, float) for item in unit_dimension)):
        result = "std::array<double, 7>{ "
        for num in unit_dimension:
            result += f"{num}, "
        result += "}"
        return result
    else:
        raise ValueError(f"The input {unit_dimension} is not a list of 7 floats.")


class BinningFunctor(RenderedObject):
    def __init__(self, name, functor_expression, attribute_mapping, return_type, unit_dimension=[0., 0., 0., 0., 0., 0., 0.]):
        self.name = name
        self.functor_expression = functor_expression
        self.attribute_mapping = attribute_mapping
        self.return_type = return_type
        self.unit_dimension = unit_dimension

    def _get_serialized(self):
        return {
            "name": self.name,
            "functor_expression": PMAccPrinter().doprint(self.functor_expression),
            "functor_preamble": generate_preamble(self.attribute_mapping),
            "return_type": translate_to_cpp_type(self.return_type),
            "unit_dimension": translate_to_cpp_unitDimVec(self.unit_dimension),
        }


class BinSpec(RenderedObject):
    def __init__(self, kind, start, stop, nsteps):
        self.kind = kind
        self.start = start
        self.stop = stop
        self.nsteps = nsteps

    def _get_serialized(self):
        return {
            "kind": self.kind,
            "start": self.start,
            "stop": self.stop,
            "nsteps": self.nsteps,
        }


class BinningAxis(RenderedObject):
    name: str
    bin_spec: BinSpec
    functor: BinningFunctor

    def __init__(self, name, bin_spec, functor, use_overflow_bins):
        self.name = name
        self.bin_spec = bin_spec
        self.functor = functor
        self.use_overflow_bins = use_overflow_bins

    def _get_serialized(self):
        return {
            "axis_name": self.name,
            "bin_spec": self.bin_spec.get_rendering_context(),
            "axis_functor": self.functor.get_rendering_context(),
            "use_overflow_bins": self.use_overflow_bins,
        }


@typeguard.typechecked
class Binning(Plugin):
    name = util.build_typesafe_property(str)
    species = util.build_typesafe_property(list[Species])
    period = util.build_typesafe_property(TimeStepSpec)
    openPMD = util.build_typesafe_property(Optional[dict])
    openPMDExt = util.build_typesafe_property(Optional[str])
    openPMDInfix = util.build_typesafe_property(Optional[str])
    dumpPeriod = util.build_typesafe_property(int)

    _name = "binning"

    def __init__(
        self,
        name,
        deposition_functor,
        axes,
        species,
        period,
        openPMD,
        openPMDExt,
        openPMDInfix,
        dumpPeriod,
    ):
        self.name = name
        self.deposition_functor = deposition_functor
        self.axes = axes
        self.species = species
        self.period = period
        self.openPMD = openPMD
        self.openPMDExt = openPMDExt
        self.openPMDInfix = openPMDInfix
        self.dumpPeriod = dumpPeriod

    def _get_serialized(self) -> dict:
        return {
            "binner_name": self.name,
            "deposition_functor": self.deposition_functor.get_rendering_context(),
            "axes": list(map(BinningAxis.get_rendering_context, self.axes)),
            "species": list(map(Species.get_rendering_context, self.species)),
            "period": self.period.get_rendering_context(),
            "openPMD": json.dumps(self.openPMD) if self.openPMD else self.openPMD,
            "openPMDExtension": self.openPMDExt,
            "openPMDInfix": self.openPMDInfix,
            "dumpPeriod": self.dumpPeriod,
        }
