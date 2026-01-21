"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import numbers
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, model_validator

from picongpu.pypicongpu.particle_functor.unit_dimension import UnitDimension
from picongpu.pypicongpu.rendering.pmaccprinter import PMAccPrinter
from picongpu.pypicongpu.rendering.renderedobject import RenderedObject


def by_bracket(attribute):
    return f"particle[{attribute}_]"


COMMON_ACCESSORS = {
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
    "timestep_size": "sim.pic.getDt()",
}

BINNING_ACCESSORS = COMMON_ACCESSORS | {
    (
        "position",
        origin.lower(),
        precision.lower(),
        unit.lower(),
    ): f"getParticlePosition<DomainOrigin::{origin}, PositionPrecision::{precision}, PositionUnits::{unit}>(domainInfo, particle)"
    for origin in ("TOTAL", "GLOBAL", "LOCAL", "MOVING_WINDOW", "LOCAL_WITH_GUARDS")
    for precision in ("CELL", "SUB_CELL")
    for unit in ("CELL", "PIC", "SI")
}

_ORIGINS = [
    ("local", f"static_cast<float3_X>({by_bracket('localCellIdx')}"),
    ("cell", f"static_cast<float3_X>({by_bracket('position')}"),
    ("total", "static_cast<float3_X>(particleOffsetToTotalOrigin)"),
]
_PRECISIONS = [("cell", ""), ("sub_cell", " + " + by_bracket("position"))]
_UNITS = [("cell", ""), ("si", "* sim.si.getCellSize()"), ("pic", "* sim.pic.getCellSize()")]

DERIVED_FIELD_ACCESSORS = COMMON_ACCESSORS | {
    ("position", origin, precision, unit): f"({o_expr}{p_expr}){u_expr}"
    for origin, o_expr in _ORIGINS
    if origin != "total"
    for precision, p_expr in _PRECISIONS
    for unit, u_expr in _UNITS
}

FILTER_ACCESSORS = DERIVED_FIELD_ACCESSORS | {
    ("position", origin, precision, unit): f"({o_expr}{p_expr}){u_expr}"
    for origin, o_expr in _ORIGINS
    if origin == "total"
    for precision, p_expr in _PRECISIONS
    for unit, u_expr in _UNITS
}

ACCESSORS = {"Binning": BINNING_ACCESSORS, "DerivedField": DERIVED_FIELD_ACCESSORS, "Filter": FILTER_ACCESSORS}


def symbol_to_string(symbol):
    return str(symbol) if not isinstance(symbol, tuple) else "[" + ",".join(map(str, symbol)) + "]"


def generate_preamble(attribute_mapping, mode: Literal["Binning", "Filter", "DerivedField"]):
    # Positions are special in that not all functors have access to all kinds of positions.
    # We only allow the ones we explicitly know how to access.
    if unknown_position_requests := [
        pos
        for pos in attribute_mapping.values()
        if isinstance(pos, tuple) and pos[0] == "position" and pos not in ACCESSORS[mode]
    ]:
        raise ValueError(
            "You requested information about a particle position that PIConGPU can't provide for "
            f"{mode=}. You gave: {unknown_position_requests=}."
        )

    return [
        {
            "statement": f"auto const {symbol_to_string(symbol)} = {ACCESSORS[mode].get(attribute, by_bracket(attribute))};"
        }
        for symbol, attribute in attribute_mapping.items()
    ]


def translate_to_cpp_type(return_type):
    try:
        # Ordering is important here because issubclass(bool, int) is True in Python world
        if issubclass(return_type, bool):
            return "bool"
        if issubclass(return_type, numbers.Integral):
            return "int"
        if issubclass(return_type, numbers.Real):
            return "float_X"
    except TypeError:
        pass
    if isinstance(return_type, str):
        return return_type
    raise ValueError(f"Cannot translate {return_type=} to a C++ type.")


class _PreambleStatement(BaseModel):
    statement: str


class ParticleFunctor(RenderedObject, BaseModel):
    name: str
    functor_expression: Annotated[str, BeforeValidator(lambda x: PMAccPrinter().doprint(x))]
    functor_preamble: list[_PreambleStatement]
    return_type: Annotated[str, BeforeValidator(lambda x: translate_to_cpp_type(x))]
    unit_dimension: UnitDimension | None = UnitDimension()
    needs_total_position: bool = False

    @model_validator(mode="after")
    def _validate(self):
        if "int" in self.return_type:
            if self.unit_dimension == UnitDimension():
                self.unit_dimension = None
            if self.unit_dimension is not None:
                raise ValueError(
                    f"unit_dimension is not supported for integral types. You gave {self.unit_dimension=}."
                )
        return self
