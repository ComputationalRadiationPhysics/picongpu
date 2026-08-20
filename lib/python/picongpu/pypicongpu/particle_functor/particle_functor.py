"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from hashlib import sha256
from json import dumps
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, computed_field, model_validator

from picongpu.pypicongpu.particle_functor.translate_to_cpp_type import translate_to_cpp_type
from picongpu.pypicongpu.particle_functor.rng_info import RNGInfo
from picongpu.pypicongpu.particle_functor.unit_dimension import UnitDimension
from picongpu.pypicongpu.rendering.pmaccprinter import PMAccPrinter
from picongpu.pypicongpu.rendering.renderedobject import RenderedObject
from picongpu.pypicongpu.util import alt


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

BINNING_ACCESSORS = (
    COMMON_ACCESSORS
    | {
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
    | {"random_number": NotImplemented}
)

_ORIGINS = [
    ("local", f"static_cast<float3_X>({by_bracket('localCellIdx')}"),
    ("cell", f"static_cast<float3_X>({by_bracket('position')}"),
    ("total", "static_cast<float3_X>(particleOffsetToTotalOrigin)"),
]
_PRECISIONS = [("cell", ""), ("sub_cell", " + " + by_bracket("position"))]
_UNITS = [("cell", ""), ("si", "* sim.si.getCellSize()"), ("pic", "* sim.pic.getCellSize()")]

DERIVED_FIELD_ACCESSORS = (
    COMMON_ACCESSORS
    | {
        ("position", origin, precision, unit): f"({o_expr}{p_expr}){u_expr}"
        for origin, o_expr in _ORIGINS
        if origin != "total"
        for precision, p_expr in _PRECISIONS
        for unit, u_expr in _UNITS
    }
    | {"random_number": NotImplemented}
)

FILTER_ACCESSORS = (
    DERIVED_FIELD_ACCESSORS
    | {
        ("position", origin, precision, unit): f"({o_expr}{p_expr}){u_expr}"
        for origin, o_expr in _ORIGINS
        if origin == "total"
        for precision, p_expr in _PRECISIONS
        for unit, u_expr in _UNITS
    }
    | {"random_number": "rng()"}
)


def random_number_command(**kwargs):
    scale = kwargs.get("scale", 1)
    if scale < 0:
        raise ValueError(f"{scale=} must be >= 0.")
    return f"random_number(rng, static_cast<typename RNGType::result_type>({kwargs.get('loc', 0)}), static_cast<typename RNGType::result_type>({scale}))"


def filter_access(name, default):
    if name in FILTER_ACCESSORS:
        return FILTER_ACCESSORS[name]
    if alt(lambda: name[0] == "random_number", False):
        return random_number_command(**dict(name[1]))
    return default


ACCESSORS = {
    "Binning": lambda name, default: BINNING_ACCESSORS.get(name, default),
    "DerivedField": lambda name, default: DERIVED_FIELD_ACCESSORS.get(name, default),
    "Filter": filter_access,
}


def symbol_to_string(symbol):
    return str(symbol) if not isinstance(symbol, tuple) else "[" + ",".join(map(str, symbol)) + "]"


def generate_preamble(attribute_mapping, mode: Literal["Binning", "Filter", "DerivedField"]):
    statements = {
        symbol: ACCESSORS[mode](attribute, by_bracket(attribute)) for symbol, attribute in attribute_mapping.items()
    }
    if unsupported_synbols := [symbol for symbol, statement in statements.items() if statement is NotImplemented]:
        raise ValueError(f"Found {unsupported_synbols=} trying to generate C++ code for one of your functors.")
    return [
        {"statement": f"auto const {symbol_to_string(symbol)} = {statement};"}
        for symbol, statement in statements.items()
    ]


class _PreambleStatement(BaseModel):
    statement: str


class ParticleFunctor(RenderedObject, BaseModel):
    name: str
    functor_expression: Annotated[str, BeforeValidator(PMAccPrinter().doprint)]
    functor_preamble: list[_PreambleStatement]
    return_type: Annotated[str, BeforeValidator(translate_to_cpp_type)]
    unit_dimension: UnitDimension | None = UnitDimension()
    needs_total_position: bool = False
    rng_info: RNGInfo | None = None

    @computed_field
    def typename(self) -> str:
        definition = self.model_dump(mode="json", exclude={"typename"})
        digest = sha256(dumps(definition, sort_keys=True).encode()).hexdigest()[:32]
        return f"{self.name}_{digest}"

    @model_validator(mode="after")
    def _validate(self):
        if "int" in self.return_type:
            if self.unit_dimension == UnitDimension():
                self.unit_dimension = None
            if self.unit_dimension is not None:
                raise ValueError(
                    f"unit_dimension is not supported for integral types. You gave {self.unit_dimension=}."
                )
        if self.needs_total_position and self.rng_info is not None:
            raise ValueError(
                f"PIConGPU does not support particle functors that need total position and random numbers. You gave: {self.rng_info=}."
            )
        return self
