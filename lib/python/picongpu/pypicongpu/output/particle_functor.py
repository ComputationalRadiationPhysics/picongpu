"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import numbers
from typing import Annotated

from pydantic import BaseModel, PrivateAttr, model_serializer, model_validator, BeforeValidator, Field

from ..rendering.pmaccprinter import PMAccPrinter
from ..rendering.renderedobject import RenderedObject


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
    "timestep_size": "sim.pic.getDt()",
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


class UnitDimension(BaseModel):
    _num_unit_dimensions: int = PrivateAttr(7)
    unit_dimension: list = _num_unit_dimensions.default * [0.0]

    @model_validator(mode="after")
    def check(self):
        if len(self.unit_dimension) != self._num_unit_dimensions:
            raise ValueError(
                f"Unit dimension vector has {len(self.unit_dimension)=} but {self._num_unit_dimensions=}. They must match."
            )
        return self

    @model_serializer(mode="plain")
    def translate_to_cpp(self) -> str:
        return f"std::array<double, {self._num_unit_dimensions}u>{{{','.join(map(str, self.unit_dimension))}}}"


class _PreambleStatement(BaseModel):
    statement: str


class ParticleFunctor(RenderedObject, BaseModel):
    name: str
    functor_expression: Annotated[str, BeforeValidator(lambda x: PMAccPrinter().doprint(x))]
    functor_preamble: Annotated[list[_PreambleStatement], BeforeValidator(lambda x: generate_preamble(x))] = Field(
        alias="attribute_mapping"
    )
    return_type: Annotated[str, BeforeValidator(lambda x: translate_to_cpp_type(x))]
    unit_dimension: UnitDimension | None = UnitDimension()

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
