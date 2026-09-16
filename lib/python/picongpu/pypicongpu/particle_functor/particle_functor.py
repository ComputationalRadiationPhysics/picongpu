"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated, Literal
from uuid import uuid4 as uuid

from pydantic import BaseModel, BeforeValidator, Field, computed_field, model_validator

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


def _format_exponent(exponent):
    value = float(exponent)
    return f"{int(value)}.0" if value == int(value) else repr(value)


def _unit_monomial(exponents):
    """Render the numeric ``sim.unit.*`` monomial for ``getUnit()``.

    The internal unit system is a monomial in the base SI values
    (``sim.unit.length = c * dt``, ``.mass``, ``.time = dt``, ``.charge``),
    so for any pure-monomial quantity the SI value of one internal unit is
    fully determined by the 7-vector. Temperature, amount-of-substance (the
    ``N_ppm`` count factor) and luminous intensity cannot be derived and must
    be provided via an explicit ``unit_factor`` instead.
    """
    length, mass, time, current = (float(exp) for exp in exponents[:4])
    if any(abs(float(exp)) > 1.0e-12 for exp in exponents[4:]):
        raise ValueError(
            "getUnit() cannot be auto-derived from this unit_dimension "
            "(it has temperature/amount-of-substance/luminous-intensity components). "
            "Provide an explicit unit_factor."
        )
    if any(abs(float(exp) - round(float(exp))) > 1.0e-12 for exp in exponents[:4]):
        raise ValueError(
            f"getUnit() cannot be auto-derived from a non-integer unit_dimension. "
            f"Provide an explicit unit_factor. You gave: {list(exponents[:4])}."
        )
    # The SI current is a charge per time, so it folds into charge^I * time^-I.
    aggregated = {
        "sim.unit.length()": int(round(length)),
        "sim.unit.mass()": int(round(mass)),
        "sim.unit.time()": int(round(time - current)),
        "sim.unit.charge()": int(round(current)),
    }
    numerator, denominator = [], []
    for base, exponent in aggregated.items():
        if exponent > 0:
            numerator.extend([base] * exponent)
        elif exponent < 0:
            denominator.extend([base] * -exponent)
    if not numerator and not denominator:
        return "1."
    if numerator and not denominator:
        return " * ".join(numerator)
    if not numerator:
        return f"1. / {' * '.join(denominator)}"
    return f"({' * '.join(numerator)}) / ({' * '.join(denominator)})"


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


class _SpeciesName(BaseModel):
    name: str
    """Compile-time name (``GetCTName_t``) of a species this functor is registered for."""


class ParticleFunctor(RenderedObject, BaseModel):
    name: str
    functor_expression: Annotated[str, BeforeValidator(PMAccPrinter().doprint)]
    functor_preamble: list[_PreambleStatement]
    return_type: Annotated[str, BeforeValidator(translate_to_cpp_type)]
    unit_dimension: UnitDimension | None = UnitDimension()
    # Already-rendered C++ text of ``getUnit()``; the public (picmi) interface
    # accepts Python expressions/callables and renders them before constructing
    # this pypicongpu model.
    unit_factor: str | None = None
    needs_total_position: bool = False
    rng_info: RNGInfo | None = None
    species_names: list[_SpeciesName] = Field(default_factory=list)
    """Compile-time names of the species this functor is registered for.

    Empty for functors that are not reusable particle filters (e.g. derived-field
    and binning functors). For reusable particle filters it lists exactly the
    species the filter is used on, so that ``particleFilters.param`` can emit a
    name-keyed ``SpeciesEligibleForSolver`` specialisation restricting the filter
    to only those species instead of every species in ``VectorAllSpecies``."""

    @computed_field
    def typename(self) -> str:
        return f"{self.name}_{uuid().hex}"

    @computed_field
    def unit_dimension_cpp(self) -> str:
        """Render the 7-vector as a C++ brace-initializer for ``getUnitDimension()``."""
        return "{" + ", ".join(_format_exponent(exp) for exp in self._exponents()) + "}"

    @computed_field
    def get_unit_cpp(self) -> str:
        if self.unit_factor is not None:
            return self.unit_factor
        return _unit_monomial(self._exponents())

    def _exponents(self) -> list[float]:
        return list(self.unit_dimension.unit_dimension) if self.unit_dimension is not None else [0.0] * 7

    def has_species(self) -> bool:
        """Whether this functor is registered for at least one species (i.e. is a
        reusable particle filter that can be narrowed by species)."""
        return bool(self.species_names)

    @computed_field
    def species_eligibility(self) -> str | None:
        """C++ expression true for exactly the species this functor is registered for.

        An OR over one ``is_same_v<GetCTName_t<T_Species>, PMACC_CSTRING(name)>`` per
        species, joined with ``||`` (single species yields no operator). ``None`` when not
        a reusable filter (empty ``species_names``), so the template keeps the primary
        ``SpeciesEligibleForSolver`` (eligible for all species). Precomputed here rather
        than in the template because moosetash has no list-index variable to suppress a
        leading ``||`` for the first element."""
        if not self.species_names:
            return None
        return " || ".join(
            f'std::is_same_v<pmacc::traits::GetCTName_t<T_Species>, PMACC_CSTRING("{name}")>'
            for s in self.species_names
            for name in (s["name"] if isinstance(s, dict) else s.name,)
        )

    @model_validator(mode="after")
    def _validate(self):
        if "int" in self.return_type:
            if self.unit_dimension == UnitDimension():
                self.unit_dimension = None
            if self.unit_dimension is not None:
                raise ValueError(
                    f"unit_dimension is not supported for integral types. You gave {self.unit_dimension=}."
                )
        # Validate up front that `getUnit()` can be auto-derived (or an explicit
        # `unit_factor` was given), so a non-derivable dimension is rejected at
        # construction rather than surfacing a wrong value at render time.
        self.get_unit_cpp
        if self.needs_total_position and self.rng_info is not None:
            raise ValueError(
                f"PIConGPU does not support particle functors that need total position and random numbers. You gave: {self.rng_info=}."
            )
        return self
