"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import re
from pydantic import BaseModel, computed_field, field_validator
from enum import Enum

from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronConstant

from ..rendering import RenderedObject
from .attribute import Attribute, Momentum, Position
from .constant import (
    Charge,
    Constant,
    DensityRatio,
    ElementProperties,
    GroundStateIonization,
    Mass,
)


class Shape(Enum):
    NGP = "NGP"
    linear = "CIC"
    quadratic = "TSC"
    cubic = "PQS"
    quartic = "PCS"
    counter = "Counter"


class Pusher(Enum):
    # supported by standard and PIConGPU
    Boris = "Boris"
    Vay = "Vay"
    Higuera = "Higuera-Cary"
    Free = "Free"
    # not supported by standard
    ReducedLandauLifshitz = "ReducedLandauLifshitz"
    Acceleration = "Acceleration"
    Photon = "Photon"
    Probe = "Probe"
    Axel = "Axel"


class Constants(BaseModel):
    mass: Mass | None
    charge: Charge | None
    density_ratio: DensityRatio | None
    element_properties: ElementProperties | None
    ground_state_ionization: GroundStateIonization | None
    synchrotron: SynchrotronConstant | None


_CONSTANT_TYPES_BY_FIELD = {
    "mass": Mass,
    "charge": Charge,
    "density_ratio": DensityRatio,
    "element_properties": ElementProperties,
    "ground_state_ionization": GroundStateIonization,
    "synchrotron": SynchrotronConstant,
}
_CONSTANT_FIELDS_BY_TYPE = {constant_type: field for field, constant_type in _CONSTANT_TYPES_BY_FIELD.items()}
_CONSTANT_FIELDS = tuple(_CONSTANT_TYPES_BY_FIELD)


def _canonical_constants(value):
    """
    normalize a raw ``constants`` value to the canonical form

    The canonical form is a dict mapping each ``Constants`` field name to the
    corresponding ``Constant`` (or ``None`` when absent). ``constants`` may
    arrive as a dict of ``field -> Constant | None`` (direct pypicongpu
    construction), a list/tuple of ``Constant`` (PICMI path), or a ``Constants``
    model instance; all are reduced to this single form so that the constant
    helpers below only ever operate on one representation.
    """
    if isinstance(value, Constants):
        return {field: getattr(value, field) for field in _CONSTANT_FIELDS}
    if isinstance(value, dict):
        unknown = sorted(set(value) - set(_CONSTANT_FIELDS))
        if unknown:
            raise ValueError(
                "unknown species constant name(s): {}; valid names are: {}".format(
                    ", ".join(unknown), ", ".join(_CONSTANT_FIELDS)
                )
            )
        return {field: value.get(field) for field in _CONSTANT_FIELDS}

    canonical = {field: None for field in _CONSTANT_FIELDS}
    for const in value:
        if const is None:
            continue
        field = _CONSTANT_FIELDS_BY_TYPE.get(type(const))
        if field is not None:
            # duplicate constants of the same type are invalid; last one wins
            canonical[field] = const
    return canonical


def has_constant_of_type(constants, needle_type: type[Constant]) -> bool:
    """
    lookup if constant of given type is present

    Searches through the canonical species constants (a dict of
    ``field -> Constant | None``) and returns true if a constant of the
    given type is present.

    :param constants: canonical species constants (see ``_canonical_constants``)
    :param needle_type: constant type to look for
    :return: whether constant of needle_type exists
    """

    return any(needle_type is type(const) for const in constants.values() if const is not None)


def get_constant_by_type(constants, needle_type: type[Constant]) -> Constant:
    """
    retrieve constant of given type, raise if not found

    Searches through the canonical species constants (a dict of
    ``field -> Constant | None``) and returns the constant of the given type
    if found. If no constant of this type is found, an error is raised.

    :param constants: canonical species constants (see ``_canonical_constants``)
    :param needle_type: constant type to look for
    :raise RuntimeError: on failure to find constant of given type
    :return: constant of given type
    """
    for const in constants.values():
        # note: check using type equality, because polymorphy messes with
        # duplicate detection & rendering
        if const is not None and needle_type is type(const):
            return const

    raise RuntimeError("no constant of requested type available: {}".format(needle_type))


class Species(RenderedObject, BaseModel):
    """
    PyPIConGPU species definition

    A "species" is a set of particles, which is defined by:

    - A set of species constants (mass, charge, etc.),
    - a set of species attributes (position, number of bound electrons), and
    - a set of operations which collectively initialize these attributes,
      where one attribute is initialized by exactly one operation.
    - (and a name)

    Note that some of the species attributes or constants are considered
    mandatory. Each species constant or attribute may only be defined once.
    """

    constants: Constants
    """PIConGPU particle flags"""

    attributes: list[Attribute]
    """PIConGPU particle attributes"""

    pusher: Pusher = Pusher["Boris"]

    name: str
    """name of the species"""

    shape: Shape = Shape("TSC")

    @computed_field
    def species_name(self) -> str:
        return self.name

    @computed_field
    def filter_name(self) -> str:
        return "all"

    @computed_field
    def filter_typename(self) -> str:
        return "All"

    @computed_field
    def typename(self) -> str:
        """
        get (standalone) C++ name for this species
        """
        return "species_" + self.name

    def __hash__(self):
        # species must be uniquely defined by name
        return hash(self.name)

    def check(self) -> None:
        """
        sanity-check self, if ok pass silently

        Ensure that:

        - species has valid name
        - constants have unique types
        - attributes have unique types
        """

        # name c++ compatible
        # quick excursion to re.[match, fullmatch, search]:
        # - re.search: match *anywhere* in the string
        # - re.match: match *full* string, but ignore trailing newline (WTF?)
        #   -> "abc\n" would be accepted (despite "$" at the end)
        # - re.fullmatch: match *actually* full string
        #   -> "abc\n" is rejected
        if not re.fullmatch(r"^[A-Za-z0-9_]+$", self.name):
            raise ValueError("species names must be c++ compatible ([A-Za-z0-9_]+)")

        # position is mandatory attribute
        # position
        if Position not in [type(a) for a in self.attributes]:
            raise ValueError("Each species must have the position attribute!")
        # momentum, @todo really necessary?, Brian Marre, 2024
        if Momentum not in [type(a) for a in self.attributes]:
            raise ValueError("Each species must have the momentum attribute!")

        # each constant type can only be used once
        # self.constants is a Constants model; iterating it yields (field, value)
        # pairs, so inspect the actual constant objects (skipping absent ones).
        const_types = [type(const) for name, const in self.constants if const is not None]
        non_unique_constants = set([c for c in const_types if const_types.count(c) > 1])
        if 0 != len(non_unique_constants):
            raise ValueError(
                "constant names must be unique per species, offending: {}".format(
                    ", ".join(map(str, non_unique_constants))
                )
            )

        # each attribute (-name) can only be used once
        attr_names = list(map(lambda attr: attr.picongpu_name, self.attributes))
        non_unique_attributes = set([c for c in attr_names if attr_names.count(c) > 1])
        if 0 != len(non_unique_attributes):
            raise ValueError(
                "attribute names must be unique per species, offending: {}".format(", ".join(non_unique_attributes))
            )

    @field_validator("constants", mode="before")
    @classmethod
    def constants_context(cls, value):
        # normalize the raw input (dict / list / Constants model) to the
        # canonical form, then look the constants up through the helpers
        constants = _canonical_constants(value)
        constants_context = {}
        for constant_name, constant_type in _CONSTANT_TYPES_BY_FIELD.items():
            if has_constant_of_type(constants, constant_type):
                constants_context[constant_name] = get_constant_by_type(constants, constant_type)
            else:
                constants_context[constant_name] = None

        return Constants(**constants_context)
