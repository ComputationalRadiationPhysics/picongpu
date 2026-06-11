"""
# SPDX-FileCopyrightText: Hannes Troepgen, Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import re
from pydantic import BaseModel, computed_field, field_validator
import typing
from enum import Enum

import typeguard

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
    CIC = "CIC"
    COUNTER = "Counter"
    NGP = "NGP"
    PCS = "PCS"
    PQS = "PQS"
    TSC = "TSC"


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


def has_constant_of_type(constants, needle_type: typing.Type[Constant]) -> bool:
    """
    lookup if constant of given type is present

    Searches through constants of this species and returns true if a
    constant of the given type is present.

    :param needle_type: constant type to look for
    :return: whether constant of needle_type exists
    """

    constants_types = list(map(type, constants))
    return needle_type in constants_types


def get_constant_by_type(constants, needle_type: typing.Type[Constant]) -> Constant:
    """
    retrieve constant of given type, raise if not found

    Searches through constants of this species and returns the constant of
    the given type if found. If no constant of this type is found, an error
    is raised.

    :param needle_type: constant type to look for
    :raise RuntimeError: on failure to find constant of given type
    :return: constant of given type
    """
    for const in constants:
        # note: check using type equality, because polymorphy messes with
        # duplicate detection & rendering
        if needle_type is type(const):
            return const

    raise RuntimeError("no constant of requested type available: {}".format(needle_type))


@typeguard.typechecked
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

    shape: Shape = Shape["TSC"]

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
        const_types = list(map(type, self.constants))
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
        constant_names_by_type = {
            "mass": Mass,
            "charge": Charge,
            "density_ratio": DensityRatio,
            "element_properties": ElementProperties,
            "ground_state_ionization": GroundStateIonization,
            "synchrotron": SynchrotronConstant,
        }

        constants_context = {}
        for constant_name, constant_type in constant_names_by_type.items():
            if has_constant_of_type(value, constant_type):
                constants_context[constant_name] = get_constant_by_type(value, constant_type)
            else:
                constants_context[constant_name] = None

        return Constants(**constants_context)
