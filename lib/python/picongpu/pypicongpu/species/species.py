"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ..rendering import RenderedObject
from .attribute import Attribute, Position, Momentum
from .constant import Constant, Charge, Mass, DensityRatio, Ionizers, ElementProperties
from .. import util

import typeguard
import typing
import re


@typeguard.typechecked
class Species(RenderedObject):
    """
    PyPIConGPU species definition

    A "species" is a set of particles, which is defined by:

    - A set of species constants (mass, charge, etc.),
    - a set of species attributes (position, number of bound electrons), and
    - a set of operations which collectively initialize these attributes,
      where one attribute is initializated by exactly one operation.
    - (and a name)

    Note that some of the species attributes or constants are considered
    mandatory. Each species constant or attribute may only be defined once.
    """

    constants = util.build_typesafe_property(typing.List[Constant])
    """PIConGPU particle flags"""

    attributes = util.build_typesafe_property(typing.List[Attribute])
    """PIConGPU particle attributes"""

    name = util.build_typesafe_property(str)
    """name of the species"""

    def get_cxx_typename(self) -> str:
        """
        get (standalone) C++ name for this species
        """
        # ensures only safe names
        self.check()

        return "species_" + self.name

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
            raise ValueError("species names must be c++ compatible " "([A-Za-z0-9_]+)")

        # position is mandatory attribute
        # position
        if Position not in [type(a) for a in self.attributes]:
            raise ValueError("Each species must have the position attribute!")
        # momentum
        if Momentum not in [type(a) for a in self.attributes]:
            raise ValueError("Each species must have the momentum attribute!")

        # all constants check()'s pass
        for const in self.constants:
            const.check()

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
        attr_names = list(map(lambda attr: attr.PICONGPU_NAME, self.attributes))
        non_unique_attributes = set([c for c in attr_names if attr_names.count(c) > 1])
        if 0 != len(non_unique_attributes):
            raise ValueError(
                "attribute names must be unique per species, " "offending: {}".format(", ".join(non_unique_attributes))
            )

    def get_constant_by_type(self, needle_type: typing.Type[Constant]) -> Constant:
        """
        retrieve constant of given type, raise if not found

        Searches through constants of this species and returns the constant of
        the given type if found. If no constant of this type is found, an error
        is raised.

        :param needle_type: constant type to look for
        :raise RuntimeError: on failure to find constant of given type
        :return: constant of given type
        """
        for const in self.constants:
            # note: check using type equality, because polymorphy messes with
            # duplicate detection & rendering
            if needle_type == type(const):
                return const

        raise RuntimeError("no constant of requested type available: {}".format(needle_type))

    def has_constant_of_type(self, needle_type: typing.Type[Constant]) -> bool:
        """
        lookup if constant of given type is present

        Searches through constants of this species and returns true iff a
        constant of the given type is present.

        :param needle_type: constant type to look for
        :return: whether constant of needle_type exists
        """

        constants_types = list(map(type, self.constants))
        return needle_type in constants_types

    def _get_serialized(self) -> dict:
        self.check()

        # Constants are rendered into sth like this:
        # {"mass": null, "charge": {OBJECT}, ...}
        # i.e. all constants are *always* defined, but they can be null
        #
        # rationale:
        # The templating engine does not consider "optional variables",
        # i.e. variables that are only present sometimes. It *always* suspects
        # a typo in the variable name in prints a warning (still continues
        # though -- to be compliant to the rendering standard).
        #
        # To accomodate this behavior, we always define all keys for constant,
        # but maybe set them to null. For this below there is a list of *all
        # known constants*. When adding a constant do not forget to add it in
        # the JSON schema too.
        #
        # Note: Attributes are only required as a set of strings for type
        # generation, so there is no similar treatment there.

        constant_names_by_type = {
            "mass": Mass,
            "charge": Charge,
            "density_ratio": DensityRatio,
            "ionizers": Ionizers,
            "element_properties": ElementProperties,
        }

        constants_context = {}
        for constant_name, constant_type in constant_names_by_type.items():
            if self.has_constant_of_type(constant_type):
                constants_context[constant_name] = self.get_constant_by_type(constant_type).get_rendering_context()
            else:
                constants_context[constant_name] = None

        return {
            "name": self.name,
            "typename": self.get_cxx_typename(),
            "attributes": list(map(lambda attr: {"picongpu_name": attr.PICONGPU_NAME}, self.attributes)),
            "constants": constants_context,
        }
