"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .constant import Constant
from ... import util
from ..util import Element

import typeguard
import typing


@typeguard.typechecked
class ElementProperties(Constant):
    """
    represents constants associated to a chemical element

    Produces PIConGPU atomic number and ionization energies.

    Note: Not necessarily all of the generated properties will be required
    during runtime. However, this is left to the compiler to optimize (which
    is a core concept of PIConGPU).
    """

    element = util.build_typesafe_property(Element)
    """represented chemical element"""

    def __init__(self):
        pass

    def check(self):
        # note: typecheck handled by property itself
        assert self.element is not None

    def get_species_dependencies(self):
        return []

    def get_attribute_dependencies(self) -> typing.List[type]:
        return []

    def get_constant_dependencies(self) -> typing.List[type]:
        return []

    def _get_serialized(self) -> dict:
        return {
            "element": self.element.get_rendering_context(),
        }
