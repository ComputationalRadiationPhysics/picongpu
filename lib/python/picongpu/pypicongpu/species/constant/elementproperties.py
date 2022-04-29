"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
License: GPLv3+
"""

from .constant import Constant
from ... import util
from ..util import Element
from typeguard import typechecked
import typing


@typechecked
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
