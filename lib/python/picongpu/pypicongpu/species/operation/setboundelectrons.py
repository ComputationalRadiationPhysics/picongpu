"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .operation import Operation
from ..species import Species
from ..attribute import BoundElectrons
from ..constant import Ionizers
from ... import util
from typeguard import typechecked


@typechecked
class SetBoundElectrons(Operation):
    """
    assigns and set the boundElectrons attribute

    Standard attribute for pre-ionization.
    """

    species = util.build_typesafe_property(Species)
    """species which will have boundElectrons set"""

    bound_electrons = util.build_typesafe_property(int)
    """number of bound electrons to set"""

    def __init__(self):
        pass

    def check_preconditions(self) -> None:
        assert self.species.has_constant_of_type(Ionizers), \
            "BoundElectrons requires Ionizers"

        if self.bound_electrons < 0:
            raise ValueError("bound electrons must be >0")

        if 0 == self.bound_electrons:
            raise ValueError(
                "bound electrons must be >0, use NoBoundElectrons to assign "
                "0 bound electrons")

    def prebook_species_attributes(self) -> None:
        self.attributes_by_species = {
            self.species: [BoundElectrons()],
        }

    def _get_serialized(self) -> dict:
        return {
            "species": self.species.get_rendering_context(),
            "bound_electrons": self.bound_electrons,
        }
