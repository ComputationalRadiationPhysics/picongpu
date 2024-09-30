"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .operation import Operation
from ..species import Species
from ..attribute import BoundElectrons
from ..constant import GroundStateIonization
from ... import util

import typeguard


@typeguard.typechecked
class SetChargeState(Operation):
    """
    assigns boundElectrons attribute and sets it to the initial charge state

    used for ionization of ions
    """

    species = util.build_typesafe_property(Species)
    """species which will have boundElectrons set"""

    charge_state = util.build_typesafe_property(int)
    """number of bound electrons to set"""

    def __init__(self):
        pass

    def check_preconditions(self) -> None:
        assert self.species.has_constant_of_type(GroundStateIonization), "BoundElectrons requires GroundStateIonization"

        if self.charge_state < 0:
            raise ValueError("charge state must be > 0")

        # may not check for charge_state > Z since Z not known in this context

    def prebook_species_attributes(self) -> None:
        self.attributes_by_species = {
            self.species: [BoundElectrons()],
        }

    def _get_serialized(self) -> dict:
        return {
            "species": self.species.get_rendering_context(),
            "charge_state": self.charge_state,
        }
