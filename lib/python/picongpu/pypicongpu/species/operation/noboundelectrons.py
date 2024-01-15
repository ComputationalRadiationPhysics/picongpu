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
class NoBoundElectrons(Operation):
    """
    assigns a BoundElectrons attribute, but leaves it a 0

    Intended use for fully ionized ions, which do CAN be ionized.
    (Fully ionized ions which can NOT be ionized do not require a
    BoundElectrons attribute, and therefore no operation to assign it.)
    """

    species = util.build_typesafe_property(Species)
    """species which will have boundElectorns set to 0"""

    def __init__(self):
        pass

    def check_preconditions(self) -> None:
        assert self.species.has_constant_of_type(Ionizers), "BoundElectrons requires Ionizers"

    def prebook_species_attributes(self) -> None:
        self.attributes_by_species = {
            self.species: [BoundElectrons()],
        }

    def _get_serialized(self) -> dict:
        """
        should not be rendered (does nothing)

        Rationale: This only provides an attribute (via
        prebook_species_attributes()), but does not do anything with the
        generated attribute -> there is no code generated -> nothing to
        serialize
        """
        raise RuntimeError("NoBoundElectrons operation has no rendering " "context representation")
