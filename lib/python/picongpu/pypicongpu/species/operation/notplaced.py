"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Tröpgen, Brian Edward Marré
License: GPLv3+
"""

from .operation import Operation
from ..species import Species
from ..attribute import Position, Weighting
from ... import util
from typeguard import typechecked


@typechecked
class NotPlaced(Operation):
    """
    assigns a position attribute, but does not place a species

    Intended for electrons which do not have a profile, but are used in
    pre-ionization.

    Provides attributes Position & Weighting.
    """

    species = util.build_typesafe_property(Species)
    """species which will not be placed"""

    def __init__(self):
        pass

    def check_preconditions(self) -> None:
        # retrieve species one to ensure it is set:
        self.species

    def prebook_species_attributes(self) -> None:
        self.attributes_by_species = {
            self.species: [Position(), Weighting()],
        }

    def _get_serialized(self) -> dict:
        """
        should not be rendered (does nothing)

        Rationale: This only provides an attribute (via
        prebook_species_attributes()), but does not do anything with the
        generated attribute -> there is no code generated -> nothing to
        serialize
        """
        raise RuntimeError(
            "NotPlaced operation has no rendering context representation")
