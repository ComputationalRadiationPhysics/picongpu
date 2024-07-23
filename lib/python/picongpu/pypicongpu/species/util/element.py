"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ...rendering import RenderedObject
from ... import util

import scipy
import periodictable


class Element(RenderedObject):
    """
    Denotes an element from the periodic table of elements

    Used to provide fundamental constants for elements, and to map them in a
    type-safe way to PIConGPU.

    The number associated is just an id.
    Note: Spelling follows periodic table, e.g. "Na", "C", "He" + typical nuclear variations

    Note that these denote Elements, but when initialized in a species *only*
    describe the core, i.e. without electrons.
    To describe atoms/ions you also need to initialize the charge_state of the species.
    """

    store = util.build_typesafe_propety(periodictable.Element)

    def __init__(self, openpmd_name: str) -> None:
        """
        get the correct substance implementation from a openPMD type name

        @param openpmd_name (case-sensitive) chemical/nuclear element symbols (e.g. "H", "D", "He", "N").

        :return: object representing the given species
        """

        # search for name in periodic table
        for element in periodictable.elements:
            if openpmd_name == element.symbol:
                self.store = element
                return

        # not found
        raise NameError("unkown element: {}".format(openpmd_name))

    def get_picongpu_name(self) -> str:
        """
        get name as used in PIConGPU

        Used for type name lookups
        """
        return self.store.name

    def get_mass_si(self) -> float:
        """
        Get mass of an individual particle

        Calculated based of "conventional atomic weight" from the periodic
        table of elements

        :return: mass in kg
        """
        return self.store.mass * scipy.constants.atomic_mass

    def get_charge_si(self) -> float:
        """
        Get charge of an individual particle *without* electrons

        Calculated from atomic number, applies with no electrons present.

        :return: charge in C
        """
        return self.ions[-1] * scipy.constants.elementary_charge

    def get_symbol(self) -> str:
        """get symbol"""
        return self.store.symbol

    def _get_serialized(self) -> dict:
        return {
            "symbol": self.get_symbol(),
            "picongpu_name": self.get_picongpu_name(),
        }
