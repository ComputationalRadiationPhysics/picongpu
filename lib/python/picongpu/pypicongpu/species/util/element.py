"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ...rendering import RenderedObject

import typeguard
import enum
import scipy


@typeguard.typechecked
class Element(RenderedObject, enum.Enum):
    """
    Denotes an element from the periodic table of elements

    Used to provide fundamental constants for elements, and to map them in a
    type-safe way to PIConGPU.

    The number associated is the number of protons.
    Note: Spelling follows periodic table, e.g. "Na", "C", "He"

    Note that these denote Elements, but when initialized in a species *only*
    represent the core, i.e. there are no electrons. To make an atom also
    initialize an appropriate ionization.
    """

    H = 1
    """hydrogen"""
    He = 2
    """helium"""
    N = 7
    """nitrogen"""

    @staticmethod
    def get_by_openpmd_name(openpmd_name: str) -> "Element":
        """
        get the correct substance implementation from a openPMD type name

        Names are (case-sensitive) element symbols (e.g. "H", "He", "N").

        :param openpmd_name: single species following openPMD species extension
        :return: object representing the given species
        """
        element_by_openpmd_name = {
            "H": Element.H,
            "He": Element.He,
            "N": Element.N,
        }
        if openpmd_name not in element_by_openpmd_name:
            raise NameError("unkown element: {}".format(openpmd_name))
        return element_by_openpmd_name[openpmd_name]

    def get_picongpu_name(self) -> str:
        """
        get name as used in PIConGPU

        Used for type name lookups
        """
        picongpu_name_by_element = {
            Element.H: "Hydrogen",
            Element.He: "Helium",
            Element.N: "Nitrogen",
        }
        return picongpu_name_by_element[self]

    def get_mass_si(self) -> float:
        """
        Get mass of an individual particle

        Calculated based of "conventional atomic weight" from the periodic
        table of elements

        :return: mass in kg
        """
        mass_by_particle = {
            Element.H: 1.008 * scipy.constants.atomic_mass,
            Element.He: 4.0026 * scipy.constants.atomic_mass,
            Element.N: 14.007 * scipy.constants.atomic_mass,
        }
        return mass_by_particle[self]

    def get_charge_si(self) -> float:
        """
        Get charge of an individual particle *without* electrons

        Calculated from atomic number, applies with no electrons present.

        :return: charge in C
        """
        return self.value * scipy.constants.elementary_charge

    def _get_serialized(self) -> dict:
        return {
            "symbol": self.name,
            "picongpu_name": self.get_picongpu_name(),
        }
