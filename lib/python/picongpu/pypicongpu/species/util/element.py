"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ...rendering import RenderedObject

import pydantic
import typeguard
import typing
import scipy
import periodictable
import re


@typeguard.typechecked
class Element(RenderedObject, pydantic.BaseModel):
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

    _store: typing.Optional[periodictable.core.Element] = None

    @staticmethod
    def parse_openpmd_isotopes(openpmd_name: str) -> tuple[int | None, str]:
        if openpmd_name == "":
            raise ValueError(f"{openpmd_name} is not a valid openPMD particle type")
        if openpmd_name[0] != "#" and re.match(r"[A-Z][a-z]?$|n$", openpmd_name):
            return None, openpmd_name

        m = re.match(r"#([1-9][0-9]*)([A-Z][a-z]?)$", openpmd_name)

        if m is None:
            raise ValueError(f"{openpmd_name} is not a valid openPMD particle type")

        mass_number = int(m.group(1))
        symbol = m.group(2)

        return mass_number, symbol

    @staticmethod
    def is_element(openpmd_name: str) -> bool:
        """does openpmd_name describe an element?"""
        mass_number, symbol = Element.parse_openpmd_isotopes(openpmd_name)

        for element in periodictable.elements:
            if symbol == element.symbol:
                if openpmd_name not in ["n"]:
                    return True
        return False

    def __init__(self, openpmd_name: str) -> None:
        """
        get the correct substance implementation from a openPMD type name

        @param openpmd_name (case-sensitive) chemical/nuclear element symbols (e.g. "H", "D", "He", "N").

        :return: object representing the given species
        """
        pydantic.BaseModel.__init__(self)

        mass_number, openpmd_name = Element.parse_openpmd_isotopes(openpmd_name)

        found = False
        # search for name in periodic table
        for element in periodictable.elements:
            if openpmd_name == element.symbol:
                if mass_number is None:
                    self._store = element
                else:
                    self._store = element[mass_number]
                found = True

        if not found:
            raise NameError(f"unknown element: {openpmd_name}")

    def get_picongpu_name(self) -> str:
        """
        get name as used in PIConGPU

        Used for type name lookups
        """
        name = self._store.name
        # element names are capitalized in piconpgu
        return name[0].upper() + name[1:]

    def get_mass_si(self) -> float:
        """
        Get mass of an individual particle

        Calculated based of "conventional atomic weight" from the periodic
        table of elements

        :return: mass in kg
        """
        return self._store.mass * scipy.constants.atomic_mass

    def get_charge_si(self) -> float:
        """
        Get charge of an individual particle *without* electrons

        Calculated from atomic number, applies with no electrons present.

        :return: charge in C
        """
        return self._store.ions[-1] * scipy.constants.elementary_charge

    def get_atomic_number(self) -> int:
        return self._store.number

    def get_symbol(self) -> str:
        """get symbol"""
        return self._store.symbol

    def _get_serialized(self) -> dict:
        return {
            "symbol": self.get_symbol(),
            "picongpu_name": self.get_picongpu_name(),
        }
