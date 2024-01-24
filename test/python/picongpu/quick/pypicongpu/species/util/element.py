"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.util import Element

import unittest
from picongpu.pypicongpu.rendering import RenderedObject
import re
import typeguard


class TestElement(unittest.TestCase):
    def test_exists(self):
        """there is at least one element"""
        self.assertNotEqual([], list(Element))

    def test_openpmd_names(self):
        """elements can be requested by openPMD name"""
        expected_element_by_name = {
            "H": Element.H,
            "He": Element.He,
            "N": Element.N,
        }
        for name, element in expected_element_by_name.items():
            self.assertEqual(element, Element.get_by_openpmd_name(name))

        for invalid_type in [[], None, 3]:
            with self.assertRaises(typeguard.TypeCheckError):
                Element.get_by_openpmd_name(invalid_type)

        for unknown_name in ["", " H", "abc"]:
            with self.assertRaisesRegex(NameError, ".*unkown.*"):
                Element.get_by_openpmd_name(unknown_name)

    def test_periodic_table_names(self):
        """names must follow the periodic table"""
        element_re = re.compile("^[A-Z][a-z]?$")
        for element in list(Element):
            self.assertTrue(element_re.match(element.name))

    def test_picongpu_names(self):
        """names must be translateable to picongpu"""
        all_picongpu_names = set()
        # all elements are defined
        for element in list(Element):
            picongpu_name = element.get_picongpu_name()
            self.assertNotEqual("", picongpu_name)
            self.assertTrue(picongpu_name not in all_picongpu_names)
            all_picongpu_names.add(picongpu_name)

    def test_mass(self):
        """all elements have mass"""
        for element in list(Element):
            self.assertTrue(0 < element.get_mass_si())

    def test_charge(self):
        """all elements have charge"""
        for element in list(Element):
            self.assertTrue(0 < element.get_charge_si())

    def test_rendering(self):
        """all elements can be rendered"""
        self.assertTrue(issubclass(Element, RenderedObject))
        for element in list(Element):
            context = element.get_rendering_context()
            self.assertEqual(context["symbol"], element.name)
            self.assertEqual(context["picongpu_name"], element.get_picongpu_name())
