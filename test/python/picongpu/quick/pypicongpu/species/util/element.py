"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.util import Element

import scipy

import unittest
from picongpu.pypicongpu.rendering import RenderedObject


class TestElement(unittest.TestCase):
    def setUp(self):
        # create test case data
        self.test_element = ["H", "#2H", "Cu", "#12C", "C"]
        self.name = ["H", "D", "Cu", "C", "C"]
        self.picongpu_names = ["Hydrogen", "Deuterium", "Copper", "Carbon", "Carbon"]
        self.mass = [
            1.00794 * scipy.constants.atomic_mass,
            2.014101778 * scipy.constants.atomic_mass,
            63.546 * scipy.constants.atomic_mass,
            12.0 * scipy.constants.atomic_mass,
            12.0107 * scipy.constants.atomic_mass,
        ]
        self.charge = [
            1.0 * scipy.constants.elementary_charge,
            1.0 * scipy.constants.elementary_charge,
            27.0 * scipy.constants.elementary_charge,
            6.0 * scipy.constants.elementary_charge,
            6.0 * scipy.constants.elementary_charge,
        ]
        self.atomic_number = [1, 1, 29, 6, 6]

    def test_parse_openpmd(self):
        valid_test_strings = ["#3H", "#15He", "#1H", "#3He", "#56Cu"]
        mass_number_results = [3, 15, 1, 3, 56]
        name_results = ["H", "He", "H", "He", "Cu"]

        for i, string in enumerate(valid_test_strings):
            mass_number, name = Element.parse_openpmd_isotopes(string)
            self.assertEqual(name, name_results[i])
            self.assertEqual(mass_number, mass_number_results[i])

        invalid_test_strings = ["#Htest", "#He3", "#Cu-56", "H3", "Fe-56"]
        for i, string in enumerate(invalid_test_strings):
            with self.assertRaisesRegex(ValueError, string + " is not a valid openPMD isotope descriptor"):
                name, massNumber = Element.parse_openpmd_isotopes(string)

    def test_basic_use(self):
        for name in self.test_element:
            Element(name)

    def test_symbol(self):
        for openpmd_name, name in zip(self.test_element, self.name):
            e = Element(openpmd_name)
            self.assertEqual(e.get_symbol(), name)

    def test_is_element(self):
        for name in self.test_element:
            self.assertTrue(Element.is_element(name))
        self.assertFalse(Element.is_element("n"))

    def test_picongpu_names(self):
        """names must be translateable to picongpu"""
        for openpmd_name, picongpu_name in zip(self.test_element, self.picongpu_names):
            name_test = Element(openpmd_name).get_picongpu_name()
            self.assertNotEqual("", picongpu_name)
            self.assertEqual(name_test, picongpu_name)

    def test_get_mass(self):
        """all elements have mass"""
        for openpmd_name, mass in zip(self.test_element, self.mass):
            self.assertAlmostEqual(Element(openpmd_name).get_mass_si(), mass)

    def test_charge(self):
        """all elements have charge"""
        for openpmd_name, charge in zip(self.test_element, self.charge):
            self.assertAlmostEqual(Element(openpmd_name).get_charge_si(), charge)

    def test_atomic_number(self):
        for openpmd_name, atomic_number in zip(self.test_element, self.atomic_number):
            e = Element(openpmd_name)
            self.assertEqual(e.get_atomic_number(), atomic_number)

    def test_rendering(self):
        """all elements can be rendered"""
        self.assertTrue(issubclass(Element, RenderedObject))
        for openpmd_name in self.test_element:
            e = Element(openpmd_name)
            context = e.get_rendering_context()
            self.assertEqual(context["symbol"], e.get_symbol())
            self.assertEqual(context["picongpu_name"], e.get_picongpu_name())
