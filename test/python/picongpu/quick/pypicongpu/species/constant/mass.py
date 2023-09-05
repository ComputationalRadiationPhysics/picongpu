"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant import Mass

import unittest
import typeguard


class TestMass(unittest.TestCase):
    def test_basic(self):
        m = Mass()
        m.mass_si = 17
        # passes
        m.check()
        self.assertEqual([], m.get_species_dependencies())
        self.assertEqual([], m.get_attribute_dependencies())
        self.assertEqual([], m.get_constant_dependencies())

    def test_mandatory(self):
        m = Mass()
        # mass_si not set -> raises:
        with self.assertRaises(Exception):
            m.check()

    def test_type(self):
        """types are checked"""
        m = Mass()
        for invalid in [None, "1", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                m.mass_si = invalid

    def test_values(self):
        """invalid values are rejected"""
        m = Mass()
        for invalid in [-1, 0, -0.0000001]:
            m.mass_si = invalid
            with self.assertRaises(ValueError):
                m.check()

    def test_rendering(self):
        """passes value through"""
        m = Mass()
        m.mass_si = 1337

        context = m.get_rendering_context()
        self.assertEqual(1337, context["mass_si"])

        m.mass_si = -1
        with self.assertRaises(ValueError):
            m.get_rendering_context()
