"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant import ElementProperties

import unittest

from picongpu.pypicongpu.species.util import Element
import typeguard


class TestElementProperties(unittest.TestCase):
    def test_basic(self):
        """basic operation"""
        ep = ElementProperties()
        ep.element = Element.H

        ep.check()

        # has no dependencies
        self.assertEqual([], ep.get_species_dependencies())
        self.assertEqual([], ep.get_attribute_dependencies())
        self.assertEqual([], ep.get_constant_dependencies())

    def test_rendering(self):
        """members are exposed"""
        ep = ElementProperties()
        ep.element = Element.N

        context = ep.get_rendering_context()

        self.assertEqual(ep.element.get_rendering_context(), context["element"])

    def test_mandatory(self):
        """element is required"""
        ep = ElementProperties()

        with self.assertRaises(Exception):
            ep.check()

        ep.element = Element.H

        # now passes
        ep.check()

    def test_typesafety(self):
        """typesafety is ensured"""
        ep = ElementProperties()

        for invalid in [None, "H", 1, [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                ep.element = invalid
