"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant import Charge

import unittest
import typeguard


class TestCharge(unittest.TestCase):
    def test_basic(self):
        c = Charge()
        c.charge_si = 0
        c.check()
        self.assertEqual([], c.get_species_dependencies())
        self.assertEqual([], c.get_attribute_dependencies())
        self.assertEqual([], c.get_constant_dependencies())

    def test_types(self):
        """types are checked"""
        c = Charge()
        for invalid in [None, [], {}, "1"]:
            with self.assertRaises(typeguard.TypeCheckError):
                c.charge_si = invalid

    def test_rendering(self):
        """rendering passes information through"""
        c = Charge()
        c.charge_si = -3.2

        context = c.get_rendering_context()
        self.assertAlmostEqual(-3.2, context["charge_si"])
