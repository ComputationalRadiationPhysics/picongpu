"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.constant import DensityRatio

import unittest
import typeguard

from picongpu.pypicongpu.species.constant import Constant


class TestDensityRatio(unittest.TestCase):
    def test_preconditions(self):
        """is a constant, has picongpu name"""
        self.assertTrue(isinstance(DensityRatio(), Constant))

    def test_basic(self):
        """simple example"""
        dr = DensityRatio()
        dr.ratio = 1.0
        # passes
        dr.check()
        self.assertEqual([], dr.get_species_dependencies())
        self.assertEqual([], dr.get_attribute_dependencies())
        self.assertEqual([], dr.get_constant_dependencies())

    def test_types(self):
        """type safety ensured"""
        dr = DensityRatio()
        for invalid in [None, "asbd", [], {}]:
            with self.assertRaises(typeguard.TypeCheckError):
                dr.ratio = invalid

        # note: type ok, value might not -> not checked here
        for valid_type in [-3, -31.2, 1, 171238]:
            dr.ratio = valid_type

    def test_value_range(self):
        """negative values prohibited"""
        dr = DensityRatio()
        for invalid in [0, -1, -0.00000001]:
            dr.ratio = invalid
            with self.assertRaisesRegex(ValueError, ".*(zero|0).*"):
                dr.check()

        for valid in [0.000001, 2, 3.5]:
            dr.ratio = valid
            dr.check()

    def test_empty(self):
        """ratio is mandatory attr"""
        dr = DensityRatio()
        with self.assertRaises(Exception):
            dr.check()

        # now passes
        dr.ratio = 1
        dr.check()

    def test_rendering_checks(self):
        """rendering context invokes check"""
        dr = DensityRatio()
        dr.ratio = -1

        with self.assertRaises(ValueError):
            # expected
            dr.check()

        with self.assertRaises(ValueError):
            # should internally call check -> same error
            dr.get_rendering_context()

    def test_rendering_passthru(self):
        """context passes ratio through"""
        dr = DensityRatio()
        dr.ratio = 13.37

        context = dr.get_rendering_context()

        self.assertAlmostEqual(13.37, context["ratio"])
